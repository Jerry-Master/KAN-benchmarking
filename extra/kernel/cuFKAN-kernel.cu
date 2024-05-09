#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void cufkan_forward_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fouriercoeffs,
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias,
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output) {
    const int inputdim = input.size(1);
    const int bs = input.size(0);
    const int outputdim = output.size(1);
    const int gridsize = fouriercoeffs.size(3);

    for (int i = blockIdx.x; i < bs ; i += gridDim.x)
        for (int j = 0; j < inputdim; j++)
        {
            float xx =  input[i][j];
            float c0 = cosf(xx); 
            float s0 = sinf(xx);
            for (int l = threadIdx.x; l < outputdim; l += blockDim.x)
            {
                float ckm = 1.0f;
                float skm = 0.0f;
                for (int k = 1; k < gridsize + 1; k++)
                {
                    float c = ckm * c0 - skm * s0;
                    float s = skm * c0 + ckm * s0;
                    ckm = c;
                    skm = s;
                    output[i][l] += fouriercoeffs[0][j][l][k - 1] * c;
                    output[i][l] += fouriercoeffs[1][j][l][k - 1] * s;
                }
            }
        }

    for (int i = blockIdx.x; i < bs; i += gridDim.x)
        for (int l = threadIdx.x; l < outputdim; l += blockDim.x)
            output[i][l] += bias[l];
}

torch::Tensor fkan_cuda_forward(
        torch::Tensor input,
        torch::Tensor fouriercoeffs,
        torch::Tensor bias) {
    const int batch_size = input.size(0);
    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .layout(input.layout())
        .device(input.device().type(), input.device().index());
    torch::Tensor output = torch::zeros({batch_size, fouriercoeffs.size(2)}, options);

    const int n_threads = 32;
    const dim3 n_blocks(min(batch_size, 1024));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cufkan_forward_cuda", ([&] {
        cufkan_forward_kernel<scalar_t><<<n_blocks, n_threads>>>(
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            fouriercoeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    }));

    return output;
}
template <typename scalar_t>
__global__ void cufkan_backward_kernel_bias( 
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_bias,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_o) {
    const int bs = grad_o.size(0);
    const int outputdim = grad_o.size(1);

    int idx = threadIdx.x + blockDim.x * blockIdx.x; // create typical 1D thread index from built-in variables
    if (idx < outputdim)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < bs; i++)
            sum += grad_o[i][idx];  // write a for loop that will cause the thread to iterate down a column, keeeping a running sum, and write the result to sums
        d_bias[idx] += sum;
    }
}

template <typename scalar_t>
__global__ void cufkan_backward_kernel_coeff(
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input, 
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_input,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fouriercoeffs, 
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_fouriercoeffs,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_o) {
    const int inputdim = input.size(1);
    const int bs = input.size(0);
    const int outputdim = grad_o.size(1);
    const int gridsize = fouriercoeffs.size(3);

    int j = threadIdx.x + blockDim.x * blockIdx.x;
    int l = threadIdx.y + blockDim.y * blockIdx.y;
    if (j < inputdim && l < outputdim)
    {
        for( int i = 0 ; i < bs ; i++)
        {
            float xx = input[i][j];
            float c0 = cosf(xx); 
            float s0 = sinf(xx);
            float ckm = 1.0f;
            float skm = 0.0f;
            for (int k = 1; k < gridsize + 1; k++)
            {
                float xxb = 0.0;
                float c = ckm * c0 - skm * s0;
                float s = skm * c0 + ckm * s0;
                ckm = c;
                skm = s;
                float sb;
                float cb;
                d_fouriercoeffs[1][j][l][k - 1] += s * grad_o[i][l];
                d_fouriercoeffs[0][j][l][k - 1] += c * grad_o[i][l];
                sb = fouriercoeffs[1][j][l][k-1] * grad_o[i][l];
                cb = fouriercoeffs[0][j][l][k-1] * grad_o[i][l];
                xxb = k*c*sb - k*s*cb;
                d_input[i][j] += xxb;
            }
        }
    }
}

std::vector<torch::Tensor> fkan_cuda_backward(
        torch::Tensor grad_o,
        torch::Tensor input,
        torch::Tensor fouriercoeffs,
        torch::Tensor bias) {
    auto options = torch::TensorOptions()
        .dtype(input.dtype())
        .layout(input.layout())
        .device(input.device().type(), input.device().index());
    torch::Tensor d_input = torch::zeros({input.size(0), input.size(1)}, options);
    torch::Tensor d_fouriercoeffs = torch::zeros({
        fouriercoeffs.size(0), 
        fouriercoeffs.size(1), 
        fouriercoeffs.size(2), 
        fouriercoeffs.size(3)
    }, options);
    torch::Tensor d_bias = torch::zeros({bias.size(0)}, options);

    const int n_threads_bias = 1024;
    const int n_blocks_bias = (grad_o.size(1) + n_threads_bias - 1) / n_threads_bias;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cufkan_backward_kernel_bias", ([&] {
        cufkan_backward_kernel_bias<scalar_t><<<n_blocks_bias, n_threads_bias>>>(
            d_bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            grad_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    }));

    const dim3 n_threads_coeff(32, 32);
    const dim3 n_blocks_coeff(
        (input.size(1) + n_threads_coeff.x - 1) / n_threads_coeff.x,
        (grad_o.size(1) + n_threads_coeff.y - 1) / n_threads_coeff.y
    );
    

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cufkan_backward_kernel_coeff", ([&] {
        cufkan_backward_kernel_coeff<scalar_t><<<n_blocks_coeff, n_threads_coeff>>>(
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            fouriercoeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            d_fouriercoeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grad_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    }));

    return {d_input, d_fouriercoeffs, d_bias};
}