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
    
}

torch::Tensor fkan_cuda_forward(
        torch::Tensor input,
        torch::Tensor fouriercoeffs,
        torch::Tensor bias) {
    torch::Tensor output = torch::zeros({input.size(0), fouriercoeffs.size(2)});

    const int threads = 1024;
    const dim3 blocks((1000 + threads - 1) / threads, threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cufkan_forward_cuda", ([&] {
        cufkan_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            fouriercoeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    }));

    return output;
}

template <typename scalar_t>
__global__ void cufkan_backward_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input, 
        torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_input,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> fouriercoeff, 
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> d_fouriercoeff, 
        const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias, 
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_bias,
        const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> grad_o) {
    
}

std::vector<torch::Tensor> fkan_cuda_backward(
        torch::Tensor grad_o,
        torch::Tensor input,
        torch::Tensor fouriercoeffs,
        torch::Tensor bias) {
    torch::Tensor d_input = torch::zeros({input.size(0), input.size(1)});
    torch::Tensor d_fouriercoeffs = torch::zeros({
        fouriercoeffs.size(0), 
        fouriercoeffs.size(1), 
        fouriercoeffs.size(2), 
        fouriercoeffs.size(3)
    });
    torch::Tensor d_bias = torch::zeros({bias.size(0)});

    const int threads = 1024;
    const dim3 blocks((1000 + threads - 1) / threads, threads);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "cufkan_backward_cuda", ([&] {
        cufkan_backward_kernel<scalar_t><<<blocks, threads>>>(
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            fouriercoeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            d_fouriercoeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            d_bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            grad_o.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    }));

    return {d_input, d_fouriercoeffs, d_bias};
}