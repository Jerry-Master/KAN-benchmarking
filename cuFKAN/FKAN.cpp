#include <torch/extension.h>
#include <vector>

void fkan_forward_( 
    torch::TensorAccessor<float, 2> input, 
    torch::TensorAccessor<float, 4> fouriercoeffs, 
    torch::TensorAccessor<float, 1> bias, 
    torch::TensorAccessor<float, 2> output )
{
    for (int b = 0; b < input.size(0); b++) { // batch size
        for (int i = 0; i < fouriercoeffs.size(1); i++) { // input dimensions
            float x = input[b][i];
            float c0 = cosf(x); 
            float s0 = sinf(x);
            for (int j = 0; j < fouriercoeffs.size(2); j++) { // output dimensions
                float cos_partial = 1.0f;
                float sin_partial = 0.0f;
                for (int k = 0; k < fouriercoeffs.size(3); k++) { // grid size
                    float cos_term = cos_partial * c0-sin_partial * s0;
                    float sin_term = sin_partial * c0+cos_partial * s0;
                    cos_partial = cos_term;
                    sin_partial = sin_term;
                    output[b][j] += fouriercoeffs[0][i][j][k] * cos_term
                                 +  fouriercoeffs[1][i][j][k] * sin_term;
                }
            }
        }
    }
    for (int b = 0; b < input.size(0); b++)  // batch size
        for (int j = 0; j < fouriercoeffs.size(2); j++)  // output dimensions
            output[b][j] += bias[j];
}

torch::Tensor fkan_forward(
        torch::Tensor input,
        torch::Tensor fouriercoeffs,
        torch::Tensor bias) {
    torch::Tensor output = torch::zeros({input.size(0), fouriercoeffs.size(2)});
    fkan_forward_( 
        input.accessor<float, 2>(), 
        fouriercoeffs.accessor<float, 4>(), 
        bias.accessor<float, 1>(),
        output.accessor<float, 2>()
    );
    return output;
}

void fkan_backward_(
    torch::TensorAccessor<float, 2> input, 
    torch::TensorAccessor<float, 2> d_input,
    torch::TensorAccessor<float, 4> fouriercoeff, 
    torch::TensorAccessor<float, 4> d_fouriercoeff, 
    torch::TensorAccessor<float, 1> bias, 
    torch::TensorAccessor<float, 1> d_bias,  
    torch::TensorAccessor<float, 2> grad_o) {

    for(int i = 0; i < input.size(0); i++)
        for(int j = 0; j < fouriercoeff.size(2); j++)
            d_bias[j] = d_bias[j] + grad_o[i][j];

    for(int b = 0; b < input.size(0); b++)
        for(int i = 0; i < fouriercoeff.size(1); i++)
        {
            float x =  input[b][i];
            float c0 = cosf(x); 
            float s0 = sinf(x);
            for(int j = 0; j < fouriercoeff.size(2); j++)
            {
                float cos_partial = 1.0f;
                float sin_partial = 0.0f;
                for(int k = 0; k < fouriercoeff.size(3); k++)
                {
                    float cos_term = cos_partial * c0 - sin_partial * s0;
                    float sin_term = sin_partial * c0 + cos_partial * s0;
                    cos_partial = cos_term;
                    sin_partial = sin_term;
                    d_fouriercoeff[1][i][j][k] += sin_term * grad_o[b][j];
                    d_fouriercoeff[0][i][j][k] += cos_term * grad_o[b][j];

                    float d_cos = fouriercoeff[0][i][j][k] * grad_o[b][j];
                    float d_sin = fouriercoeff[1][i][j][k] * grad_o[b][j];
                    d_input[b][i] += (k + 1) * cos_term * d_sin - (k + 1) * sin_term * d_cos;
                }
            }
        }
}

std::vector<torch::Tensor> fkan_backward(
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
    fkan_backward_(
        input.accessor<float, 2>(), 
        d_input.accessor<float, 2>(),
        fouriercoeffs.accessor<float, 4>(), 
        d_fouriercoeffs.accessor<float, 4>(),
        bias.accessor<float, 1>(), 
        d_bias.accessor<float, 1>(),
        grad_o.accessor<float, 2>() );
    return {d_input, d_fouriercoeffs, d_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fkan_forward, "FKAN forward");
    m.def("backward", &fkan_backward, "FKAN backward");
}