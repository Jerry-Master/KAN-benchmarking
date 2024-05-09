#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor fkan_cuda_forward(
    torch::Tensor input,
    torch::Tensor fouriercoeffs,
    torch::Tensor bias);

std::vector<torch::Tensor> fkan_cuda_backward(
    torch::Tensor grad_o,
    torch::Tensor input,
    torch::Tensor fouriercoeffs,
    torch::Tensor bias);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fkan_forward(
        torch::Tensor input,
        torch::Tensor fouriercoeffs,
        torch::Tensor bias) {
    CHECK_INPUT(input);
    CHECK_INPUT(fouriercoeffs);
    CHECK_INPUT(bias);

    return fkan_cuda_forward(input, fouriercoeffs,  bias);
}

std::vector<torch::Tensor> fkan_backward(
        torch::Tensor grad_o,
        torch::Tensor input,
        torch::Tensor fouriercoeffs,
        torch::Tensor bias) {
    CHECK_INPUT(grad_o);
    CHECK_INPUT(input);
    CHECK_INPUT(fouriercoeffs);
    CHECK_INPUT(bias);

    return fkan_cuda_backward(
        grad_o,
        input,
        fouriercoeffs,
        bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fkan_forward, "FKAN forward (CUDA)");
    m.def("backward", &fkan_backward, "FKAN backward (CUDA)");
}