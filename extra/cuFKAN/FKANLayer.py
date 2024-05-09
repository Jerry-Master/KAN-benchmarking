import torch
from torch import nn
import numpy as np
import cuFKAN_kernel


class FKANFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fouriercoeffs, bias):
        output = cuFKAN_kernel.forward(input, fouriercoeffs, bias)
        ctx.save_for_backward(input, fouriercoeffs, bias)
        return output

    @staticmethod
    def backward(ctx, grad_o):
        outputs = cuFKAN_kernel.backward(grad_o.contiguous(), *ctx.saved_tensors)
        d_input, d_fouriercoeffs, d_bias = outputs
        return d_input, d_fouriercoeffs, d_bias
    

class cuFKANFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fouriercoeffs, bias):
        output = cuFKAN_kernel.forward_gpu(input, fouriercoeffs, bias)
        ctx.save_for_backward(input, fouriercoeffs, bias)
        return output

    @staticmethod
    def backward(ctx, grad_o):
        outputs = cuFKAN_kernel.backward_gpu(grad_o.contiguous(), *ctx.saved_tensors)
        d_input, d_fouriercoeffs, d_bias = outputs
        return d_input, d_fouriercoeffs, d_bias


class FKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize):
        super(FKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim
        self.fouriercoeffs = nn.Parameter( 
            torch.randn(2, inputdim, outdim, gridsize) / 
                (np.sqrt(inputdim) * np.sqrt(gridsize))
        )
        self.bias  = nn.Parameter(torch.zeros(outdim))

    def forward(self, input):
        if input.get_device() == -1:
            return FKANFunction.apply(input, self.fouriercoeffs, self.bias)
        else:
            return cuFKANFunction.apply(input, self.fouriercoeffs, self.bias)