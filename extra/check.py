from typing import Dict, Callable, Tuple
import torch
from torch import nn
import numpy as np
from kan import create_dataset
from fftKAN import NaiveFourierKANLayer
from cuFKAN import FKANLayer


class FourierKAN(nn.Module):
    def __init__(self, layers: Tuple[int, int, int], gridsize: int, device: str):
        super().__init__()
        torch.manual_seed(42)
        self.layer1 = NaiveFourierKANLayer(layers[0], layers[1], gridsize=gridsize).to(device)
        self.layer2 = NaiveFourierKANLayer(layers[1], layers[2], gridsize=gridsize).to(device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    

class FKAN(nn.Module):
    def __init__(self, layers: Tuple[int, int, int], gridsize: int, device: str):
        super().__init__()
        torch.manual_seed(42)
        self.layer1 = FKANLayer(layers[0], layers[1], gridsize=gridsize).to(device)
        self.layer2 = FKANLayer(layers[1], layers[2], gridsize=gridsize).to(device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    


def check(
        dataset: Dict[str, torch.Tensor], 
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        batch_size: int,
        inp_size: int,
        hid_size: int,
        device: str
    ):
    train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
    tensor_input = dataset['train_input'][train_id]
    tensor_input = tensor_input.to(device)

    tensor_output = dataset['train_label'][train_id]
    tensor_output = tensor_output.to(device)

    model1 = FourierKAN(layers=[inp_size, hid_size, 1], gridsize=5, device=device)
    model2 = FKAN(layers=[inp_size, hid_size, 1], gridsize=5, device=device)
    models_are_equal = True
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(param1, param2, atol=1e-4):
            models_are_equal = False
            break
    print('The model parameters coincide:', models_are_equal)

    pred1 = model1(tensor_input)
    pred2 = model2(tensor_input)
    predictions_are_equal = torch.allclose(pred1, pred2, atol=1e-4)
    print('Forward is correctly implemented:', predictions_are_equal)

    loss1 = loss_fn(pred1, tensor_output)
    loss2 = loss_fn(pred2, tensor_output)

    loss1.backward()
    loss2.backward()

    gradients_are_equal = True
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(param1.grad, param2.grad, atol=1e-4):
            gradients_are_equal = False
            break
    print('Backward is correctly implemented:', gradients_are_equal)


def main():
    inp_size = 1000
    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(
        f, 
        n_var=inp_size,
        ranges=[-1,1],
        train_num=1000, 
        test_num=1000,
        normalize_input=False,
        normalize_label=False,
        device='cpu',
        seed=0
    )
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    check(dataset, loss_fn, 100, inp_size, 1000, 'cpu')


if __name__=='__main__':
    main()