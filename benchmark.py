import argparse
from typing import Callable, Dict, Tuple
import time
import numpy as np
import torch
from torch import nn
from kan import create_dataset
from kan import KAN as pyKAN
from efficient_kan import KAN as effKAN
from FourierKAN.fftKAN import NaiveFourierKANLayer


class MLP(nn.Module):
    def __init__(self, layers: Tuple[int, int, int], device: str):
        super().__init__()
        self.layer1 = nn.Linear(layers[0], layers[1], device=device)
        self.layer2 = nn.Linear(layers[1], layers[2], device=device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        x = nn.functional.sigmoid(x)
        return x
    

class FourierKAN(nn.Module):
    def __init__(self, layers: Tuple[int, int, int], gridsize: int, device: str):
        super().__init__()
        self.layer1 = NaiveFourierKANLayer(layers[0], layers[1], gridsize=gridsize).to(device)
        self.layer2 = NaiveFourierKANLayer(layers[1], layers[2], gridsize=gridsize).to(device)

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def benchmark(
        dataset: Dict[str, torch.Tensor],
        device: str,
        bs: int,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        model: nn.Module,
        reps: int
    ) -> Dict[str, float]:
    forward_times = []
    backward_times = []
    forward_mems = []
    backward_mems = []
    for k in range(1 + reps):
        train_id = np.random.choice(dataset['train_input'].shape[0], bs, replace=False)
        tensor_input = dataset['train_input'][train_id]
        tensor_input = tensor_input.to(device)

        tensor_output = dataset['train_label'][train_id]
        tensor_output = tensor_output.to(device)

        if device == 'cpu':
            t0 = time.time()
            pred = model(tensor_input)
            t1 = time.time()
            if k > 0:
                forward_times.append((t1 - t0) * 1000)
            train_loss = loss_fn(pred, tensor_output)
            t2 = time.time()
            train_loss.backward()
            t3 = time.time()
            if k > 0:
                backward_times.append((t3 - t2) * 1000)
        elif device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            pred = model(tensor_input)
            end.record()

            torch.cuda.synchronize()
            if k > 0:
                forward_times.append(start.elapsed_time(end))
                forward_mems.append(torch.cuda.max_memory_allocated())

            train_loss = loss_fn(pred, tensor_output)

            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            train_loss.backward()
            end.record()

            torch.cuda.synchronize()
            if k > 0:
                backward_times.append(start.elapsed_time(end))
                backward_mems.append(torch.cuda.max_memory_allocated())
    return {
        'forward': np.mean(forward_times),
        'backward': np.mean(backward_times),
        'forward-memory': np.mean(forward_mems) / (1024 ** 3),
        'backward-memory': np.mean(backward_mems) / (1024 ** 3),
    }


def save_results(t: Dict[str, Dict[str, float]], out_path: str):
    maxlen = np.max([len(k) for k in t.keys()])
    with open(out_path, 'w') as f:
        print(f"{' '*maxlen}  |  {'forward':>11}  |  {'backward':>11}  |  {'forward':>11}  |  {'backward':>11}  |  {'num params':>11}  |  {'num trainable params':>20}", file=f)
        print(f"{' '*maxlen}  |  {'forward':>11}  |  {'backward':>11}  |  {'forward':>11}  |  {'backward':>11}  |  {'num params':>11}  |  {'num trainable params':>20}")
        print('-'*130, file=f)
        print('-'*130)
        for key in t.keys():
            print(f"{key:<{maxlen}}  |  {t[key]['forward']:8.2f} ms  |  {t[key]['backward']:8.2f} ms  |  {t[key]['forward-memory']:8.2f} GB  |  {t[key]['backward-memory']:8.2f} GB  |  {t[key]['params']:>11}  |  {t[key]['train_params']:>20}", file=f)
            print(f"{key:<{maxlen}}  |  {t[key]['forward']:8.2f} ms  |  {t[key]['backward']:8.2f} ms  |  {t[key]['forward-memory']:8.2f} GB  |  {t[key]['backward-memory']:8.2f} GB  |  {t[key]['params']:>11}  |  {t[key]['train_params']:>20}")


def count_params(model: nn.Module) -> Tuple[int, int]:
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params, pytorch_total_params_train


def _create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', default='times.txt', type=str)
    parser.add_argument('--method', choices=['pykan', 'efficientkan', 'fourierkan', 'mlp', 'all'], type=str)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--inp-size', type=int, default=2, help='The dimension of the input variables.')
    parser.add_argument('--hid-size', type=int, default=50, help='The dimension of the hidden layer.')
    parser.add_argument('--reps', type=int, default=10, help='Number of times to repeat execution and average.')
    parser.add_argument('--just-cuda', action='store_true', help='Whether to only execute the cuda version.')
    return parser


def main():
    parser = _create_parser()
    args = parser.parse_args()

    f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
    dataset = create_dataset(
        f, 
        n_var=args.inp_size,
        ranges = [-1,1],
        train_num=1000, 
        test_num=1000,
        normalize_input=False,
        normalize_label=False,
        device='cpu',
        seed=0
    )
    loss_fn = lambda x, y: torch.mean((x - y) ** 2)
    
    res = {}
    if args.method == 'pykan':
        if not args.just_cuda:
            model = pyKAN(width=[args.inp_size, args.hid_size, 1], grid=5, k=3, seed=0)
            model.to('cpu')
            res['pykan-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['pykan-cpu']['params'], res['pykan-cpu']['train_params'] = count_params(model)
        model = pyKAN(width=[args.inp_size, args.hid_size, 1], grid=5, k=3, seed=0, device='cuda')  # For gpu pass device here
        res['pykan-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['pykan-gpu']['params'], res['pykan-gpu']['train_params'] = count_params(model)
    if args.method == 'efficientkan' or args.method == 'all':
        model = effKAN(layers_hidden=[args.inp_size, args.hid_size, 1], grid_size=5, spline_order=3)
        if not args.just_cuda:
            model.to('cpu')
            res['effkan-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['effkan-cpu']['params'], res['effkan-cpu']['train_params'] = count_params(model)
        model.to('cuda')
        res['effkan-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['effkan-gpu']['params'], res['effkan-gpu']['train_params'] = count_params(model)
    if args.method == 'fourierkan' or args.method == 'all':
        model = FourierKAN(layers=[args.inp_size, args.hid_size, 1], gridsize=5, device='cpu')
        if not args.just_cuda:
            res['fourierkan-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['fourierkan-cpu']['params'], res['fourierkan-cpu']['train_params'] = count_params(model)
        model.to('cuda')
        res['fourierkan-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['fourierkan-gpu']['params'], res['fourierkan-gpu']['train_params'] = count_params(model)
    if args.method == 'mlp' or args.method == 'all':
        model = MLP(layers=[args.inp_size, args.hid_size * 10, 1], device='cpu')
        if not args.just_cuda:
            res['mlp-cpu'] = benchmark(dataset, 'cpu', args.batch_size, loss_fn, model, args.reps)
            res['mlp-cpu']['params'], res['mlp-cpu']['train_params'] = count_params(model)
        model.to('cuda')
        res['mlp-gpu'] = benchmark(dataset, 'cuda', args.batch_size, loss_fn, model, args.reps)
        res['mlp-gpu']['params'], res['mlp-gpu']['train_params'] = count_params(model)

    save_results(res, args.output_path)

if __name__=='__main__':
    main()