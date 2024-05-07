# Kolmogorov Arnold Efficiency Benchmark

Given the current popularity of the Kolmogorov Arnold Networks and its critiques about efficiency, here is a repository that measures the current state of the efficiency of the implementations. It is true that the original implementation was painfully slow. But optimizations have been made to increase the efficiency up to several orders of magnitude.

## Usage

The main script is `benchmark.py`. You can check its parameters by running `python benchmark.py --help`. The result should be 

```
usage: benchmark.py [-h] [--output-path OUTPUT_PATH] [--method {pykan,efficientkan,fourierkan,mlp,all}]
                    [--batch-size BATCH_SIZE] [--inp-size INP_SIZE] [--hid-size HID_SIZE] [--reps REPS] [--just-cuda]

options:
  -h, --help            show this help message and exit
  --output-path OUTPUT_PATH
  --method {pykan,efficientkan,fourierkan,mlp,all}
  --batch-size BATCH_SIZE
  --inp-size INP_SIZE   The dimension of the input variables.
  --hid-size HID_SIZE   The dimension of the hidden layer.
  --reps REPS           Number of times to repeat execution and average.
  --just-cuda           Whether to only execute the cuda version.
```

The benchmark is simply a network with 1 hidden layer where you can vary the number of input and hidden neurons and the output is 1D. This basically covers all the cases you want to profile since any other model is just made of several layers of this form. 

The MLP uses 10 times the hidden size in order for the number of parameters to be roughly the same and therefore comparable. And the original implementation is not included in the 'all' comparison because it takes orders of magnitude more time to execute.


## Setup
### Automatic

In order to run the benchmarks yourself you first have to setup your environment. I have provided a .bat file you can click if you are a Windows user and .sh file you can run in the bash shell if you are a linux user. For Windows users the prerequisites are conda and git. For linux users the prerequisites are python3.10, venv and git. If this method is not working, below is an explanation of the steps required. I have tested them on Windows 11 and WSL 2 Ubuntu 22.04.

### Manual
The first thing is to clone the submodules. For that you can either clone everything the first time:

```
git clone --recurse-submodules https://github.com/Jerry-Master/KAN-benchmarking.git
```

If you didn't do that, then you can later do it with:

```
git submodule update --init --recursive
```

After that, you need an active python3.10 virtual environment. You can either create it with conda:

```
conda create --name kan python=3.10 -y
conda activate kan
```

Or with venv:

```
python3.10 -m venv .venv
source .venv/bin/activate
```

Then, the dependencies need to be installed. The most important is pytorch. Follow [pytorch official instructions](https://pytorch.org/get-started/locally/) for your system. Once that has finished, install the rest of the dependencies, both from pypi and from the submodules:

```
pip install matplotlib==3.6.2 numpy==1.24.4 scikit_learn==1.1.3 setuptools==65.5.0 sympy==1.11.1 tqdm==4.66.2
cd pykan
pip install -e .
cd ../efficient-kan
pip install -e .
```

To check everything is installed correctly run `python benchmark.py --help` and see there are no exception thrown.