# Kolmogorov Arnold Efficiency Benchmark

Given the current popularity of the Kolmogorov Arnold Networks and its critiques about efficiency, here is a repository that measures the current state of the efficiency of the implementations. It is true that the original implementation was painfully slow. But optimizations have been made to increase the efficiency up to several orders of magnitude. Current best implementation in terms of efficiency seems to be [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN/tree/5f7efdd18e749bcc99481bd87dc90bdeafb920d8). And it seems there is still room for improvement [in that kernel](https://github.com/Jerry-Master/KAN-benchmarking/issues/4#issue-2297007801).

## Usage

The main script is `benchmark.py`. You can check its parameters by running `python benchmark.py --help`. The result should be 

```
usage: benchmark.py [-h] [--output-path OUTPUT_PATH] [--method {pykan,efficientkan,fourierkan,mlp,all}]
                    [--batch-size BATCH_SIZE] [--inp-size INP_SIZE] [--hid-size HID_SIZE] [--reps REPS] [--just-cuda]

options:
  -h, --help            show this help message and exit
  --output-path OUTPUT_PATH
  --method {pykan,efficientkan,fourierkan,fusedfourierkan,chebykan,cufkan,fast-kan,faster-kan,rbf-kan,mlp,all}
  --batch-size BATCH_SIZE
  --inp-size INP_SIZE   The dimension of the input variables.
  --hid-size HID_SIZE   The dimension of the hidden layer.
  --reps REPS           Number of times to repeat execution and average.
  --just-cuda           Whether to only execute the cuda version.
```

The benchmark is simply a network with 1 hidden layer where you can vary the number of input and hidden neurons and the output is 1D. This basically covers all the cases you want to profile since any other model is just made of several layers of this form. 

The MLP uses 10 times the hidden size in order for the number of parameters to be roughly the same and therefore comparable. And the original implementation is not included in the 'all' comparison because it takes orders of magnitude more time to execute. The fast-kan, faster-kan and rbf-kan algorithms have the default `num_grids` value changed to also have a comparable number of parameters.

An example of the output is in the times-*.txt files where BS means batch size, I for input size and H for hidden size. Also provided below (executed on a NVIDIA A5000 and an Intel i9-10900X):

```
                     |      forward  |     backward  |      forward  |     backward  |   num params  |  num trainable params
----------------------------------------------------------------------------------------------------------------------------------
effkan-cpu           |     31.98 ms  |     44.49 ms  |       nan GB  |       nan GB  |     10010000  |              10010000
effkan-gpu           |      4.76 ms  |      4.54 ms  |      0.13 GB  |      0.19 GB  |     10010000  |              10010000
fourierkan-cpu       |    727.35 ms  |    936.78 ms  |       nan GB  |       nan GB  |     10011001  |              10011001
fourierkan-gpu       |     17.93 ms  |     14.40 ms  |      1.96 GB  |      2.01 GB  |     10011001  |              10011001
fusedfourierkan-cpu  |    908.43 ms  |   1637.14 ms  |       nan GB  |       nan GB  |     10011001  |              10011001
fusedfourierkan-gpu  |     30.30 ms  |     84.61 ms  |      0.09 GB  |      0.13 GB  |     10011001  |              10011001
cufkan-cpu           |   1467.37 ms  |   3767.40 ms  |       nan GB  |       nan GB  |     10011001  |              10011001
cufkan-gpu           |      5.95 ms  |     49.74 ms  |      0.09 GB  |      0.13 GB  |     10011001  |              10011001
chebykan-cpu         |     20.29 ms  |     12.38 ms  |       nan GB  |       nan GB  |     10010000  |              10010000
chebykan-gpu         |      1.03 ms  |      1.21 ms  |      0.14 GB  |      0.13 GB  |     10010000  |              10010000
fast-kan-cpu         |      9.96 ms  |     17.06 ms  |       nan GB  |       nan GB  |     10015019  |              10015001
fast-kan-gpu         |      1.44 ms  |      2.13 ms  |      0.11 GB  |      0.14 GB  |     10015019  |              10015001
faster-kan-cpu       |     10.58 ms  |     15.42 ms  |       nan GB  |       nan GB  |     10014022  |              10014000
faster-kan-gpu       |      1.20 ms  |      2.01 ms  |      0.12 GB  |      0.14 GB  |     10014022  |              10014000
rbf-kan-cpu          |     12.59 ms  |     12.07 ms  |       nan GB  |       nan GB  |     10011019  |              10011001
rbf-kan-gpu          |      1.12 ms  |      2.08 ms  |      0.11 GB  |      0.13 GB  |     10011019  |              10011001
wav-kan-cpu          |   3063.83 ms  |   4453.43 ms  |       nan GB  |       nan GB  |      8012002  |               8012002
wav-kan-gpu          |     28.55 ms  |     65.31 ms  |      5.28 GB  |      5.28 GB  |      8012002  |               8012002
----------------------------------------------------------------------------------------------------------------------------------
mlp-cpu              |      9.77 ms  |      7.27 ms  |       nan GB  |       nan GB  |     10020001  |              10020001
mlp-gpu              |      0.49 ms  |      1.07 ms  |      0.10 GB  |      0.13 GB  |     10020001  |              10020001
----------------------------------------------------------------------------------------------------------------------------------
pykan-cpu            |     15.59 ms  |     17.53 ms  |       nan GB  |       nan GB  |         2431  |                  1551
pykan-gpu            |     50.56 ms  |     93.93 ms  |      0.02 GB  |      0.02 GB  |         2431  |                  1551
```

We are getting really close to MLP performance.

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
cd ../fast-kan
pip install -e .
```

To check everything is installed correctly run `python benchmark.py --help` and see there are no exception thrown.

### CUDA kernels

The code has the option to benchmark the Fused Fourier KAN, but you need to compile and install it first. This section provides instructions for Windows and Linux. 

First of all, you need to have installed cmake and the CUDA toolkit. Try to be consistent. If you use pytorch with CUDA 12.x, then the CUDA toolkit you install should also be 12.x. Check your installed version with `nvcc --version`. 

For Linux users, you can link against the installed torch package in your virtual environment:

```
cd FusedFourierKAN/build
cmake -DCMAKE_PREFIX_PATH=".venv/lib/python3.10/site-packages/torch" ..
cmake --build .
```

Later on, if you encounter this error

```
undefined symbol: _ZN3c108ListType3getERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEENS_4Type24SingletonOrSharedTypePtrIS9_EE
```

That is because the torch package distributed through pypi was compiled with pre-CXX11 ABI. To solve it you need to compile your kernels for that ABI too. For that you will have to change the configuration:

```
cmake -DCMAKE_PREFIX_PATH=".venv/lib/python3.10/site-packages/torch" -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" ..
```

If when compiling you still see that it is not using that flag, look into `CMakeFiles/fusedFourierKAN.dir/flags.make` and make sure that the `-D_GLIBCXX_USE_CXX11_ABI` has value 0 and not 1.

For Windows, download the C++ version of pytorch. Choose the Release version and extract the zip wherever you want and get the path of the folder `libtorch`. With that you can configure the project:

```
cd FusedFourierKAN/build
cmake -DCMAKE_PREFIX_PATH="/absolute/path/to/libtorch" ..
```

Once you have configured the project, you can try with `cmake --build . --config Release`. If that does not work, the configuration should have created some Visual Studio solutions. Open those and compile the project there in Release mode. If you get some error about  '/RTC1' and '/O2' not being compatible that is because you are not compiling in Release mode. Once finished, look for the location of the DLL. It should be on `FusedFourierKAN/build/Release/fusedFourierKAN.dll`. Go to `FusedFourierKAN/FusedFourierKAN/ffKANFunction.py` and modify line 10 to be `pluginpath = os.path.join(parent, "../build/Release/fusedFourierKAN.dll")`. Same with `FusedFourierKAN/FusedFourierKAN/ffKANGPUFunction.py`.

Once you have the dynamic library compiled, either in Linux or Windows, the last step is to install the python library:

```
cd FusedFourierKAN
pip install -e .
```

The DLL for the Visual Studio and cmake versions of Windows are in the releases section of the repo. The .so file compiled in WSL 2 Ubuntu 22.04 is also there. In case you are having trouble compiling, you could just download those and put them in the corresponding folders mentioned below.

#### Extra

The Fourier KAN kernel is reimplemented in the `extra` folder. The installation is included in the setup files. If you want to do it manually you have to first install the kernel:

```
cd extra/kernel
pip install .
```

And then install the python layer:

```
cd extra/
pip install .
```

You should have a working nvcc command and torch already installed in the environment.
