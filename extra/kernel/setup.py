from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(
    name='cuFKAN-kernel',
    ext_modules=[
        CUDAExtension('cuFKAN_kernel', [
            'cuFKAN.cpp',
            'cuFKAN-kernel.cu',
        ]),
        CppExtension('cuFKAN_kernel_cpp', ['FKAN.cpp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=find_packages(),
    author="Jerry-Master",
    author_email="joseperez2000@hotmail.es",
    description="Fourier KAN operations in CPU and GPU.",
    license="Apache-2.0",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.10",
        "Typing :: Stubs Only",
    ],
    python_requires=">=3.10",
)
