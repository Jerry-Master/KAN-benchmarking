from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='FKAN_cpp',
    ext_modules=[cpp_extension.CppExtension('FKAN_cpp', ['FKAN.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)