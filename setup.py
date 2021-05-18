
import torch
from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

compile_args = {"cxx": [], "nvcc": [] }

setup(
        name = 'layer_norm_test',
        version = '1.0',
        author = 'layer norm',
        packages=find_packages(),
        ext_modules=[cpp_extension.CppExtension(
            'native',
            ['native.cpp'],
            extra_compile_args=compile_args,
            )],
        cmdclass={'build_ext': cpp_extension.BuildExtension})

