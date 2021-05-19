
import torch
from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

compile_args = {"cxx": [], "nvcc": [] }

if torch.__version__ < "1.8":
    version = torch.__version__.split('.')
    compile_args['cxx'] += ["-DTORCH_VERSION_MAJOR={}".format(version[0])]
    compile_args['cxx'] += ["-DTORCH_VERSION_MINOR={}".format(version[1])]

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

