from setuptools import setup
from torch.utils import cpp_extension

setup(name='cutil',
      ext_modules=[cpp_extension.CppExtension('segtester.cutil.cutil', ['segtester/cutil/cutil.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})