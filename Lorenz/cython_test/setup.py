# run from terminal
# python3 setup.py build_ext --inplace
#
from distutils.core import setup
from Cython.Build import cythonize

setup(name='lorenz app',
      ext_modules=cythonize("lorenz_run.py"))