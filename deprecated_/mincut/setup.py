
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# run: python setup.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules=cythonize("mincut.pyx"),
    include_dirs=[numpy.get_include()]
)