#from distutils.core import setup
#from Cython.Build import cythonize
#import numpy
 
#setup(
#    ext_modules = cythonize("_coxph_loss.pyx"),include_dirs=[numpy.get_include()]
#)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext  =  [Extension( "_aft_loss", sources=["_aft_loss.pyx"] )]

setup(
   name = "testing", 
   cmdclass={'build_ext' : build_ext}, 
   include_dirs = [np.get_include()],   
   ext_modules=ext
   )