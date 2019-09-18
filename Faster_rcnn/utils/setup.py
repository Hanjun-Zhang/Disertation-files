#!/usr/bin/python  
#python version: 2.7.3  
#Filename: SetupTestOMP.py  
   
# Run as:    
#    python setup.py build_ext --inplace    
     
import sys    
sys.path.insert(0, "..")    
     
from distutils.core import setup    
from distutils.extension import Extension    
from Cython.Build import cythonize    
from Cython.Distutils import build_ext  
     
# ext_module = cythonize("TestOMP.pyx")    
ext_modules = Extension(  
                        "cython_bbox",  
            ["bbox.pyx"],  
            extra_compile_args=["/openmp"],  
            extra_link_args=["/openmp"],  
            )  
    
setup(  
    name='tf_faster_rcnn',
    ext_modules=[ext_modules],
    # inject our custom trigger
    cmdclass={'build_ext': build_ext}, 
)
# python setup.py build_ext --include-dirs=C:\Users\Lenovo\Anaconda3\Lib\site-packages\numpy\core\include