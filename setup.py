from distutils.core import setup
from Cython.Build import cythonize
from Cython.Compiler import Options
from distutils.extension import Extension
from numpy import get_include

Options.annotate = True
compiler_directives = {"boundscheck": False, "cdivision": True,
                       "wraparound": False}
cflags = ["-Wno-unused-function", "-Wno-#warnings"]

extensions = [Extension(
    "*", ["vlasov/cython/*.pyx"],
    include_dirs=[get_include()], extra_compile_args=cflags)]

setup(
    name='vlasov',
    version='0.0.1',
    packages=['vlasov'],
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives))
