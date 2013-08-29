from distutils.core import setup, Extension
import numpy.distutils.misc_util

__version__ = "0.1"

c_xmeans = Extension(
    '_xmeans',
    sources = ['pyxmeans/_xmeans.c'],
)

setup (
    name = 'pyxmeans',
    version = __version__,
    description = 'Fast and dirty xmeans',
    ext_modules = [c_xmeans,],
    include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
)
