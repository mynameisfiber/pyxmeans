from distutils.core import setup, Extension
import numpy.distutils.misc_util

__version__ = "0.6.2"

_minibatch = Extension(
    'pyxmeans._minibatch',
    sources = ['pyxmeans/_minibatch.c', ] + ['pyxmeans/c_minibatch/' + filename for filename in ("distance.c", "generate_data.c", "minibatch.c",)],
    extra_compile_args = ["-O3", "-std=c99", "-fopenmp", "-Wall", "-p", "-pg", ],
    extra_link_args = ["-lgomp", "-lc", "-lm"],
)

setup (
    name = 'pyxmeans',
    version = __version__,
    description = 'Fast and dirty xmeans',
    ext_modules = [_minibatch,],
    packages = ["pyxmeans", ],
    include_dirs = numpy.distutils.misc_util.get_numpy_include_dirs(),
)
