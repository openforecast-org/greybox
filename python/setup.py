import os

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Vendor includes shipped via git submodules under python/extern/.
# carma is the numpy <-> Armadillo bridge by RUrlus; we ship the headers
# so native modules can `#include <carma>` without requiring users to
# install it separately. The C++ files currently in `_native/` do not yet
# depend on carma, but the include path is exposed for future use.
_EXTERN_INCLUDE_DIRS = [
    os.path.join(os.path.dirname(__file__), "extern", "carma", "include"),
]

ext_modules = [
    Pybind11Extension(
        "greybox._native_lowess",
        sources=["src/greybox/_native/lowess.cpp"],
        include_dirs=_EXTERN_INCLUDE_DIRS,
        cxx_std=17,
    ),
    Pybind11Extension(
        "greybox._native_supsmu",
        sources=["src/greybox/_native/supsmu.cpp"],
        include_dirs=_EXTERN_INCLUDE_DIRS,
        cxx_std=17,
    ),
    Pybind11Extension(
        "greybox._native_dgamma",
        sources=["src/greybox/_native/dgamma.cpp"],
        include_dirs=_EXTERN_INCLUDE_DIRS,
        cxx_std=17,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
