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


class BuildExtNoFPContract(build_ext):
    """``build_ext`` that forbids the compiler from fusing multiply-add.

    These extensions exist to reproduce R bit for bit, and a fused
    multiply-add rounds once where the source asks for twice. ``lowess.cpp``
    accumulates ``c += w[k] * (diff * diff)``, which a compiler with an FMA
    instruction available may contract into a single ``fma(w, d*d, c)``; that
    is one ULP on the first pass, and because each robustness iteration
    recomputes the weights from the previous pass's residuals it compounds --
    measured here as 2 of 16 points differing at ``iter=0`` and 12 of 16 at
    the default ``iter=3``.

    Clang defaults to ``-ffp-contract=on``, which permits contraction inside a
    single statement, and every arm64 chip has the instruction, so macOS wheels
    contract while x86-64 Linux and Windows ones do not: the baseline x86-64
    ISA has no FMA, so those compilers cannot contract even when allowed to.
    That is why this only ever showed up on macOS. MSVC's ``/fp:precise``
    default already forbids it; it is passed explicitly so the intent survives
    a future default change.
    """

    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            flags = ["/fp:precise"]
        else:
            flags = ["-ffp-contract=off"]
        for ext in self.extensions:
            ext.extra_compile_args = list(ext.extra_compile_args or []) + flags
        super().build_extensions()


setup(ext_modules=ext_modules, cmdclass={"build_ext": BuildExtNoFPContract})
