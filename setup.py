#!/usr/bin/env python

"""diffpy.srreal - calculators for PDF, bond valence sum, and other
quantities based on atom pair interaction.

Packages:   diffpy.srreal
"""

import numpy
import sys
import glob
from setuptools import setup, Extension
from ctypes.util import find_library


def get_boost_libraries():
    base_lib = "boost_python"
    major, minor = str(sys.version_info[0]), str(sys.version_info[1])
    tags = [f"{major}{minor}", major, ""]
    mttags = ["", "-mt"]
    candidates = [base_lib + tag for tag in tags for mt in mttags] + [base_lib]
    for lib in candidates:
        if find_library(lib):
            return [lib]
    raise RuntimeError("Cannot find a suitable Boost.Python library.")


ext_kws = {
    "libraries": ["diffpy"] + get_boost_libraries(),
    "extra_compile_args": ["-std=c++11"],
    "extra_link_args": [],
    "include_dirs": [numpy.get_include()],
}


def create_extensions():
    "Initialize Extension objects for the setup function."
    ext = Extension("diffpy.srreal.srreal_ext", glob.glob("src/extensions/*.cpp"), **ext_kws)
    return [ext]


# Extensions not included in pyproject.toml
setup_args = dict(
    ext_modules=[],
)


if __name__ == "__main__":
    setup_args["ext_modules"] = create_extensions()
    setup(**setup_args)
