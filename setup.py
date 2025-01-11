#!/usr/bin/env python

"""diffpy.srreal - calculators for PDF, bond valence sum, and other
quantities based on atom pair interaction.

Packages:   diffpy.srreal
"""

import glob
import os
import sys
from ctypes.util import find_library
from pathlib import Path

import numpy
from setuptools import Extension, setup


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


def get_boost_config():
    boost_path = os.environ.get("BOOST_PATH", "")
    if boost_path:
        inc = Path(boost_path) / "include"
        lib = Path(boost_path) / "lib"
    else:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            raise EnvironmentError(
                "Neither BOOST_PATH nor CONDA_PREFIX are set. " "Please install Boost or set BOOST_PATH."
            )
        if os.name == "nt":
            inc = Path(conda_prefix) / "Library" / "include"
            lib = Path(conda_prefix) / "Library" / "lib"
        else:
            inc = Path(conda_prefix) / "include"
            lib = Path(conda_prefix) / "lib"
    return {"include_dirs": [str(inc)], "library_dirs": [str(lib)]}


boost_cfg = get_boost_config()
ext_kws = {
    "libraries": ["diffpy"] + get_boost_libraries(),
    "extra_compile_args": ["-std=c++11"],
    "extra_link_args": [],
    "include_dirs": [numpy.get_include()] + boost_cfg["include_dirs"],
    "library_dirs": boost_cfg["library_dirs"],
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
