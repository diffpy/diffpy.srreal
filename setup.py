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
    major, minor = sys.version_info[:2]
    candidates = [
        f"boost_python{major}{minor}",
        f"boost_python{major}",
        "boost_python",
    ]

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        libdir = os.path.join(conda_prefix, "lib")
        for name in candidates:
            so = f"lib{name}.so"
            if os.path.isfile(os.path.join(libdir, so)):
                return [name]

    # fallback to ldconfig
    for name in candidates:
        found = find_library(name)
        if found:
            return [name]
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
                "Neither BOOST_PATH nor CONDA_PREFIX are set. "
                "Please install Boost or set BOOST_PATH."
            )
        if os.name == "nt":
            inc = Path(conda_prefix) / "Library" / "include"
            lib = Path(conda_prefix) / "Library" / "lib"
        else:
            inc = Path(conda_prefix) / "include"
            lib = Path(conda_prefix) / "lib"
    return {"include_dirs": [str(inc)], "library_dirs": [str(lib)]}


def get_objcryst_libraries():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        raise EnvironmentError(
            "CONDA_PREFIX is not set. "
            "Please install ObjCryst using conda and activate the environment."
        )
    if os.name == "nt":
        libdir = Path(conda_prefix) / "Library" / "lib"
    else:
        libdir = Path(conda_prefix) / "lib"

    libs = []
    for fn in os.listdir(libdir):
        stem = Path(fn).stem
        if "objcryst" not in stem.lower():
            continue
        # strip a leading "lib"
        # so that setuptools does -lObjCryst, not -llibObjCryst
        if os.name != "nt" and stem.startswith("lib"):
            stem = stem[3:]
        libs.append(stem)

    if not libs:
        raise RuntimeError(f"No ObjCryst libraries found in {libdir}")
    return libs


if os.name == "nt":
    compile_args = ["/std:c++14"]
    macros = [("_USE_MATH_DEFINES", None)]
    extra_link_args = ["/FORCE:MULTIPLE"]
else:
    compile_args = ["-std=c++11"]
    macros = []
    extra_link_args = []

boost_cfg = get_boost_config()
objcryst_libs = get_objcryst_libraries()

ext_kws = {
    "libraries": ["diffpy"] + get_boost_libraries() + objcryst_libs,
    "extra_compile_args": compile_args,
    "extra_link_args": extra_link_args,
    "include_dirs": [numpy.get_include()] + boost_cfg["include_dirs"],
    "library_dirs": boost_cfg["library_dirs"],
    # "runtime_library_dirs": boost_cfg["library_dirs"],
    "define_macros": macros,
}


def create_extensions():
    "Initialize Extension objects for the setup function."
    ext = Extension(
        "diffpy.srreal.srreal_ext",
        glob.glob("src/extensions/*.cpp"),
        **ext_kws,
    )
    return [ext]


# Extensions not included in pyproject.toml
def ext_modules():
    if set(sys.argv) & {"build_ext", "bdist_wheel", "install"}:
        return create_extensions()
    return []


if __name__ == "__main__":
    setup(ext_modules=ext_modules())
