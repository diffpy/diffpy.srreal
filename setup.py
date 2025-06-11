#!/usr/bin/env python

# Installation script for diffpy.srreal
"""diffpy.srreal - calculators for PDF, bond valence sum, and other
quantities based on atom pair interaction.

Packages:   diffpy.srreal
"""

import glob
import os
import re
import sys

from numpy.distutils.misc_util import get_numpy_include_dirs
from setuptools import Extension, find_packages, setup

# Use this version when git data are not available, like in git zip archive.
# Update when tagging a new release.
FALLBACK_VERSION = "1.3.0.post0"


# define extension arguments here
ext_kws = {
    "libraries": ["diffpy"],
    "extra_compile_args": ["-std=c++11"],
    "extra_link_args": [],
    "include_dirs": get_numpy_include_dirs(),
}

# determine if we run with Python 3.
PY3 = sys.version_info[0] == 3


# Figure out the tagged name of boost_python library.
def get_boost_libraries():
    """Check for installed boost_python shared library.

    Returns list of required boost_python shared libraries that are
    installed on the system. If required libraries are not found, an
    Exception will be thrown.
    """
    baselib = "boost_python"
    major, minor = (str(x) for x in sys.version_info[:2])
    pytags = [major + minor, major, ""]
    mttags = ["", "-mt"]
    boostlibtags = [(pt + mt) for mt in mttags for pt in pytags] + [""]
    from ctypes.util import find_library

    for tag in boostlibtags:
        lib = baselib + tag
        found = find_library(lib)
        if found:
            break

    # Show warning when library was not detected.
    if not found:
        import platform
        import warnings

        ldevname = "LIBRARY_PATH"
        if platform.system() == "Darwin":
            ldevname = "DYLD_FALLBACK_LIBRARY_PATH"
        wmsg = (
            "Cannot detect name suffix for the %r library.  " "Consider setting %s."
        ) % (baselib, ldevname)
        warnings.warn(wmsg)

    libs = [lib]
    return libs


def create_extensions():
    "Initialize Extension objects for the setup function."
    blibs = [n for n in get_boost_libraries() if not n in ext_kws["libraries"]]
    ext_kws["libraries"] += blibs
    ext = Extension(
        "diffpy.srreal.srreal_ext", glob.glob("src/extensions/*.cpp"), **ext_kws
    )
    return [ext]


# versioncfgfile holds version data for git commit hash and date.
# It must reside in the same directory as version.py.
MYDIR = os.path.dirname(os.path.abspath(__file__))
versioncfgfile = os.path.join(MYDIR, "src/diffpy/srreal/version.cfg")
gitarchivecfgfile = os.path.join(MYDIR, ".gitarchive.cfg")


def gitinfo():
    from subprocess import PIPE, Popen

    kw = dict(stdout=PIPE, cwd=MYDIR, universal_newlines=True)
    proc = Popen(["git", "describe", "--match=v[[:digit:]]*"], **kw)
    desc = proc.stdout.read()
    proc = Popen(["git", "log", "-1", "--format=%H %ct %ci"], **kw)
    glog = proc.stdout.read()
    rv = {}
    rv["version"] = ".post".join(desc.strip().split("-")[:2]).lstrip("v")
    rv["commit"], rv["timestamp"], rv["date"] = glog.strip().split(None, 2)
    return rv


def getversioncfg():
    if PY3:
        from configparser import RawConfigParser
    else:
        from ConfigParser import RawConfigParser
    vd0 = dict(version=FALLBACK_VERSION, commit="", date="", timestamp=0)
    # first fetch data from gitarchivecfgfile, ignore if it is unexpanded
    g = vd0.copy()
    cp0 = RawConfigParser(vd0)
    cp0.read(gitarchivecfgfile)
    if len(cp0.get("DEFAULT", "commit")) > 20:
        g = cp0.defaults()
        mx = re.search(r"\btag: v(\d[^,]*)", g.pop("refnames"))
        if mx:
            g["version"] = mx.group(1)
    # then try to obtain version data from git.
    gitdir = os.path.join(MYDIR, ".git")
    if os.path.exists(gitdir) or "GIT_DIR" in os.environ:
        try:
            g = gitinfo()
        except OSError:
            pass
    # finally, check and update the active version file
    cp = RawConfigParser()
    cp.read(versioncfgfile)
    d = cp.defaults()
    rewrite = not d or (
        g["commit"]
        and (g["version"] != d.get("version") or g["commit"] != d.get("commit"))
    )
    if rewrite:
        cp.set("DEFAULT", "version", g["version"])
        cp.set("DEFAULT", "commit", g["commit"])
        cp.set("DEFAULT", "date", g["date"])
        cp.set("DEFAULT", "timestamp", g["timestamp"])
        with open(versioncfgfile, "w") as fp:
            cp.write(fp)
    return cp


versiondata = getversioncfg()

with open(os.path.join(MYDIR, "README.rst")) as fp:
    long_description = fp.read()

# define distribution
setup_args = dict(
    name="diffpy.srreal",
    version=versiondata.get("DEFAULT", "version"),
    packages=find_packages(os.path.join(MYDIR, "src")),
    package_dir={"": "src"},
    test_suite="diffpy.srreal.tests",
    include_package_data=True,
    ext_modules=[],
    install_requires=[
        "diffpy.structure",
    ],
    zip_safe=False,
    author="Simon J.L. Billinge group",
    author_email="sb2896@columbia.edu",
    maintainer="Pavol Juhas",
    maintainer_email="pavol.juhas@gmail.com",
    description=(
        "calculators for PDF, bond valence sum, and other "
        "quantities based on atom pair interaction."
    ),
    long_description=long_description,
    long_description_content_type="text/x-rst",
    license="BSD-style license",
    url="https://github.com/diffpy/diffpy.srreal/",
    keywords="PDF BVS atom overlap calculator real-space",
    classifiers=[
        # List of possible values at
        # http://pypi.python.org/pypi?:action=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: C++",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
    ],
)

if __name__ == "__main__":
    setup_args["ext_modules"] = create_extensions()
    setup(**setup_args)

# End of file
