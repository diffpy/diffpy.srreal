#!/usr/bin/env python

# Installation script for diffpy.srreal

"""diffpy.srreal - calculators for PDF, bond valence sum, and other
quantities based on atom pair interaction.

Packages:   diffpy.srreal
"""

import os
import re
import sys
import glob
from setuptools import setup
from setuptools import Extension
from numpy.distutils.misc_util import get_numpy_include_dirs


# Use this version when git data are not available, like in git zip archive.
# Update when tagging a new release.
FALLBACK_VERSION = '1.3.0.post0'


# define extension arguments here
ext_kws = {
        'libraries' : ['diffpy'],
        'extra_compile_args' : ['-std=c++11'],
        'extra_link_args' : [],
        'include_dirs' : get_numpy_include_dirs(),
}

# determine if we run with Python 3.
PY3 = (sys.version_info[0] == 3)

# Figure out the tagged name of boost_python library.
def get_boost_libraries():
    """Check for installed boost_python shared library.

    Returns list of required boost_python shared libraries that are installed
    on the system. If required libraries are not found, an Exception will be
    thrown.
    """
    baselib = "boost_python"
    major, minor = (str(x) for x in sys.version_info[:2])
    pytags = [major + minor, major, '']
    mttags = ['', '-mt']
    boostlibtags = [(pt + mt) for mt in mttags for pt in pytags] + ['']
    from ctypes.util import find_library
    for tag in boostlibtags:
        lib = baselib + tag
        found = find_library(lib)
        if found: break

    # Show warning when library was not detected.
    if not found:
        import platform
        import warnings
        ldevname = 'LIBRARY_PATH'
        if platform.system() == 'Darwin':
            ldevname = 'DYLD_FALLBACK_LIBRARY_PATH'
        wmsg = ("Cannot detect name suffix for the %r library.  "
                "Consider setting %s.") % (baselib, ldevname)
        warnings.warn(wmsg)

    libs = [lib]
    return libs


def create_extensions():
    "Initialize Extension objects for the setup function."
    blibs = [n for n in get_boost_libraries()
            if not n in ext_kws['libraries']]
    ext_kws['libraries'] += blibs
    ext = Extension('diffpy.srreal.srreal_ext',
                    glob.glob('src/extensions/*.cpp'),
                    **ext_kws)
    return [ext]


# Extensions not included in pyproject.toml
setup_args = dict(
    ext_modules = [],
)


if __name__ == '__main__':
    setup_args['ext_modules'] = create_extensions()
    setup(**setup_args)

# End of file
