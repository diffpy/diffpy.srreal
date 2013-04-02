#!/usr/bin/env python

# Installation script for diffpy.Structure

"""diffpy.srreal - prototype for new PDF calculator and assortment
of real space utilities.

Packages:   diffpy.srreal
Scripts:    (none yet)
"""

import glob
from setuptools import setup, find_packages
from setuptools import Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

# define extensions here
ext_kws = {
        'libraries' : ['diffpy'],
        'extra_compile_args' : [],
        'extra_link_args' : [],
        'include_dirs' : get_numpy_include_dirs(),
}

srreal_ext = Extension('diffpy.srreal.srreal_ext',
    glob.glob('srrealmodule/*.cpp'),
    **ext_kws)

# define distribution
dist = setup(
        name = "diffpy.srreal",
        version = "0.2a1",
        namespace_packages = ['diffpy'],
        packages = find_packages(exclude=['tests']),
        test_suite = 'diffpy.srreal.tests',
        include_package_data = True,
        data_files = [
            ('diffpy/srreal/tests/testdata',
                glob.glob('diffpy/srreal/tests/testdata/*')),
        ],
        ext_modules = [srreal_ext],
        install_requires = [
            'diffpy.Structure',
        ],
        dependency_links = [
            # REMOVE dev.danse.us for a public release.
            'http://dev.danse.us/packages/',
            "http://www.diffpy.org/packages/",
        ],
        zip_safe = False,

        author = "Simon J.L. Billinge",
        author_email = "sb2896@columbia.edu",
        description = "Prototype for new PDF calculator and other real " + \
                      "space utilities.",
        license = "BSD",
        url = "http://www.diffpy.org/",
        keywords = "PDF calculator real-space utilities",
)

# End of file
