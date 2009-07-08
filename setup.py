#!/usr/bin/env python

# Installation script for diffpy.Structure

"""diffpy.srreal - prototype for new PDF calculator and assortment
of real space utilities.

Packages:   diffpy.srreal
Scripts:    (none yet)
"""

from setuptools import setup, find_packages
from setuptools import Extension
import fix_setuptools_chmod

# define extension here
srrealmodule = Extension('diffpy.srreal.pdf_ext', [
            'srrealmodule/pdf_ext.cpp',
            ],
        include_dirs = ['libsrreal'],
        extra_compile_args = [],
        extra_link_args = [],
        libraries = ['srreal', 'boost_python'],
)

# define distribution
dist = setup(
        name = "diffpy.srreal",
        version = "0.1a",
        namespace_packages = ['diffpy'],
        packages = find_packages(exclude=['PDFAPI']),
        ext_modules = [srrealmodule],
        entry_points = {
            'console_scripts' : [
                'downhill1=diffpy.srreal.applications.downhill1:main',
                'colorFromOverlap=' + \
                    'diffpy.srreal.applications.colorFromOverlap:main',
                'colorFromOverlapCmpPDF=' + \
                    'diffpy.srreal.applications.colorFromOverlapCmpPDF:main',
                'crystalCoordination=' + \
                    'diffpy.srreal.applications.crystalCoordination:main',
            ],
        },
        install_requires = [
            'diffpy.Structure',
            'diffpy.pdffit2',
            'elements',
        ],
        dependency_links = [
            # REMOVE dev.danse.us for a public release.
            'http://dev.danse.us/packages/',
            "http://www.diffpy.org/packages/",
        ],

        author = "Simon J.L. Billinge",
        author_email = "sb2896@columbia.edu",
        description = "Prototype for new PDF calculator and other real " + \
                      "space utilities.",
        license = "BSD",
        url = "http://www.diffpy.org/",
        keywords = "PDF calculator real-space utilities",
)

# End of file
