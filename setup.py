#!/usr/bin/env python

# Installation script for diffpy.Structure

"""diffpy.srreal - prototype for new PDF calculator and assortment
of real space utilities.

Packages:   diffpy.srreal
Scripts:    (none yet)
"""

from setuptools import setup, find_packages
import fix_setuptools_chmod

# define distribution
dist = setup(
        name = "diffpy.srreal",
        version = "0.1a",
        namespace_packages = ['diffpy'],
        packages = ['diffpy.srreal'],
        install_requires = [
            'diffpy.Structure',
            'diffpy.pdffit2',
        ],
        dependency_links = [
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
