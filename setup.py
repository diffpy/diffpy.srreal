#!/usr/bin/env python

# Installation script for diffpy.Structure

"""diffpy.srreal - prototype for new PDF calculator and assortment
of real space utilities.

Packages:   diffpy.srreal
Scripts:    (none yet)
"""

import os
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


def gitversion():
    from subprocess import Popen, PIPE
    proc = Popen(['git', 'describe'], stdout=PIPE)
    desc = proc.stdout.read().strip()
    proc = Popen(['git', 'log', '-1', '--format=%ai'], stdout=PIPE)
    isodate = proc.stdout.read()
    date = isodate.split()[0].replace('-', '')
    rv = desc + '-' + date
    return rv


def getsetupcfg():
    cfgfile = 'setup.cfg'
    from ConfigParser import SafeConfigParser
    cp = SafeConfigParser()
    cp.read(cfgfile)
    if not os.path.isdir('.git'):  return cp
    d = cp.defaults()
    vcfg = d.get('version', '')
    vgit = gitversion()
    if vgit != vcfg:
        cp.set('DEFAULT', 'version', vgit)
        cp.write(open(cfgfile, 'w'))
    return cp

cp = getsetupcfg()

# define distribution
dist = setup(
        name = "diffpy.srreal",
        version = cp.get('DEFAULT', 'version'),
        namespace_packages = ['diffpy'],
        packages = find_packages(),
        test_suite = 'diffpy.srreal.tests',
        include_package_data = True,
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
