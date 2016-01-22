#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2008 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################


"""Definition of __version__, __date__, __gitsha__, libdiffpy_version_info.
"""

from pkg_resources import resource_filename
from ConfigParser import RawConfigParser

# obtain version information from the version.cfg file
cp = RawConfigParser(dict(version='', date='', commit='', timestamp=0))
if not cp.read(resource_filename(__name__, 'version.cfg')):
    from warnings import warn
    warn('Package metadata not found, execute "./setup.py egg_info".')

__version__ = cp.get('DEFAULT', 'version')
__date__ = cp.get('DEFAULT', 'date')
__gitsha__ = cp.get('DEFAULT', 'commit')
__timestamp__ = cp.getint('DEFAULT', 'timestamp')

del cp

# version information on the active libdiffpy shared library -----------------

from collections import namedtuple
from diffpy.srreal.srreal_ext import _get_libdiffpy_version_info_dict

libdiffpy_version_info = namedtuple('libdiffpy_version_info',
        "version version_number major minor micro patch date git_sha")
vd = _get_libdiffpy_version_info_dict()
libdiffpy_version_info = libdiffpy_version_info(
        version = vd['version_str'],
        version_number = vd['version'],
        major = vd['major'],
        minor = vd['minor'],
        micro = vd['micro'],
        patch = vd['patch'],
        date = vd['date'],
        git_sha = vd['git_sha'])
del vd


def get_libdiffpy_version_info():
    """
    Get version data for the active libdiffpy shared library.

    Returns
    -------
    libdiffpy_version_info
        a namedtuple which contains libdiffpy version data.


    .. note:: Deprecated in diffpy.srreal 1.1
          `libdiffpy_version_info` will be removed in diffpy.srreal 1.2.
    """
    from warnings import warn
    warn("get_libdiffpy_version_info is deprecated, "
         "use the libdiffpy_version_info object.", DeprecationWarning)
    return libdiffpy_version_info

# End of file
