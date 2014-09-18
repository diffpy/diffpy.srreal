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


"""Definition of __version__, __date__, __gitsha__.
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

# Access to version data of the libdiffpy shared library ---------------------

def get_libdiffpy_version_info():
    """Get version data for the linked libdiffpy shared library.

    Return a singleton instance of libdiffpy_version_info class.
    """
    global _lvi
    if _lvi is not None:  return _lvi
    from diffpy.srreal.srreal_ext import _get_libdiffpy_version_info_dict
    vd = _get_libdiffpy_version_info_dict()

    class libdiffpy_version_info(object):

        """Version information for the loaded libdiffpy shared library.

        version  -- version string for the loaded libdiffpy library.
        version_number -- Integer encoding of the library version.
        major    -- Major version number of the library.
        minor    -- Minor version number of the library.
        date     -- Git commit date of the libdiffpy sources.
        git_sha  -- Git commit hash of this libdiffpy version.
        """

        version = vd['version_str']
        version_number = vd['version']
        major = vd['major']
        minor = vd['minor']
        date = vd['date']
        git_sha = vd['git_sha']

    # class libdiffpy_version_info

    _lvi = libdiffpy_version_info()
    return get_libdiffpy_version_info()

_lvi = None

# End of file
