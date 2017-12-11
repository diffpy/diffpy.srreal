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

"""
Definitions of version-related constants and of libdiffpy_version_info.

Notes
-----
Variable `__gitsha__` is deprecated as of version 1.3.
Use `__git_commit__` instead.
"""

__all__ = ['__date__', '__git_commit__', '__timestamp__', '__version__',
           'libdiffpy_version_info']


from diffpy.srreal._version_data import __version__
from diffpy.srreal._version_data import __date__
from diffpy.srreal._version_data import __git_commit__
from diffpy.srreal._version_data import __timestamp__

# TODO remove deprecated __gitsha__ in version 1.4.
__gitsha__ = __git_commit__

# version information on the active libdiffpy shared library -----------------

from collections import namedtuple
from diffpy.srreal.srreal_ext import _get_libdiffpy_version_info_dict

libdiffpy_version_info = namedtuple('libdiffpy_version_info',
        "major minor micro patch version_number version date git_sha")
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
          `libdiffpy_version_info` will be removed in diffpy.srreal 1.3.
    """
    from warnings import warn
    warn("get_libdiffpy_version_info is deprecated, "
         "use the libdiffpy_version_info object.", DeprecationWarning)
    return libdiffpy_version_info

# End of file
