#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     Complex Modeling Initiative
#                   (c) 2016 Brookhaven Science Associates,
#                   Brookhaven National Laboratory.
#                   All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################


"""\
Cancel registration of Python-extended C++ classes when Python exits.

Note
----
Python finishes before the shared library libdiffpy, but the registry of
class prototypes is implemented in libdiffpy.  Any Python-extended classes
thus need to be removed from the registry prior to Python exit.

This module is not intended for direct use.  It is used implicitly within
a call of _registerThisType.
"""


import weakref


# Routine to be used from srreal_ext module ----------------------------------

def registerForCleanUp(obj):
    """Remember to clean up the specified prototype at Python exit.

    Parameters
    ----------
    obj : wrapped class that has class registry
        This is an object being added to the C++ registry of prototypes.
        If active at Python exit, the associated string type will be
        removed from the class registry.

    No return value.
    """
    _cleanup_handler.add(obj)
    return

# ----------------------------------------------------------------------------

class _DerivedClassesCleanUpHandler(object):

    def __init__(self):
        self._references = set()
        return


    def add(self, obj):
        wr = weakref.ref(obj)
        self._references.add(wr)
        return


    def __del__(self):
        while self._references:
            wr = self._references.pop()
            obj = wr()
            if obj is not None:
                obj._deregisterType(obj.type())
        return

# end of class _DerivedClassesCleanUpHandler


# create singleton instance of the cleanup handler
_cleanup_handler = _DerivedClassesCleanUpHandler()
del _DerivedClassesCleanUpHandler


# End of file.
