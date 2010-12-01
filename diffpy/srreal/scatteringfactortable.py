##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 Trustees of the Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################


"""class ScatteringFactorTable -- scattering factors for atoms, ions and
isotopes.
"""

# module version
__id__ = "$Id$"

# exported items, these also makes them show in pydoc.
__all__ = ['ScatteringFactorTable']

from diffpy.srreal.srreal_ext import ScatteringFactorTable

# Pickling Support -----------------------------------------------------------

def _sft_getstate(self):
    state = (self.__dict__, self.getAllCustom())
    return state

def _sft_setstate(self, state):
    if len(state) != 2:
        emsg = ("expected 2-item tuple in call to __setstate__, got %r" +
                repr(state))
        raise ValueError(emsg)
    st = iter(state)
    self.__dict__.update(st.next())
    self.resetAll()
    for k, v in st.next().iteritems():
        self.setCustom(k, v)
    return

def _sft_reduce(self):
    if type(self) is ScatteringFactorTable:
        factory = _sft_create
        factory_args = (self.type(),)
    else:
        factory = type(self)
        factory_args = ()
    rv = (factory, factory_args, self.__getstate__())
    return rv

def _sft_create(tp):
    return ScatteringFactorTable.createByType(tp)

# inject pickle methods to ScatteringFactorTable

ScatteringFactorTable.__getstate__ = _sft_getstate
ScatteringFactorTable.__setstate__ = _sft_setstate
ScatteringFactorTable.__reduce__ = _sft_reduce

# End of file
