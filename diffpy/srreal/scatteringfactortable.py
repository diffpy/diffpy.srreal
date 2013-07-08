#!/usr/bin/env python
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


# exported items, these also makes them show in pydoc.
__all__ = ['ScatteringFactorTable']

from diffpy.srreal.srreal_ext import ScatteringFactorTable
from diffpy.srreal.srreal_ext import SFTXray
from diffpy.srreal.srreal_ext import SFTElectron
from diffpy.srreal.srreal_ext import SFTNeutron
from diffpy.srreal.srreal_ext import SFTElectronNumber

# Pickling Support -----------------------------------------------------------

def _sft_getstate(self):
    state = (self.__dict__, )
    return state

def _sft_setstate(self, state):
    if len(state) != 1:
        emsg = ("expected 1-item tuple in call to __setstate__, got " +
                repr(state))
        raise ValueError(emsg)
    self.__dict__.update(state[0])
    return

def _sft_reduce(self):
    from diffpy.srreal.srreal_ext import ScatteringFactorTableOwner
    owner = ScatteringFactorTableOwner()
    owner.scatteringfactortable = self
    args = (owner,)
    rv = (_sft_create, args, self.__getstate__())
    return rv

def _sft_create(owner):
    return owner.scatteringfactortable

# inject pickle methods to ScatteringFactorTable

ScatteringFactorTable.__getstate__ = _sft_getstate
ScatteringFactorTable.__setstate__ = _sft_setstate
ScatteringFactorTable.__reduce__ = _sft_reduce

SFTXray.__getstate__ = _sft_getstate
SFTXray.__setstate__ = _sft_setstate
SFTXray.__reduce__ = _sft_reduce

SFTElectron.__getstate__ = _sft_getstate
SFTElectron.__setstate__ = _sft_setstate
SFTElectron.__reduce__ = _sft_reduce

SFTNeutron.__getstate__ = _sft_getstate
SFTNeutron.__setstate__ = _sft_setstate
SFTNeutron.__reduce__ = _sft_reduce

SFTElectronNumber.__getstate__ = _sft_getstate
SFTElectronNumber.__setstate__ = _sft_setstate
SFTElectronNumber.__reduce__ = _sft_reduce

# End of file
