#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################


"""\
class ScatteringFactorTable -- scattering factors for atoms, ions and isotopes.
"""


# exported items, these also makes them show in pydoc.
__all__ = ['ScatteringFactorTable',
        'SFTXray', 'SFTElectron', 'SFTNeutron', 'SFTElectronNumber',
        'SFAverage']

from diffpy.srreal.srreal_ext import ScatteringFactorTable
from diffpy.srreal.srreal_ext import SFTXray
from diffpy.srreal.srreal_ext import SFTElectron
from diffpy.srreal.srreal_ext import SFTNeutron
from diffpy.srreal.srreal_ext import SFTElectronNumber
from diffpy.srreal.wraputils import _pickle_getstate, _pickle_setstate
from diffpy.srreal.sfaverage import SFAverage

# Pickling Support -----------------------------------------------------------

def _sft_create(owner):
    return owner.scatteringfactortable

def _sft_reduce(self):
    from diffpy.srreal.srreal_ext import ScatteringFactorTableOwner
    owner = ScatteringFactorTableOwner()
    owner.scatteringfactortable = self
    args = (owner,)
    rv = (_sft_create, args)
    return rv

def _sft_reduce_with_state(self):
    rv = _sft_reduce(self) + (self.__getstate__(),)
    return rv

# inject pickle methods to ScatteringFactorTable

ScatteringFactorTable.__reduce__ = _sft_reduce_with_state
ScatteringFactorTable.__getstate__ = _pickle_getstate
ScatteringFactorTable.__setstate__ = _pickle_setstate

SFTXray.__reduce__ = _sft_reduce
SFTElectron.__reduce__ = _sft_reduce
SFTNeutron.__reduce__ = _sft_reduce
SFTElectronNumber.__reduce__ = _sft_reduce

# End of file
