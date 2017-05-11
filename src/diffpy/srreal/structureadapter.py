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


"""class StructureAdapter -- adapter of any structure object to the interface
    expected by srreal PairQuantity calculators

Routines:

createStructureAdapter -- create StructureAdapter from a Python object
nometa       -- create StructureAdapter with disabled _customPQConfig method
                this prevents copying of diffpy.Structure pdffit metadata
                to PDFCalculator object
nosymmetry   -- create StructureAdapter with disabled symmetry expansion.

Constants:

EMPTY        -- singleton instance of an empty structure.
"""

def createStructureAdapter(stru):
    '''
    Create StructureAdapter from a Python object.

    stru -- an object that is convertible to StructureAdapter, i.e., it has
            a registered factory that converts Python structure object to
            StructureAdapter.  Return stru if already a StructureAdapter.

    Return a StructureAdapter instance.
    Raise TypeError if stru cannot be converted to StructureAdapter.
    '''
    if isinstance(stru, StructureAdapter):  return stru
    import inspect
    # build fully-qualified names of Python types in method resolution order
    cls = type(stru)
    fqnames = [str(tp).split("'")[1] for tp in inspect.getmro(cls)]
    for fqn in fqnames:
        if not fqn in _adapter_converters_registry:  continue
        factory = _adapter_converters_registry[fqn]
        return factory(stru)
    # none of the registered factories could convert the stru object
    emsg = "Cannot create structure adapter for %r." % (stru,)
    raise TypeError(emsg)


def RegisterStructureAdapter(fqname, fnc=None):
    '''Function decorator that marks it as a converter of specified
    object type to StructureAdapter class in diffpy.srreal.  The registered
    structure object types can be afterwards directly used with calculators
    in diffpy.srreal as they would be implicitly converted to the internal
    diffpy.srreal structure type.

    fqname   -- fully qualified class name for the convertible objects.
                This is the quoted string included in "str(type(obj))".
                The converter function would be called for object of the
                same or derived types.
    fnc      -- function that converts the fqname type to StructureAdapter.

    Note: When fnc is None RegisterStructureAdapter works as a decorator
    and the conversion function can be specified below, i.e.,

        @RegisterStructureAdapter('my.structure.Type')
        def convertMyStructure(stru):
            ...

    See diffpy.srreal.structureconverters module for usage example.
    '''
    def __wrapper(fnc):
        _adapter_converters_registry[fqname] = fnc
        return fnc
    if fnc is None:
        return __wrapper
    return __wrapper(fnc)

_adapter_converters_registry = {}

# import of srreal_ext calls RegisterStructureAdapter, therefore it has
# to be at the end of this module.

from diffpy.srreal.srreal_ext import StructureAdapter
from diffpy.srreal.srreal_ext import Atom, AtomicStructureAdapter
from diffpy.srreal.srreal_ext import PeriodicStructureAdapter
from diffpy.srreal.srreal_ext import CrystalStructureAdapter
from diffpy.srreal.srreal_ext import StructureDifference
from diffpy.srreal.srreal_ext import nometa, nosymmetry
from diffpy.srreal.srreal_ext import _emptyStructureAdapter
from diffpy.srreal.srreal_ext import BaseBondGenerator

EMPTY = _emptyStructureAdapter()
del _emptyStructureAdapter

# silence the pyflakes syntax checker
assert all((Atom, AtomicStructureAdapter, PeriodicStructureAdapter,
            CrystalStructureAdapter, StructureDifference,
            nometa, nosymmetry, BaseBondGenerator))

# End of file
