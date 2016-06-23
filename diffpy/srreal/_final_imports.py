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
Finalize tweak of classes from the extension module srreal_ext.

This private module handles loading of Python-level tweaks of the
extension-defined classes.  Any client that imports this module
must call the `import_now` function.  If this module is not loaded
by the time of srreal_ext initialization, `import_now` is executed
from srreal_ext.

This avoids unresolvable import dependencies for any order of imports.
"""


def import_now():
    '''
    Import all Python modules that tweak extension-defined classes.
    '''
    import diffpy.srreal.attributes
    import diffpy.srreal.atomradiitable
    import diffpy.srreal.bondcalculator
    import diffpy.srreal.bvscalculator
    import diffpy.srreal.overlapcalculator
    import diffpy.srreal.pdfbaseline
    import diffpy.srreal.pdfenvelope
    import diffpy.srreal.peakprofile
    import diffpy.srreal.peakwidthmodel
    import diffpy.srreal.scatteringfactortable
    import diffpy.srreal.pdfcalculator
    import diffpy.srreal.structureconverters
    return
