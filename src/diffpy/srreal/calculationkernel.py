#!/usr/bin/env python
##############################################################################
#
# (c) 2026 The Trustees of Columbia University in the City of New York.
# All rights reserved.
#
# File coded by: Billinge Group members and community contributors.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/diffpy/diffpy.srreal/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Selection of the kernel that sums the contributions of atom pairs.

PDFCalculator and DebyePDFCalculator send full pair sums to the selected
kernel, by default the best one the CPU supports.  For refinements with
'cuda-mixed', use larger finite-difference steps and verify the fit in
double precision.
"""

# exported items
__all__ = [
    "availableCalculationKernels",
    "getCalculationKernel",
    "setCalculationKernel",
]

from diffpy.srreal.srreal_ext import (
    availableCalculationKernels,
    getCalculationKernel,
    setCalculationKernel,
)

# End of file
