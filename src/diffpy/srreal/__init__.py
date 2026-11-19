#!/usr/bin/env python
##############################################################################
#
# (c) 2025 The Trustees of Columbia University in the City of New York.
# (c) 2026 diffpy.srreal contributors.
# All rights reserved.
#
# File coded by: Simon Billinge, Billinge Group members.
#
# See GitHub contributions for a more detailed list of contributors.
# https://github.com/diffpy/diffpy.srreal/graphs/contributors
#
# See LICENSE.rst for license information.
#
##############################################################################
"""Calculators for PDF, bond valence sum, and other quantities based on
atom pair interaction."""

# Windows DLL loading fix for Python 3.8+
# On Windows, add the conda library bin directory to DLL search path
# before importing extension modules that depend on DLLs
import os
import sys

if sys.platform == "win32" and sys.version_info >= (3, 8):
    # Try to add conda library bin directory to DLL search path
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        lib_bin_dir = os.path.join(conda_prefix, "Library", "bin")
        if os.path.isdir(lib_bin_dir):
            os.add_dll_directory(lib_bin_dir)

# package version
from diffpy.srreal.version import __version__  # noqa

# silence the pyflakes syntax checker
assert __version__ or True

# End of file
