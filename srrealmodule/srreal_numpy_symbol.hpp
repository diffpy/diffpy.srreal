/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* Definition of the PY_ARRAY_UNIQUE_SYMBOL for the srreal extension module.
*
*****************************************************************************/

#ifndef SRREAL_NUMPY_SYMBOL_HPP_INCLUDED
#define SRREAL_NUMPY_SYMBOL_HPP_INCLUDED

// This macro is required for extension modules that are in several files.
// It must be defined before inclusion of numpy/arrayobject.h
#define PY_ARRAY_UNIQUE_SYMBOL DIFFPY_SRREAL_NUMPY_ARRAY_SYMBOL

#endif  // SRREAL_NUMPY_SYMBOL_HPP_INCLUDED
