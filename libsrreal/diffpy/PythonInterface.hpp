/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Common functions for interfacing with Python interpreter.
*
* $Id$
*
*****************************************************************************/

#ifndef PYTHONINTERFACE_HPP_INCLUDED
#define PYTHONINTERFACE_HPP_INCLUDED

#include <Python.h>
#include <string>

namespace diffpy {

// routines

void initializePython(int py_argc=0, char* py_argv[]=NULL);
std::string getPythonErrorString();

}   // namespace diffpy

#endif  // PYTHONINTERFACE_HPP_INCLUDED
