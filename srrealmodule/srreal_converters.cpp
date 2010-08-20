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
* utilities for conversion between C++ and Python types.
* boost python converters are prone to conflicts.
*
* $Id$
*
*****************************************************************************/

#include <string>
#include "srreal_converters.hpp"

namespace srrealmodule {

/// helper function that holds the flag for accomplished numpy import
bool& did_import_array()
{
    static bool flag = false;
    return flag;
}


/// helper for raising RuntimeError on a call of pure virtual function
void throwPureVirtualCalled(const char* fncname)
{
    std::string emsg = "Pure virtual function '";
    emsg += fncname;
    emsg += "' called.";
    PyErr_SetString(PyExc_RuntimeError, emsg.c_str());
    boost::python::throw_error_already_set();
}

}   // namespace srrealmodule

// End of file
