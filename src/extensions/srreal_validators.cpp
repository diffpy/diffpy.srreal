/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   (c) 2013 Brookhaven Science Associates,
*                   Brookhaven National Laboratory.
*                   All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Helper functions for argument checking.
*
*****************************************************************************/

#include <boost/python/errors.hpp>

#include "srreal_validators.hpp"

namespace srrealmodule {

using namespace boost::python;

void ensure_index_bounds(int idx, int lo, int hi)
{
    if (idx < lo || idx >= hi)
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range.");
        throw_error_already_set();
    }
}


void ensure_non_negative(int value)
{
    if (value < 0)
    {
        PyErr_SetString(PyExc_ValueError, "Value cannot be negative.");
        throw_error_already_set();
    }
}

}   // namespace srrealmodule

// End of file
