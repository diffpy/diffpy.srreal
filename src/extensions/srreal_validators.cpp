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

#include "srreal_validators.hpp"

namespace srrealmodule {

void ensure_index_bounds(int idx, int lo, int hi)
{
    if (idx < lo || idx >= hi)
    {
        throw nb::index_error("Index out of range.");
    }
}


void ensure_non_negative(int value)
{
    if (value < 0)
    {
        throw nb::value_error("Value cannot be negative.");
    }
}


bool isiterable(nb::object obj)
{
#if PY_MAJOR_VERSION >= 3
    nb::object Iterable = nb::module_::import_("collections.abc").attr("Iterable");
#else
    nb::object Iterable = nb::module_::import_("collections").attr("Iterable");
#endif
    bool rv = (1 == PyObject_IsInstance(obj.ptr(), Iterable.ptr()));
    return rv;
}

}   // namespace srrealmodule

// End of file
