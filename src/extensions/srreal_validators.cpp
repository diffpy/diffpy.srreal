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
#include <boost/python/import.hpp>

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


bool isiterable(boost::python::object obj)
{
    using boost::python::import;
#if PY_MAJOR_VERSION >= 3
    object Iterable = import("collections.abc").attr("Iterable");
#else
    object Iterable = import("collections").attr("Iterable");
#endif
    bool rv = (1 == PyObject_IsInstance(obj.ptr(), Iterable.ptr()));
    return rv;
}

}   // namespace srrealmodule

// End of file
