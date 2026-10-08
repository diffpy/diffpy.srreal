/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   (c) 2016 Brookhaven Science Associates,
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
* Utilities for wrapping classes that derive from HasClassRegistry.
*
*****************************************************************************/

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace srrealmodule {

void register_for_cleanup(PyObject* pobj)
{
    nb::object obj = nb::borrow(pobj);
    nb::object reg = nb::module_::import_("diffpy.srreal._cleanup").attr("registerForCleanUp");
    reg(obj);
}


/// get dictionary of Python-defined docstrings for the cls class.
nb::object get_registry_docstrings(nb::object& cls)
{
    nb::object mod = nb::module_::import_("diffpy.srreal._docstrings");
    nb::object getdocs = mod.attr("get_registry_docstrings");
    nb::object rv = getdocs(cls);
    return rv;
}


}   // namespace srrealmodule

// End of file
