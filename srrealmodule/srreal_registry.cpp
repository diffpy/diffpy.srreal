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

#include <boost/python/object.hpp>
#include <boost/python/import.hpp>

namespace srrealmodule {

using namespace boost::python;

void register_for_cleanup(PyObject* pobj)
{
    object obj(borrowed(pobj));
    object reg = import("diffpy.srreal._cleanup").attr("registerForCleanUp");
    reg(obj);
}


/// get dictionary of Python-defined docstrings for the cls class.
object get_registry_docstrings(object& cls)
{
    object mod = import("diffpy.srreal._docstrings");
    object getdocs = mod.attr("get_registry_docstrings");
    object rv = getdocs(cls);
    return rv;
}


}   // namespace srrealmodule

// End of file
