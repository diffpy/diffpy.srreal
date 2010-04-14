/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Bindings to the diffpy::Attributes class.
*
* $Id$
*
*****************************************************************************/

#include <string>
#include <boost/python.hpp>

#include <diffpy/Attributes.hpp>
#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_Attributes {

DECLARE_PYSET_METHOD_WRAPPER(namesOfDoubleAttributes,
        namesOfDoubleAttributes_asset)


const char* getattr_setattr_code = "\
def __getattr__(self, name):\n\
    try:\n\
        rv = self._getDoubleAttr(name)\n\
    except Exception, e:\n\
        raise AttributeError(e)\n\
    return rv\n\
\n\
\n\
def __setattr__(self, name, value):\n\
    if self._hasDoubleAttr(name):\n\
        self._setDoubleAttr(name, value)\n\
    else:\n\
        object.__setattr__(self, name, value)\n\
    return\n\
";


}   // namespace nswrap_Attributes

// Wrapper definition --------------------------------------------------------

void wrap_Attributes()
{
    using diffpy::Attributes;
    using namespace boost::python;
    using namespace nswrap_Attributes;
    // store custom __getattr__ and __setattr__ in the locals dictionary
    object globals = import("__main__").attr("__dict__");
    dict locals;
    exec(getattr_setattr_code, globals, locals);
    // ready for class definition
    class_<Attributes>("Attributes")
        .def("_getDoubleAttr", &Attributes::getDoubleAttr)
        .def("_setDoubleAttr", &Attributes::setDoubleAttr)
        .def("_hasDoubleAttr", &Attributes::hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                namesOfDoubleAttributes_asset<Attributes>)
        .def("__getattr__", locals["__getattr__"])
        .def("__setattr__", locals["__setattr__"])
        ;
}

}   // namespace srrealmodule

// End of file
