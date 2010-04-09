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

}   // namespace nswrap_Attributes

// Wrapper definition --------------------------------------------------------

void wrap_Attributes()
{
    using diffpy::Attributes;
    using namespace boost::python;
    using namespace nswrap_Attributes;

    class_<Attributes>("Attributes")
        .def("_getDoubleAttr", &Attributes::getDoubleAttr)
        .def("_setDoubleAttr", &Attributes::setDoubleAttr)
        .def("_hasDoubleAttr", &Attributes::hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                namesOfDoubleAttributes_asset<Attributes>)
        ;

}

}   // namespace srrealmodule

// End of file
