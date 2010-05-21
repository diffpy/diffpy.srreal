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
* Bindings to the BaseBondGenerator class.  So far the wrapper is intended
* only for accessing the C++ created BaseBondGenerator instances and there
* is no support for method overrides from Python.
*
* $Id$
*
*****************************************************************************/

#include <string>
#include <boost/python.hpp>

#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_BaseBondGenerator {

DECLARE_PYARRAY_METHOD_WRAPPER(r0, r0_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(r1, r1_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(r01, r01_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(Ucartesian0, Ucartesian0_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(Ucartesian1, Ucartesian1_asarray)

}   // namespace nswrap_BaseBondGenerator

// Wrapper definition --------------------------------------------------------

void wrap_BaseBondGenerator()
{
    using namespace boost::python;
    using namespace diffpy::srreal;
    using namespace nswrap_BaseBondGenerator;

    class_<BaseBondGenerator>("BaseBondGenerator",
            init<const StructureAdapter*>())
        .def("rewind", &BaseBondGenerator::rewind)
        .def("finished", &BaseBondGenerator::finished)
        .def("next", &BaseBondGenerator::next)
        .def("nextsite", &BaseBondGenerator::nextsite)
        .def("selectAnchorSite", &BaseBondGenerator::selectAnchorSite)
        .def("selectSiteRange", &BaseBondGenerator::selectSiteRange)
        .def("setRmin", &BaseBondGenerator::setRmin)
        .def("setRmax", &BaseBondGenerator::setRmax)
        .def("getRmin", &BaseBondGenerator::getRmin,
                return_value_policy<copy_const_reference>())
        .def("getRmax", &BaseBondGenerator::getRmax,
                return_value_policy<copy_const_reference>())
        .def("site0", &BaseBondGenerator::site0)
        .def("site1", &BaseBondGenerator::site1)
        .def("multiplicity", &BaseBondGenerator::multiplicity)
        .def("r0", r0_asarray<BaseBondGenerator>)
        .def("r1", r1_asarray<BaseBondGenerator>)
        .def("distance", &BaseBondGenerator::distance)
        .def("r01", r01_asarray<BaseBondGenerator>)
        .def("Ucartesian0", Ucartesian0_asarray<BaseBondGenerator>)
        .def("Ucartesian1", Ucartesian1_asarray<BaseBondGenerator>)
        .def("msd", &BaseBondGenerator::msd)
        ;
}

}   // namespace srrealmodule

// End of file
