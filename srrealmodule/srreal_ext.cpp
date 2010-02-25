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
* srreal_ext - boost python interface to srreal function in libdiffpy
*
* $Id$
*
*****************************************************************************/

#include <string>

#include <boost/python.hpp>

#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/BVSCalculator.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>
#include <diffpy/srreal/ScatteringFactorTable.hpp>

#include "srreal_converters.hpp"


// Helpers -------------------------------------------------------------------

namespace {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setsft_overloads,
        setScatteringFactorTable, 1, 1)

}   // namespace

// Module Definitions --------------------------------------------------------

BOOST_PYTHON_MODULE(srreal_ext)
{
    using namespace boost::python;

    // initialize converters
    register_srreal_converters();

    // BVSCalculator

    using diffpy::srreal::BVSCalculator;

    class_<BVSCalculator>("BVSCalculator")
        .def("_getDoubleAttr", &BVSCalculator::getDoubleAttr)
        .def("_setDoubleAttr", &BVSCalculator::setDoubleAttr)
        .def("_hasDoubleAttr", &BVSCalculator::hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                &BVSCalculator::namesOfDoubleAttributes)
        .def("valences", &BVSCalculator::valences)
        .def("bvmsdiff", &BVSCalculator::bvmsdiff)
        .def("bvrmsdiff", &BVSCalculator::bvrmsdiff)
        .def("eval", &BVSCalculator::eval<object>,
                return_value_policy<copy_const_reference>())
        ;

    // PDFCalculator

    using diffpy::srreal::getScatteringFactorTableTypes;
    def("getScatteringFactorTableTypes", getScatteringFactorTableTypes);

    using diffpy::srreal::PDFCalculator;
    class_<PDFCalculator>("PDFCalculator")
        .def("_getDoubleAttr", &PDFCalculator::getDoubleAttr)
        .def("_setDoubleAttr", &PDFCalculator::setDoubleAttr)
        .def("_hasDoubleAttr", &PDFCalculator::hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                &PDFCalculator::namesOfDoubleAttributes)
        .def("getPDF", &PDFCalculator::getPDF)
        .def("getRDF", &PDFCalculator::getRDF)
        .def("getRgrid", &PDFCalculator::getRgrid)
        .def("eval", &PDFCalculator::eval<object>,
                return_value_policy<copy_const_reference>())
        .def("setScatteringFactorTable",
                (void(PDFCalculator::*)(const std::string&)) NULL,
                setsft_overloads())
        .def("getRadiationType",
                &PDFCalculator::getRadiationType,
                return_value_policy<copy_const_reference>())
        ;
}

// End of file
