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
* pdf_ext - boost python wrap to PDF related C++ classes and functions
*
* $Id$
*
*****************************************************************************/

#include <string>

#include <boost/python.hpp>

#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>
#include <diffpy/srreal/ScatteringFactorTable.hpp>
#include "srreal_converters.hpp"


// Helpers -------------------------------------------------------------------

namespace {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setsft_overloads,
        setScatteringFactorTable, 1, 1)

}   // namespace

// Module Definitions --------------------------------------------------------

BOOST_PYTHON_MODULE(pdf_ext)
{
    using std::string;
    using namespace boost::python;
    using diffpy::srreal::PDFCalculator;
    using diffpy::srreal::getScatteringFactorTableTypes;

    // initialize converters
    initialize_srreal_converters();

    def("getScatteringFactorTableTypes", getScatteringFactorTableTypes);

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
                (void(PDFCalculator::*)(const string&)) NULL,
                setsft_overloads())
        .def("getRadiationType",
                &PDFCalculator::getRadiationType,
                return_value_policy<copy_const_reference>())
        ;
}

// End of file
