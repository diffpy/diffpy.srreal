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
* srreal_ext - boost python interface to the srreal C++ codes in libdiffpy
*
* $Id$
*
*****************************************************************************/

#include <string>

#include <boost/python.hpp>

#include <diffpy/srreal/DebyePDFCalculator.hpp>
#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/BVSCalculator.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>
#include <diffpy/srreal/ScatteringFactorTable.hpp>

#include "srreal_converters.hpp"

using diffpy::srreal::PairQuantity;
using diffpy::srreal::DebyePDFCalculator;
using diffpy::srreal::PDFCalculator;
using diffpy::srreal::getScatteringFactorTableTypes;
using diffpy::srreal::BVSCalculator;
using namespace diffpy::srreal_converters;
using namespace boost;

// Result converters ---------------------------------------------------------

namespace {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setsft_overloads,
        setScatteringFactorTable, 1, 1)

// Common wrappers

DECLARE_PYARRAY_METHOD_WRAPPER(value, value_asarray)
DECLARE_PYSET_METHOD_WRAPPER(namesOfDoubleAttributes,
        namesOfDoubleAttributes_asset)

// PairQuantity::eval is a template non-constant method,
// so it needs a special wrapper

template <class T>
python::object eval_asarray(T& obj, const python::object& a)
{
    python::object rv = convertToNumPyArray(obj.eval(a));
    return rv;
}


// BVSCalculator wrappers

DECLARE_PYARRAY_METHOD_WRAPPER(valences, valences_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(bvdiff, bvdiff_asarray)

// DebyePDFCalculator and PDFCalculator wrappers

DECLARE_PYARRAY_METHOD_WRAPPER(getPDF, getPDF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getRDF, getRDF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getRgrid, getRgrid_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getF, getF_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(getQgrid, getQgrid_asarray)
DECLARE_PYSET_FUNCTION_WRAPPER(getScatteringFactorTableTypes,
        getScatteringFactorTableTypes_asset)

}   // namespace

// Module Definitions --------------------------------------------------------

BOOST_PYTHON_MODULE(srreal_ext)
{
    using namespace boost::python;

    // initialize numpy arrays
    import_array();

    // BVSCalculator

    class_<BVSCalculator>("BVSCalculator")
        .def("_getDoubleAttr", &BVSCalculator::getDoubleAttr)
        .def("_setDoubleAttr", &BVSCalculator::setDoubleAttr)
        .def("_hasDoubleAttr", &BVSCalculator::hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                namesOfDoubleAttributes_asset<BVSCalculator>)
        .def("value", value_asarray<BVSCalculator>)
        .def("eval", eval_asarray<BVSCalculator>)
        .def("valences", valences_asarray<BVSCalculator>)
        .def("bvdiff", bvdiff_asarray<BVSCalculator>)
        .def("bvmsdiff", &BVSCalculator::bvmsdiff)
        .def("bvrmsdiff", &BVSCalculator::bvrmsdiff)
        ;

    // DebyePDFCalculator

    def("getScatteringFactorTableTypes", getScatteringFactorTableTypes_asset);

    class_<DebyePDFCalculator>("DebyePDFCalculator_ext")
        .def("_getDoubleAttr", &DebyePDFCalculator::getDoubleAttr)
        .def("_setDoubleAttr", &DebyePDFCalculator::setDoubleAttr)
        .def("_hasDoubleAttr", &DebyePDFCalculator::hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                namesOfDoubleAttributes_asset<DebyePDFCalculator>)
        .def("value", value_asarray<DebyePDFCalculator>)
        .def("eval", eval_asarray<DebyePDFCalculator>)
        .def("getPDF", getPDF_asarray<DebyePDFCalculator>)
        .def("getRgrid", getRgrid_asarray<DebyePDFCalculator>)
        .def("getF", getF_asarray<DebyePDFCalculator>)
        .def("getQgrid", getQgrid_asarray<DebyePDFCalculator>)
        .def("setOptimumQstep", &DebyePDFCalculator::setOptimumQstep)
        .def("isOptimumQstep", &DebyePDFCalculator::isOptimumQstep)
        .def("setScatteringFactorTable",
                (void(DebyePDFCalculator::*)(const std::string&)) NULL,
                setsft_overloads())
        .def("getRadiationType",
                &DebyePDFCalculator::getRadiationType,
                return_value_policy<copy_const_reference>())
        ;
    // PDFCalculator

    class_<PDFCalculator>("PDFCalculator_ext")
        .def("_getDoubleAttr", &PDFCalculator::getDoubleAttr)
        .def("_setDoubleAttr", &PDFCalculator::setDoubleAttr)
        .def("_hasDoubleAttr", &PDFCalculator::hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                namesOfDoubleAttributes_asset<PDFCalculator>)
        .def("value", value_asarray<PDFCalculator>)
        .def("eval", eval_asarray<PDFCalculator>)
        .def("getPDF", getPDF_asarray<PDFCalculator>)
        .def("getRDF", getRDF_asarray<PDFCalculator>)
        .def("getRgrid", getRgrid_asarray<PDFCalculator>)
        .def("setScatteringFactorTable",
                (void(PDFCalculator::*)(const std::string&)) NULL,
                setsft_overloads())
        .def("getRadiationType",
                &PDFCalculator::getRadiationType,
                return_value_policy<copy_const_reference>())
        ;
}

// End of file
