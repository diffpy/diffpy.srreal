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

using diffpy::srreal::getPeakProfileTypes;
using diffpy::srreal::getPeakWidthModelTypes;
using diffpy::srreal::getScatteringFactorTableTypes;
using diffpy::srreal::PairQuantity;
using diffpy::srreal::BVSCalculator;
using diffpy::srreal::DebyePDFCalculator;
using diffpy::srreal::PDFCalculator;

using namespace diffpy::srreal_converters;
using namespace boost;

// Declaration of the external wrappers --------------------------------------

void wrap_BaseBondGenerator();

// Result converters ---------------------------------------------------------

namespace {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setsft_overloads,
        setScatteringFactorTable, 1, 1)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setpkf_overloads,
        setPeakProfile, 1, 1)

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
DECLARE_PYSET_FUNCTION_WRAPPER(getPeakProfileTypes,
        getPeakProfileTypes_asset)
DECLARE_PYSET_FUNCTION_WRAPPER(getPeakWidthModelTypes,
        getPeakWidthModelTypes_asset)
DECLARE_PYSET_FUNCTION_WRAPPER(getScatteringFactorTableTypes,
        getScatteringFactorTableTypes_asset)

}   // namespace

// Module Definitions --------------------------------------------------------

BOOST_PYTHON_MODULE(srreal_ext)
{
    using namespace boost::python;

    // initialize numpy arrays
    import_array();

    // execute external wrappers
    wrap_BaseBondGenerator();

    class_<PairQuantity>("PairQuantity")
        .def("_getDoubleAttr", &PairQuantity::getDoubleAttr)
        .def("_setDoubleAttr", &PairQuantity::setDoubleAttr)
        .def("_hasDoubleAttr", &PairQuantity::hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                namesOfDoubleAttributes_asset<PairQuantity>)
        .def("value", value_asarray<PairQuantity>)
        .def("eval", eval_asarray<PairQuantity>)
        ;

    // BVSCalculator

    class_<BVSCalculator, bases<PairQuantity> >("BVSCalculator")
        .def("valences", valences_asarray<BVSCalculator>)
        .def("bvdiff", bvdiff_asarray<BVSCalculator>)
        .def("bvmsdiff", &BVSCalculator::bvmsdiff)
        .def("bvrmsdiff", &BVSCalculator::bvrmsdiff)
        ;

    // DebyePDFCalculator

    def("getPeakProfileTypes", getPeakProfileTypes_asset);
    def("getPeakWidthModelTypes", getPeakWidthModelTypes_asset);
    def("getScatteringFactorTableTypes", getScatteringFactorTableTypes_asset);

    class_<DebyePDFCalculator, bases<PairQuantity> >("DebyePDFCalculator_ext")
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

    class_<PDFCalculator, bases<PairQuantity> >("PDFCalculator_ext")
        .def("getPDF", getPDF_asarray<PDFCalculator>)
        .def("getRDF", getRDF_asarray<PDFCalculator>)
        .def("getRgrid", getRgrid_asarray<PDFCalculator>)
        .def("setPeakProfile",
                (void(PDFCalculator::*)(const std::string&)) NULL,
                setpkf_overloads())
        .def("setScatteringFactorTable",
                (void(PDFCalculator::*)(const std::string&)) NULL,
                setsft_overloads())
        .def("getRadiationType",
                &PDFCalculator::getRadiationType,
                return_value_policy<copy_const_reference>())
        ;
}

// End of file
