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


#include <set>
#include <string>

#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <numpy/arrayobject.h>

#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>
#include <diffpy/srreal/ScatteringFactorTable.hpp>

using namespace boost;
using diffpy::srreal::PDFCalculator;
using diffpy::srreal::QuantityType;

namespace {

python::object convertQuantityType(const QuantityType& v)
{
    using std::copy;
    int vsize = v.size();
    python::object rv(
            python::handle<>(PyArray_SimpleNew(1, &vsize, PyArray_DOUBLE)));
    double* rvdata = (double*) PyArray_DATA((PyArrayObject*) rv.ptr());
    copy(v.begin(), v.end(), rvdata);
    return rv;
}


python::object convertSetOfStrings(const std::set<std::string>& s)
{
    using namespace std;
    python::object rv(python::handle<>(PySet_New(NULL)));
    std::set<string>::const_iterator si;
    for (si = s.begin(); si != s.end(); ++si)
    {
        rv.attr("add")(*si);
    }
    return rv;
}


python::object namesOfDoubleAttributes_asset(const PDFCalculator& obj)
{
    return convertSetOfStrings(obj.namesOfDoubleAttributes());
}


python::object getPDF_asarray(const PDFCalculator& obj)
{
    return convertQuantityType(obj.getPDF());
}


python::object getRDF_asarray(const PDFCalculator& obj)
{
    return convertQuantityType(obj.getRDF());
}


python::object getRgrid_asarray(const PDFCalculator& obj)
{
    return convertQuantityType(obj.getRgrid());
}


python::object eval_asarray(PDFCalculator& obj, const python::object& stru)
{
    return convertQuantityType(obj.eval(stru));
}

python::object getScatteringFactorTableTypes_asset()
{
    using diffpy::srreal::getScatteringFactorTableTypes;
    set<string> sftt = getScatteringFactorTableTypes();
    return convertSetOfStrings(sftt);
}


BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setsft_overloads,
        setScatteringFactorTable, 1, 1)

}   // namespace

BOOST_PYTHON_MODULE(pdf_ext)
{
    using namespace boost::python;
    // initialize numpy arrays
    import_array();

    def("getScatteringFactorTableTypes", getScatteringFactorTableTypes_asset);

    class_<PDFCalculator>("PDFCalculator")
        .def("_getDoubleAttr", &PDFCalculator::getDoubleAttr)
        .def("_setDoubleAttr", &PDFCalculator::setDoubleAttr)
        .def("_hasDoubleAttr", &PDFCalculator::hasDoubleAttr)
        .def("_namesOfDoubleAttributes", namesOfDoubleAttributes_asset)
        .def("getPDF", getPDF_asarray)
        .def("getRDF", getRDF_asarray)
        .def("getRgrid", getRgrid_asarray)
        .def("eval", eval_asarray)
        .def("setScatteringFactorTable",
                (void(PDFCalculator::*)(const string&)) NULL,
                setsft_overloads())
        .def("getRadiationType",
                &PDFCalculator::getRadiationType,
                return_value_policy<copy_const_reference>())
        ;
}
