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


#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <numpy/arrayobject.h>

#include <diffpy/srreal/PDFCalculator.hpp>

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


python::object getPDFarray(const PDFCalculator& obj)
{
    return convertQuantityType(obj.getPDF());
}


python::object getRDFarray(const PDFCalculator& obj)
{
    return convertQuantityType(obj.getRDF());
}


python::object getRgridarray(const PDFCalculator& obj)
{
    return convertQuantityType(obj.getRgrid());
}

}   // namespace


BOOST_PYTHON_MODULE(pdf_ext)
{
    using namespace boost::python;
    // initialize numpy arrays
    import_array();

    class_<PDFCalculator>("PDFCalculator")
        .def("_getDoubleAttr", &PDFCalculator::getDoubleAttr)
        .def("_setDoubleAttr", &PDFCalculator::setDoubleAttr)
        .def("_namesOfDoubleAttributes",
                &PDFCalculator::namesOfDoubleAttributes)
        .def("getPDF", getPDFarray)
        .def("getRDF", getRDFarray)
        .def("getRgrid", getRgridarray)
        ;
}
