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
* utilities for conversion between C++ and Python types.
* boost python converters are prone to conflicts.
*
* $Id$
*
*****************************************************************************/

#include <boost/python/exception_translator.hpp>
#include <string>
#include <valarray>
#include <stdexcept>

#include <diffpy/Attributes.hpp>

#include "srreal_converters.hpp"
// numpy/arrayobject.h needs to be included after srreal_converters.hpp,
// which defines PY_ARRAY_UNIQUE_SYMBOL.  NO_IMPORT_ARRAY indicates
// import_array will be called in the extension module initializer.
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>


namespace {

// exception translations ----------------------------------------------------

using diffpy::attributes::DoubleAttributeError;
using std::invalid_argument;

void translate_DoubleAttributeError(const DoubleAttributeError& e)
{
    PyErr_SetString(PyExc_AttributeError, e.what());
}


void translate_invalid_argument(const invalid_argument& e)
{
    PyErr_SetString(PyExc_ValueError, e.what());
}


boost::python::object newNumPyArray(int dim, const int* sz, int typenum)
{
    using namespace std;
    using namespace boost;
    // copy the size information to an array of npy_intp
    valarray<npy_intp> npsza(dim);
    npy_intp& npsz = npsza[0];
    copy(sz, sz + dim, &npsz);
    // create numpy array
    python::object rv(
            python::handle<>(PyArray_SimpleNew(dim, &npsz, typenum)));
    return rv;
}

}   // namespace

namespace srrealmodule {

/// this function registers all exception translators
void wrap_exceptions()
{
    using boost::python::register_exception_translator;
    register_exception_translator<DoubleAttributeError>(
            &translate_DoubleAttributeError);
    register_exception_translator<invalid_argument>(
            &translate_invalid_argument);
}


/// helper for creating numpy array of doubles
NumPyArray_DoublePtr createNumPyDoubleArray(int dim, const int* sz)
{
    boost::python::object rvobj = newNumPyArray(dim, sz, PyArray_DOUBLE);
    double* rvdata = static_cast<double*>(PyArray_DATA(rvobj.ptr()));
    NumPyArray_DoublePtr rv(rvobj, rvdata);
    return rv;
}


/// helper for creating numpy array of integers
NumPyArray_IntPtr createNumPyIntArray(int dim, const int* sz)
{
    boost::python::object rvobj = newNumPyArray(dim, sz, PyArray_INT);
    int* rvdata = static_cast<int*>(PyArray_DATA(rvobj.ptr()));
    NumPyArray_IntPtr rv(rvobj, rvdata);
    return rv;
}


/// helper for raising RuntimeError on a call of pure virtual function
void throwPureVirtualCalled(const char* fncname)
{
    std::string emsg = "Pure virtual function '";
    emsg += fncname;
    emsg += "' called.";
    PyErr_SetString(PyExc_RuntimeError, emsg.c_str());
    boost::python::throw_error_already_set();
}

}   // namespace srrealmodule

// End of file
