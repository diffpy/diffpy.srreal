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

#include <string>
#include <valarray>
#include "srreal_converters.hpp"

// numpy/arrayobject.h needs to be included after srreal_converters.hpp,
// which defines PY_ARRAY_UNIQUE_SYMBOL.  NO_IMPORT_ARRAY indicates
// import_array will be called in the extension module initializer.
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

namespace srrealmodule {

/// helper for creating numpy array of doubles
NumPyArray_DoublePtr createNumPyDoubleArray(int dim, const int* sz)
{
    using namespace std;
    using namespace boost;
    // import numpy array once
    // copy the size information to an array of npy_intp
    valarray<npy_intp> npsza(dim);
    npy_intp& npsz = npsza[0];
    copy(sz, sz + dim, &npsz);
    // create numpy array
    python::object rvobj(
            python::handle<>(PyArray_SimpleNew(dim, &npsz, PyArray_DOUBLE)));
    double* rvdata = (double*) PyArray_DATA((PyArrayObject*) rvobj.ptr());
    NumPyArray_DoublePtr rv(rvobj, rvdata);
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
