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
* shared converters between C++ and python types.
*
* $Id$
*
*****************************************************************************/

#include <set>
#include <string>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include <diffpy/srreal/PairQuantity.hpp>
#include "srreal_converters.hpp"

using namespace boost;
using diffpy::srreal::QuantityType;


namespace {

struct QuantityType_to_numpyarray
{
    static PyObject* convert(const QuantityType& v)
    {
        using std::copy;
        npy_intp vsize = v.size();
        PyObject* rv = PyArray_SimpleNew(1, &vsize, PyArray_DOUBLE);
        double* rvdata = static_cast<double*>(
                PyArray_DATA(reinterpret_cast<PyArrayObject*>(rv)));
        copy(v.begin(), v.end(), rvdata);
        return rv;
    }

    static PyTypeObject const* get_pytype()
    {
        return &PyDoubleArrType_Type;
    }

};


template <class C>
struct stl_to_pyset
{
    static PyObject* convert(const C& v)
    {
        python::object rvset(python::handle<>(PySet_New(NULL)));
        python::object rvset_add = rvset.attr("add");
        typename C::const_iterator ii;
        for (ii = v.begin(); ii != v.end(); ++ii)  rvset_add(*ii);
        return python::incref(rvset.ptr());
    }
};

}   // namespace


void initialize_srreal_converters()
{
    static bool did_initialize = false;
    if (did_initialize)   return;
    // initialize numpy arrays
    import_array();
    // Data type converters
    using python::to_python_converter;
    to_python_converter<QuantityType, QuantityType_to_numpyarray>();
    typedef std::set<std::string> stlsetstring;
    to_python_converter<stlsetstring, stl_to_pyset<stlsetstring> >();
    did_initialize = true;
}

// End of file.
