/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* utilities for conversion between C++ and Python types.
* boost python converters are prone to conflicts.
*
*****************************************************************************/

#include <boost/python/slice.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/exception_translator.hpp>

#include <string>
#include <valarray>
#include <stdexcept>
#include <cassert>
#include <cstdlib>

#include <diffpy/Attributes.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>

#include "srreal_converters.hpp"

#include "srreal_numpy_symbol.hpp"
// numpy/arrayobject.h needs to be included after srreal_numpy_symbol.hpp,
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


bool isiterable(boost::python::object obj)
{
    using namespace boost::python;
    object Iterable = import("collections").attr("Iterable");
    bool rv = (1 == PyObject_IsInstance(obj.ptr(), Iterable.ptr()));
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
    boost::python::object rvobj = newNumPyArray(dim, sz, NPY_DOUBLE);
    PyArrayObject* a = reinterpret_cast<PyArrayObject*>(rvobj.ptr());
    double* rvdata = static_cast<double*>(PyArray_DATA(a));
    NumPyArray_DoublePtr rv(rvobj, rvdata);
    return rv;
}


/// helper for creating numpy array of the same shape as the argument
NumPyArray_DoublePtr createNumPyDoubleArrayLike(boost::python::object& obj)
{
    assert(PyArray_Check(obj.ptr()));
    PyArrayObject* a = reinterpret_cast<PyArrayObject*>(obj.ptr());
    // create numpy array
    boost::python::object rvobj(
            boost::python::handle<>(
                PyArray_NewLikeArray(
                    a, NPY_CORDER, PyArray_DescrFromType(NPY_DOUBLE), 0)));
    PyArrayObject* a1 = reinterpret_cast<PyArrayObject*>(rvobj.ptr());
    double* rvdata = static_cast<double*>(PyArray_DATA(a1));
    NumPyArray_DoublePtr rv(rvobj, rvdata);
    return rv;
}


/// helper for creating a numpy array view on a double array
boost::python::object
createNumPyDoubleView(double* data, int dim, const int* sz)
{
    using namespace std;
    using namespace boost;
    valarray<npy_intp> npsza(dim);
    npy_intp* npsz = &(npsza[0]);
    copy(sz, sz + dim, npsz);
    python::object rv(
            python::handle<>(
                PyArray_SimpleNewFromData(dim, npsz, NPY_DOUBLE, data)));
    return rv;
}


/// NumPy array view specializations for R3::Vector
boost::python::object viewAsNumPyArray(::diffpy::srreal::R3::Vector& v)
{
    using namespace diffpy::srreal;
    double* data = &(v[0]);
    int sz = R3::Ndim;
    return createNumPyDoubleView(data, 1, &sz);
}


/// NumPy array view specializations for R3::Matrix
boost::python::object viewAsNumPyArray(::diffpy::srreal::R3::Matrix& mx)
{
    using namespace diffpy::srreal;
    double* data = &(mx(0, 0));
    int sz[2] = {R3::Ndim, R3::Ndim};
    return createNumPyDoubleView(data, 2, sz);
}


/// Copy NumPy array to R3::Vector
void assignR3Vector(
        ::diffpy::srreal::R3::Vector& dst, boost::python::object& value)
{
    using namespace boost;
    using diffpy::srreal::R3::Ndim;
    // If value is numpy array, try direct data access
    if (PyArray_Check(value.ptr()))
    {
        PyArrayObject* a = reinterpret_cast<PyArrayObject*>(
                PyArray_ContiguousFromAny(value.ptr(), NPY_DOUBLE, 1, 1));
        if (a && Ndim == PyArray_DIM(a, 0))
        {
            double* p = static_cast<double*>(PyArray_DATA(a));
            std::copy(p, p + Ndim, dst.data().begin());
            Py_DECREF(a);
            return;
        }
        Py_XDECREF(a);
    }
    // handle scalar assignment
    python::extract<double> getvalue(value);
    if (getvalue.check())
    {
        std::fill(dst.data().begin(), dst.data().end(), getvalue());
        return;
    }
    // finally assign using array view
    python::object dstview = viewAsNumPyArray(dst);
    dstview[python::slice()] = value;
}


/// Copy possible NumPy array to R3::Matrix
void assignR3Matrix(
        ::diffpy::srreal::R3::Matrix& dst, boost::python::object& value)
{
    using namespace boost;
    using diffpy::srreal::R3::Ndim;
    // If value is numpy array, try direct data access
    if (PyArray_Check(value.ptr()))
    {
        PyArrayObject* a = reinterpret_cast<PyArrayObject*>(
                PyArray_ContiguousFromAny(value.ptr(), NPY_DOUBLE, 2, 2));
        if (a && Ndim == PyArray_DIM(a, 0) && Ndim == PyArray_DIM(a, 1))
        {
            double* p = static_cast<double*>(PyArray_DATA(a));
            std::copy(p, p + Ndim * Ndim, dst.data().begin());
            Py_DECREF(a);
            return;
        }
        Py_XDECREF(a);
    }
    // handle scalar assignment
    python::extract<double> getvalue(value);
    if (getvalue.check())
    {
        std::fill(dst.data().begin(), dst.data().end(), getvalue());
        return;
    }
    // finally assign using array view
    python::object dstview = viewAsNumPyArray(dst);
    dstview[python::slice()] = value;
}


/// helper for creating numpy array of integers
NumPyArray_IntPtr createNumPyIntArray(int dim, const int* sz)
{
    boost::python::object rvobj = newNumPyArray(dim, sz, NPY_INT);
    PyArrayObject* a = reinterpret_cast<PyArrayObject*>(rvobj.ptr());
    int* rvdata = static_cast<int*>(PyArray_DATA(a));
    NumPyArray_IntPtr rv(rvobj, rvdata);
    return rv;
}


/// efficient conversion of Python object to a QuantityType
diffpy::srreal::QuantityType&
extractQuantityType(
        boost::python::object obj,
        diffpy::srreal::QuantityType& rv)
{
    using namespace boost;
    using diffpy::srreal::QuantityType;
    // extract QuantityType directly
    python::extract<QuantityType&> getqt(obj);
    if (getqt.check())  return getqt();
    // copy data directly if it is a numpy array of doubles
    PyArrayObject* a = PyArray_Check(obj.ptr()) ?
        reinterpret_cast<PyArrayObject*>(obj.ptr()) : NULL;
    bool isdoublenumpyarray = a &&
        (1 == PyArray_NDIM(a)) &&
        (NPY_DOUBLE == PyArray_TYPE(a));
    if (isdoublenumpyarray)
    {
        double* src = static_cast<double*>(PyArray_DATA(a));
        npy_intp stride = PyArray_STRIDE(a, 0) / PyArray_ITEMSIZE(a);
        rv.resize(PyArray_SIZE(a));
        QuantityType::iterator dst = rv.begin();
        for (; dst != rv.end(); ++dst, src += stride)  *dst = *src;
        return rv;
    }
    // otherwise copy elementwise converting each element to a double
    python::stl_input_iterator<double> begin(obj), end;
    rv.assign(begin, end);
    return rv;
}


/// efficient conversion of Python object to a numpy array of doubles
NumPyArray_DoublePtr extractNumPyDoubleArray(::boost::python::object& obj)
{
    PyObject* pobj = PyArray_ContiguousFromAny(obj.ptr(), NPY_DOUBLE, 0, 0);
    if (!pobj)
    {
        const char* emsg = "Cannot convert this object to numpy array.";
        PyErr_SetString(PyExc_TypeError, emsg);
        boost::python::throw_error_already_set();
        abort();
    }
    boost::python::object rvobj((boost::python::handle<>(pobj)));
    PyArrayObject* a = reinterpret_cast<PyArrayObject*>(pobj);
    double* rvdata = static_cast<double*>(PyArray_DATA(a));
    NumPyArray_DoublePtr rv(rvobj, rvdata);
    return rv;
}


/// extract integer with a support for numpy.int types
int extractint(boost::python::object obj)
{
    using namespace boost;
    python::extract<int> geti(obj);
    if (geti.check())  return geti();
    PyObject* pobj = obj.ptr();
    if (PyArray_CheckScalar(pobj))
    {
        int rv = PyArray_PyIntAsInt(pobj);
        return rv;
    }
    // nothing worked, call geti which will raise an exception
    return geti();
}


/// extract a vector of integers from a numpy array, iterable or scalar
std::vector<int> extractintvector(boost::python::object obj)
{
    using namespace boost::python;
    std::vector<int> rv;
    // iterable of integers
    if (isiterable(obj))
    {
        PyArrayObject* a = PyArray_Check(obj.ptr()) ?
            reinterpret_cast<PyArrayObject*>(obj.ptr()) : NULL;
        // handle numpy array of integers
        bool isintegernumpyarray =
            a && (1 == PyArray_NDIM(a)) && PyArray_ISINTEGER(a);
        if (isintegernumpyarray)
        {
            object aobj = obj;
            if (NPY_INT != PyArray_TYPE(a))
            {
                object a1(handle<>(PyArray_Cast(a, NPY_INT)));
                aobj = a1;
            }
            PyArrayObject* a1 = reinterpret_cast<PyArrayObject*>(aobj.ptr());
            assert(NPY_INT == PyArray_TYPE(a1));
            int* pfirst = static_cast<int*>(PyArray_DATA(a1));
            int* plast = pfirst + PyArray_SIZE(a1);
            rv.assign(pfirst, plast);
            return rv;
        }
        // otherwise translate every item in the iterable
        stl_input_iterator<object> ii(obj), end;
        rv.reserve(len(obj));
        for (; ii != end; ++ii)
        {
            int idx = extractint(*ii);
            rv.push_back(idx);
        }
        return rv;
    }
    // try to handle it as a scalar
    int idx = extractint(obj);
    rv.push_back(idx);
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


namespace diffpy {
namespace srreal {

/// shared converter that first tries to extract the pointer and then calls
/// diffpy.srreal.structureadapter.createStructureAdapter
StructureAdapterPtr createStructureAdapter(::boost::python::object stru)
{
    using namespace boost::python;
    StructureAdapterPtr adpt;
    extract<StructureAdapterPtr> getadpt(stru);
    if (getadpt.check())  adpt = getadpt();
    else
    {
        object mod = import("diffpy.srreal.structureadapter");
        object convertinpython = mod.attr("createStructureAdapter");
        adpt = extract<StructureAdapterPtr>(convertinpython(stru));
    }
    return adpt;
}

}   // namespace srreal
}   // namespace diffpy

// End of file
