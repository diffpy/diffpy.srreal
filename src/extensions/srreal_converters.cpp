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

// TODO: replace Py_DECREF with nb::steal

#include <string>
#include <stdexcept>
#include <cassert>
#include <cstdlib>
#include <limits>
#include <ranges>

#include <diffpy/Attributes.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>

#include "srreal_converters.hpp"
#include "srreal_validators.hpp"

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


nb::object newNumPyArray(int dim, const int* sz, int typenum)
{
    std::vector<npy_intp> dims(static_cast<size_t>(dim));
    std::transform(
        sz,
        sz + dim,
        dims.begin(),
        [](int v) { return static_cast<npy_intp>(v); }
    );
    
    PyObject *arr = PyArray_SimpleNew(
        dim,
        dims.empty() ? nullptr : dims.data(),
        typenum
    );

    if (!arr)
        throw nb::python_error();
    
    return nb::steal<nb::object>(arr);
}

}   // namespace

namespace srrealmodule {

/// this function registers all exception translators
void wrap_exceptions()
{
    nb::register_exception_translator(
        [](const std::exception_ptr& p, void*) 
        {
            try 
            {
                if (p)
                    std::rethrow_exception(p);
            } 
            catch (const DoubleAttributeError& e) 
            {
                PyErr_SetString(PyExc_AttributeError, e.what());
            } 
            catch (const invalid_argument& e) 
            {
                PyErr_SetString(PyExc_ValueError, e.what());
            }
        }
    );
}


/// helper for creating numpy array of doubles
NumPyArray_DoublePtr createNumPyDoubleArray(int dim, const int* sz)
{
    nb::object rvobj = newNumPyArray(dim, sz, NPY_DOUBLE);
    PyArrayObject* a = reinterpret_cast<PyArrayObject*>(rvobj.ptr());
    double* rvdata = static_cast<double*>(PyArray_DATA(a));
    NumPyArray_DoublePtr rv(rvobj, rvdata);
    return rv;
}


/// helper for creating numpy array of the same shape as the argument
NumPyArray_DoublePtr createNumPyDoubleArrayLike(nb::object& obj)
{
    assert(PyArray_Check(obj.ptr()));
    PyArrayObject* a = reinterpret_cast<PyArrayObject*>(obj.ptr());
    // create numpy array
    PyObject *pobj = PyArray_NewLikeArray(
        a,
        NPY_CORDER,
        PyArray_DescrFromType(NPY_DOUBLE),
        0
    );

    if (!pobj)
        throw nb::python_error();

    nb::object rvobj = nb::steal<nb::object>(pobj);
    PyArrayObject* a1 = reinterpret_cast<PyArrayObject*>(rvobj.ptr());
    double* rvdata = static_cast<double*>(PyArray_DATA(a1));
    NumPyArray_DoublePtr rv(rvobj, rvdata);
    return rv;
}


/// helper for creating a numpy array view on a double array
nb::object
createNumPyDoubleView(double* data, int dim, const int* sz)
{
    std::vector<npy_intp> dims(static_cast<size_t>(dim));
    std::transform(
        sz,
        sz + dim,
        dims.begin(),
        [](int v) { return static_cast<npy_intp>(v); }
    );

    PyObject* pobj = PyArray_SimpleNewFromData(
        dim,
        dims.data(),
        NPY_DOUBLE,
        data
    );

    if (!pobj)
        throw nb::python_error();

    return nb::steal<nb::object>(pobj);
}


/// NumPy array view specializations for R3::Vector
nb::object viewAsNumPyArray(::diffpy::srreal::R3::Vector& v)
{
    using namespace diffpy::srreal;
    double* data = &(v[0]);
    int sz = R3::Ndim;
    return createNumPyDoubleView(data, 1, &sz);
}


/// NumPy array view specializations for R3::Matrix
nb::object viewAsNumPyArray(::diffpy::srreal::R3::Matrix& mx)
{
    using namespace diffpy::srreal;
    double* data = &(mx(0, 0));
    int sz[2] = {R3::Ndim, R3::Ndim};
    return createNumPyDoubleView(data, 2, sz);
}


/// Copy NumPy array to R3::Vector
void assignR3Vector(
        ::diffpy::srreal::R3::Vector& dst, nb::object& value)
{
    using diffpy::srreal::R3::Ndim;
    // If value is numpy array, try direct data access
    if (PyArray_Check(value.ptr()))
    {
        PyArrayObject* a = reinterpret_cast<PyArrayObject*>(
                PyArray_ContiguousFromAny(value.ptr(), NPY_DOUBLE, 1, 1));
        if (a && Ndim == PyArray_DIM(a, 0))
        {
            double* p = static_cast<double*>(PyArray_DATA(a));
            std::ranges::copy(p, p + Ndim, dst.data().begin());
            Py_DECREF(a);
            return;
        }
        Py_XDECREF(a);
    }
    // handle scalar assignment
    double scalar;
    if (nb::try_cast<double>(value, scalar))
    {
        std::ranges::fill(dst.data().begin(), dst.data().end(), scalar);
        return;
    }
    // finally assign using array view
    nb::object dstview = viewAsNumPyArray(dst);
    dstview[nb::slice(nb::none(), nb::none(), nb::none())] = value;
}


/// Copy possible NumPy array to R3::Matrix
void assignR3Matrix(
        ::diffpy::srreal::R3::Matrix& dst, nb::object& value)
{
    using diffpy::srreal::R3::Ndim;
    // If value is numpy array, try direct data access
    if (PyArray_Check(value.ptr()))
    {
        PyArrayObject* a = reinterpret_cast<PyArrayObject*>(
                PyArray_ContiguousFromAny(value.ptr(), NPY_DOUBLE, 2, 2));
        if (a && Ndim == PyArray_DIM(a, 0) && Ndim == PyArray_DIM(a, 1))
        {
            double* p = static_cast<double*>(PyArray_DATA(a));
            std::ranges::copy(p, p + Ndim * Ndim, dst.data().begin());
            Py_DECREF(a);
            return;
        }
        Py_XDECREF(a);
    }
    // handle scalar assignment
    double scalar;
    if (nb::try_cast<double>(value, scalar))
    {
        std::ranges::fill(dst.data().begin(), dst.data().end(), scalar);
        return;
    }
    // finally assign using array view
    nb::object dstview = viewAsNumPyArray(dst);
    dstview[nb::slice(nb::none(), nb::none(), nb::none())] = value;
}


/// helper for creating numpy array of integers
NumPyArray_IntPtr createNumPyIntArray(int dim, const int* sz)
{
    nb::object rvobj = newNumPyArray(dim, sz, NPY_INT);
    PyArrayObject* a = reinterpret_cast<PyArrayObject*>(rvobj.ptr());
    int* rvdata = static_cast<int*>(PyArray_DATA(a));
    NumPyArray_IntPtr rv(rvobj, rvdata);
    return rv;
}


/// efficient conversion of Python object to a QuantityType
diffpy::srreal::QuantityType&
extractQuantityType(
        nb::object obj,
        diffpy::srreal::QuantityType& rv)
{
    using diffpy::srreal::QuantityType;
    // extract QuantityType directly
    QuantityType* qt = nullptr;
    if (nb::try_cast<QuantityType*>(obj, qt) && qt)
        return *qt;
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
    std::vector<double> tmp;

    PyObject* iter = PyObject_GetIter(obj.ptr());
    if (!iter)
        throw nb::python_error();

    nb::object it = nb::steal<nb::object>(iter);

    while (PyObject* item = PyIter_Next(it.ptr())) 
    {
        nb::object item_obj = nb::steal<nb::object>(item);
        tmp.push_back(nb::cast<double>(item_obj));
    }

    if (PyErr_Occurred())
        throw nb::python_error();

    rv.assign(tmp.begin(), tmp.end());
    return rv;
}


/// efficient conversion of Python object to a numpy array of doubles
NumPyArray_DoublePtr extractNumPyDoubleArray(::nb::object& obj)
{
    PyObject* pobj = PyArray_ContiguousFromAny(obj.ptr(), NPY_DOUBLE, 0, 0);
    if (!pobj)
    {
        PyErr_Clear();
        throw nb::type_error("Cannot convert this object to numpy array.");
    }
    nb::object rvobj = nb::steal<nb::object>(pobj);
    PyArrayObject* a = reinterpret_cast<PyArrayObject*>(pobj);
    double* rvdata = static_cast<double*>(PyArray_DATA(a));
    NumPyArray_DoublePtr rv(rvobj, rvdata);
    return rv;
}


/// extract double with a support for numpy.int types
double extractdouble(nb::object obj)
{
    double x;
    if (nb::try_cast<double>(obj, x))
        return x;

    PyObject* pobj = obj.ptr();
    if (PyArray_CheckScalar(pobj))
    {
        PyArray_Descr* descr = PyArray_DescrFromType(NPY_DOUBLE);
        if (!descr)
            throw nb::python_error();

        int ok = PyArray_CastScalarToCtype(pobj, &x, descr);
        Py_DECREF(descr);

        if (ok < 0)
            throw nb::python_error();

        return x;
    }
    // nothing worked, call default behavior which will raise an exception
    return nb::cast<double>(obj);
}


/// extract integer with a support for numpy.int types
int extractint(nb::object obj)
{
    PyObject* pobj = obj.ptr();
    if (PyArray_CheckScalar(pobj))
    {
        int rv = PyArray_PyIntAsInt(pobj);
        if (rv == -1 && PyErr_Occurred())
            throw nb::python_error();
        return rv;
    }

    PyObject* idx = PyNumber_Index(pobj);
    if (!idx)
        throw nb::python_error();

    nb::object idx_obj = nb::steal<nb::object>(idx);
    long value = PyLong_AsLong(idx_obj.ptr());
    if (value == -1 && PyErr_Occurred())
        throw nb::python_error();

    if (value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max())
    {
        PyErr_SetString(PyExc_OverflowError, "integer index out of range");
        throw nb::python_error();
    }

    return static_cast<int>(value);
}


/// extract a vector of integers from a numpy array, iterable or scalar
std::vector<int> extractintvector(nb::object obj)
{
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
            nb::object aobj = obj;
            if (NPY_INT != PyArray_TYPE(a))
            {
                PyObject* casted = PyArray_Cast(a, NPY_INT);
                if (!casted)
                    throw nb::python_error();

                aobj = nb::steal<nb::object>(casted);
            }
            PyArrayObject* a1 = reinterpret_cast<PyArrayObject*>(aobj.ptr());
            assert(NPY_INT == PyArray_TYPE(a1));
            int* pfirst = static_cast<int*>(PyArray_DATA(a1));
            int* plast = pfirst + PyArray_SIZE(a1);
            rv.assign(pfirst, plast);
            return rv;
        }
        // otherwise translate every item in the iterable
        rv.reserve(len(obj));

        for (nb::handle item : obj)
            rv.push_back(extractint(nb::borrow<nb::object>(item)));

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
    throw std::runtime_error(emsg);
}

}   // namespace srrealmodule


namespace diffpy {
namespace srreal {

/// shared converter that first tries to extract the pointer and then calls
/// diffpy.srreal.structureadapter.createStructureAdapter
StructureAdapterPtr createStructureAdapter(nb::object stru)
{
    StructureAdapterPtr adpt;
    if (nb::try_cast<StructureAdapterPtr>(stru, adpt))
        return adpt;

    nb::object mod = nb::module_::import_("diffpy.srreal.structureadapter");
    nb::object convertinpython = mod.attr("createStructureAdapter");
    return nb::cast<StructureAdapterPtr>(convertinpython(stru));
}

}   // namespace srreal
}   // namespace diffpy

// End of file
