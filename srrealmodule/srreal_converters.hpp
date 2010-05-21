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

#ifndef SRREAL_CONVERTERS_HPP_INCLUDED
#define SRREAL_CONVERTERS_HPP_INCLUDED

#include <algorithm>

#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include <diffpy/srreal/R3linalg.hpp>
#include <diffpy/srreal/PairQuantity.hpp>

namespace srrealmodule {

/// this macro defines a wrapper function for a C++ method,
/// that converts the result to numpy array
#define DECLARE_PYARRAY_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    ::boost::python::object wrapper(const T& obj) \
    { \
        ::boost::python::object rv = convertToNumPyArray(obj.method()); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method with one argument,
/// that converts the result to numpy array
#define DECLARE_PYARRAY_METHOD_WRAPPER1(method, wrapper) \
    template <class T, class T1> \
    ::boost::python::object wrapper(const T& obj, const T1& a1) \
    { \
        ::boost::python::object rv = convertToNumPyArray(obj.method(a1)); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to a python set
#define DECLARE_PYSET_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    ::boost::python::object wrapper(const T& obj) \
    { \
        ::boost::python::object rv = convertToPythonSet(obj.method()); \
        return rv; \
    } \


/// this macro defines a wrapper for C++ function without arguments
/// that converts the result to a python set
#define DECLARE_PYSET_FUNCTION_WRAPPER(fnc, wrapper) \
    ::boost::python::object wrapper() \
    { \
        ::boost::python::object rv = convertToPythonSet(fnc()); \
        return rv; \
    } \


/// flag for imported numpy array
static bool did_import_array;

/// helper function for importing numpy array - once and only once.
inline
void initialize_numpy()
{
    if (did_import_array)  return;
    import_array();
    did_import_array = true;
}


/// template function for converting iterables to numpy array of doubles
template <class Iter>
::boost::python::object
convertToNumPyArray(Iter first, Iter last)
{
    using namespace ::boost;
    initialize_numpy();
    npy_intp sz = last - first;
    python::object rv(
            python::handle<>(PyArray_SimpleNew(1, &sz, PyArray_DOUBLE)));
    double* rvdata = (double*) PyArray_DATA((PyArrayObject*) rv.ptr());
    ::std::copy(first, last, rvdata);
    return rv;
}


/// specialization for R3::Vector
inline ::boost::python::object
convertToNumPyArray(const ::diffpy::srreal::R3::Vector& value)
{
    return convertToNumPyArray(value.data(), value.data() + value.length());
}


/// specialization for R3::Matrix
inline ::boost::python::object
convertToNumPyArray(const ::diffpy::srreal::R3::Matrix& mx)
{
    using namespace ::boost;
    using namespace ::diffpy::srreal;
    initialize_numpy();
    npy_intp sz[2] = {R3::Ndim, R3::Ndim};
    python::object rv(
            python::handle<>(PyArray_SimpleNew(2, sz, PyArray_DOUBLE)));
    double* rvdata = (double*) PyArray_DATA((PyArrayObject*) rv.ptr());
    ::std::copy(mx.data(), mx.data() + sz[0] * sz[1], rvdata);
    return rv;
}


/// specialization for QuantityType
inline ::boost::python::object
convertToNumPyArray(const ::diffpy::srreal::QuantityType& value)
{
    return convertToNumPyArray(value.begin(), value.end());
}


/// template function for converting C++ STL container to a python set
template <class T>
::boost::python::object
convertToPythonSet(const T& value)
{
    using namespace ::boost;
    python::object rvset(python::handle<>(PySet_New(NULL)));
    python::object rvset_add = rvset.attr("add");
    typename T::const_iterator ii;
    for (ii = value.begin(); ii != value.end(); ++ii)  rvset_add(*ii);
    return rvset;
}

}   // namespace srrealmodule

#endif  // SRREAL_CONVERTERS_HPP_INCLUDED
