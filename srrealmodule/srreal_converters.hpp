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

namespace diffpy {
namespace srreal_converters {

/// this macro defines a wrapper function for a C++ method,
/// that converts the result to numpy array
#define DECLARE_PYARRAY_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    ::boost::python::object wrapper(const T& obj) \
    { \
        ::boost::python::object rv = convertToNumPyArray(obj.method()); \
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


/// template function for converting iterables to numpy array of doubles
template <class Iter>
::boost::python::object
convertToNumPyArray(Iter first, Iter last)
{
    using namespace ::boost;
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


}   // namespace srreal_converters
}   // namespace diffpy

#endif  // SRREAL_CONVERTERS_HPP_INCLUDED
