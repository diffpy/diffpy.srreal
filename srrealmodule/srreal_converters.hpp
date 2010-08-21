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
#include <diffpy/srreal/R3linalg.hpp>
#include <diffpy/srreal/PairQuantity.hpp>

// This macro is required for extension modules that are in several files.
// It must be defined before includsion of numpy/arrayobject.h
#define PY_ARRAY_UNIQUE_SYMBOL DIFFPY_SRREAL_NUMPY_ARRAY_SYMBOL

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


/// Type for numpy array object and a raw pointer to its double data
typedef std::pair<boost::python::object, double*> NumPyArray_DoublePtr;

/// helper for creating numpy array of doubles
NumPyArray_DoublePtr createNumPyDoubleArray(int dim, const int* sz);


/// template function for converting iterables to numpy array of doubles
template <class Iter>
::boost::python::object
convertToNumPyArray(Iter first, Iter last)
{
    int sz = last - first;
    NumPyArray_DoublePtr ap = createNumPyDoubleArray(1, &sz);
    ::std::copy(first, last, ap.second);
    return ap.first;
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
    using namespace diffpy::srreal;
    int sz[2] = {R3::Ndim, R3::Ndim};
    NumPyArray_DoublePtr ap = createNumPyDoubleArray(2, sz);
    ::std::copy(mx.data(), mx.data() + sz[0] * sz[1], ap.second);
    return ap.first;
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


/// helper for raising RuntimeError on a call of pure virtual function
void throwPureVirtualCalled(const char* fncname);


/// template class for getting overrides to pure virtual method
template <class T>
class wrapper_srreal : public ::boost::python::wrapper<T>
{
    protected:
        ::boost::python::override
        get_pure_virtual_override(const char* name) const
        {
            ::boost::python::override f = this->get_override(name);
            if (!f)  throwPureVirtualCalled(name);
            return f;
        }
};


}   // namespace srrealmodule

#endif  // SRREAL_CONVERTERS_HPP_INCLUDED
