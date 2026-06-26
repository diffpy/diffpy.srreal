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

// TODO: Go through type casters, some of which are found redundant or could be
// replaced with nanobind builtins.

#ifndef SRREAL_CONVERTERS_HPP_INCLUDED
#define SRREAL_CONVERTERS_HPP_INCLUDED

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/shared_ptr.h>

#include <algorithm>
#include <string>

#include <diffpy/srreal/forwardtypes.hpp>
#include <diffpy/srreal/R3linalg.hpp>
#include <diffpy/srreal/QuantityType.hpp>
#include <diffpy/version.hpp>

#if DIFFPY_VERSION < 1004000000
#error diffpy.srreal requires libdiffpy 1.4.0 or later.
#endif

namespace nb = nanobind;

/// Conversion function that supports implicit conversions in
/// PairQuantity::eval and PairQuantity::setStructure

namespace diffpy {
namespace srreal {

StructureAdapterPtr
createStructureAdapter(nb::object);

}   // namespace srreal
}   // namespace diffpy

namespace srrealmodule {

using diffpy::srreal::createStructureAdapter;

/// this macro creates a setter for overloaded method that can accept
/// either instance or a type string
#define DECLARE_BYTYPE_SETTER_WRAPPER(method, wrapper) \
    template <class T, class V> \
    void wrapper(T& obj, nb::object value) \
    { \
        std::string tp;                                       \
        if (nb::try_cast<std::string>(value, tp))             \
            obj.method##ByType(tp);                           \
        else                                                  \
            obj.method(nb::cast<typename V::SharedPtr>(value)); \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to numpy array
#define DECLARE_PYARRAY_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    nb::object wrapper(const T& obj) \
    { \
        nb::object rv = convertToNumPyArray(obj.method()); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method with one argument,
/// that converts the result to numpy array
#define DECLARE_PYARRAY_METHOD_WRAPPER1(method, wrapper) \
    template <class T, class T1> \
    nb::object wrapper(const T& obj, const T1& a1) \
    { \
        nb::object rv = convertToNumPyArray(obj.method(a1)); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to numpy character array
#define DECLARE_PYCHARARRAY_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    nb::object wrapper(const T& obj) \
    { \
        nb::list lst = convertToPythonList(obj.method()); \
        nb::object tochararray = \
            nb::module_::import_("numpy").attr("char").attr("array"); \
        nb::object rv = tochararray(lst); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to a python set
#define DECLARE_PYSET_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    nb::object wrapper(const T& obj) \
    { \
        nb::object rv = convertToPythonSet(obj.method()); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method with one argument,
/// that converts the result to a python set
#define DECLARE_PYSET_METHOD_WRAPPER1(method, wrapper) \
    template <class T, class T1> \
    nb::object wrapper(const T& obj, const T1& a1) \
    { \
        nb::object rv = convertToPythonSet(obj.method(a1)); \
        return rv; \
    } \


/// this macro defines a wrapper for C++ function without arguments
/// that converts the result to a python set
#define DECLARE_PYSET_FUNCTION_WRAPPER(fnc, wrapper) \
    nb::object wrapper() \
    { \
        nb::object rv = convertToPythonSet(fnc()); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to a python list
#define DECLARE_PYLIST_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    nb::object wrapper(const T& obj) \
    { \
        nb::object rv = convertToPythonList(obj.method()); \
        return rv; \
    } \


/// this macro defines wrapper for a C++ method with one argument,
/// that convert the result to python list
#define DECLARE_PYLIST_METHOD_WRAPPER1(method, wrapper) \
    template <class T, class T1> \
    nb::object wrapper(const T& obj, const T1& a1) \
    { \
        nb::object rv = convertToPythonList(obj.method(a1)); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to a python list of NumPy arrays
#define DECLARE_PYLISTARRAY_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    nb::list wrapper(const T& obj) \
    { \
        nb::list rvlist; \
        fillPyListWithArrays(rvlist, obj.method()); \
        return rvlist; \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to a list of Python sets
#define DECLARE_PYLISTSET_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    nb::list wrapper(const T& obj) \
    { \
        nb::list rvlist; \
        fillPyListWithSets(rvlist, obj.method()); \
        return rvlist; \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to a python dict
#define DECLARE_PYDICT_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    nb::object wrapper(const T& obj) \
    { \
        nb::object rv = convertToPythonDict(obj.method()); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method with one argument,
/// that converts the result to a python dict
#define DECLARE_PYDICT_METHOD_WRAPPER1(method, wrapper) \
    template <class T, class T1> \
    nb::object wrapper(const T& obj, const T1& a1) \
    { \
        nb::object rv = convertToPythonDict(obj.method(a1)); \
        return rv; \
    } \


/// helper template function for DECLARE_PYLISTARRAY_METHOD_WRAPPER
template <class T>
void fillPyListWithArrays(nb::list lst, const T& value)
{
    typename T::const_iterator v = value.begin();
    for (; v != value.end(); ++v)  lst.append(convertToNumPyArray(*v));
}


/// template function for converting C++ STL container to a python set
template <class T>
nb::object
convertToPythonSet(const T& value)
{
    nb::set rv;
    for (auto const& item : value)
        rv.add(item);
    return rv;
}


/// helper template function for DECLARE_PYLISTSET_METHOD_WRAPPER
template <class T>
void fillPyListWithSets(nb::list lst, const T& value)
{
    typename T::const_iterator v = value.begin();
    for (; v != value.end(); ++v)  lst.append(convertToPythonSet(*v));
}


/// Type for numpy array object and a raw pointer to its double data
typedef std::pair<nb::object, double*> NumPyArray_DoublePtr;


/// helper for creating numpy array of doubles
NumPyArray_DoublePtr createNumPyDoubleArray(int dim, const int* sz);


/// helper for creating numpy array of the same shape as the argument
NumPyArray_DoublePtr createNumPyDoubleArrayLike(nb::object& obj);


/// helper for creating numpy views on existing double array
nb::object createNumPyDoubleView(double*, int dim, const int* sz);


/// template function for converting iterables to numpy array of doubles
template <class Iter>
nb::object
convertToNumPyArray(Iter first, Iter last)
{
    int sz = last - first;
    NumPyArray_DoublePtr ap = createNumPyDoubleArray(1, &sz);
    std::copy(first, last, ap.second);
    return ap.first;
}


/// specialization for R3::Vector
inline nb::object
convertToNumPyArray(const ::diffpy::srreal::R3::Vector& value)
{
    return convertToNumPyArray(value.begin(), value.end());
}


/// specialization for R3::Matrix
inline nb::object
convertToNumPyArray(const ::diffpy::srreal::R3::Matrix& mx)
{
    using namespace diffpy::srreal;
    int sz[2] = {R3::Ndim, R3::Ndim};
    NumPyArray_DoublePtr ap = createNumPyDoubleArray(2, sz);
    double* xo = ap.second;
    *(xo++) = mx(0, 0); *(xo++) = mx(0, 1); *(xo++) = mx(0, 2);
    *(xo++) = mx(1, 0); *(xo++) = mx(1, 1); *(xo++) = mx(1, 2);
    *(xo++) = mx(2, 0); *(xo++) = mx(2, 1); *(xo++) = mx(2, 2);
    return ap.first;
}


/// specialization for std::vector<R3::Vector>
inline nb::object
convertToNumPyArray(const ::std::vector<diffpy::srreal::R3::Vector>& vr3v)
{
    using namespace diffpy::srreal;
    int n = vr3v.size();
    int sz[2] = {n, R3::Ndim};
    NumPyArray_DoublePtr ap = createNumPyDoubleArray(2, sz);
    double* p = ap.second;
    std::vector<R3::Vector>::const_iterator v = vr3v.begin();
    for (; v != vr3v.end(); ++v)
    {
        const double* pv = &((*v)[0]);
        const double* pvlast = pv + R3::Ndim;
        for (; pv != pvlast; ++p, ++pv)  *p = *pv;
    }
    assert(p == ap.second + sz[0] * sz[1]);
    return ap.first;
}


/// specialization for QuantityType
inline nb::object
convertToNumPyArray(const ::diffpy::srreal::QuantityType& value)
{
    return convertToNumPyArray(value.begin(), value.end());
}


/// NumPy array view specializations for R3::Vector
nb::object
viewAsNumPyArray(::diffpy::srreal::R3::Vector&);


/// NumPy array view specializations for R3::Matrix
nb::object
viewAsNumPyArray(::diffpy::srreal::R3::Matrix&);


/// Copy possible NumPy array to R3::Vector
void assignR3Vector(
        ::diffpy::srreal::R3::Vector& dst, nb::object& value);


/// Copy possible NumPy array to R3::Matrix
void assignR3Matrix(
        ::diffpy::srreal::R3::Matrix& dst, nb::object& value);


/// Type for numpy array object and a raw pointer to its double data
typedef std::pair<nb::object, int*> NumPyArray_IntPtr;


/// helper for creating numpy array of integers
NumPyArray_IntPtr createNumPyIntArray(int dim, const int* sz);


/// specialization for a vector of integers
inline nb::object
convertToNumPyArray(const ::std::vector<int>& value)
{
    int sz = value.size();
    NumPyArray_IntPtr ap = createNumPyIntArray(1, &sz);
    std::copy(value.begin(), value.end(), ap.second);
    return ap.first;
}


/// template function for converting C++ STL container to a python list
template <class T>
nb::list
convertToPythonList(const T& value)
{
    nb::list rv;
    for (auto const& item : value)
        rv.append(item);
    return rv;
}


/// template converter of a C++ map-like container to a python dictionary
template <class T>
nb::dict
convertToPythonDict(const T& value)
{
    nb::dict rv;
    for (auto const& item : value)
        rv[nb::cast(item.first)] = nb::cast(item.second);
    return rv;
}


/// efficient conversion of Python object to a QuantityType
/// If obj wraps a QuantityType reference, return that reference.
/// Otherwise copy the obj values to rv and return rv.
::diffpy::srreal::QuantityType&
extractQuantityType(nb::object obj,
        ::diffpy::srreal::QuantityType& rv);


/// efficient conversion of Python object to a numpy array of doubles
NumPyArray_DoublePtr extractNumPyDoubleArray(nb::object& obj);


/// extract double with a support for numpy numeric types
double extractdouble(nb::object obj);


/// extract integer with a support for numpy.int types
int extractint(nb::object obj);


/// extract a vector of integers from a numpy array, iterable or scalar
std::vector<int> extractintvector(nb::object obj);


/// helper for raising RuntimeError on a call of pure virtual function
void throwPureVirtualCalled(const char* fncname);

}   // namespace srrealmodule

// Include shared wrapper definitions ----------------------------------------

#include "srreal_converters.ipp"

#endif  // SRREAL_CONVERTERS_HPP_INCLUDED
