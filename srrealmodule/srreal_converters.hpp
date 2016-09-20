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

#ifndef SRREAL_CONVERTERS_HPP_INCLUDED
#define SRREAL_CONVERTERS_HPP_INCLUDED

#include <boost/python/object.hpp>
#include <boost/python/import.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/override.hpp>

#include <algorithm>
#include <string>

#include <diffpy/srreal/forwardtypes.hpp>
#include <diffpy/srreal/R3linalg.hpp>
#include <diffpy/srreal/QuantityType.hpp>
#include <diffpy/version.hpp>

#if DIFFPY_VERSION < 1003002000
#error "diffpy.srreal requires libdiffpy 1.3.2 or later."
#endif

/// Conversion function that supports implicit conversions in
/// PairQuantity::eval and PairQuantity::setStructure

namespace diffpy {
namespace srreal {

StructureAdapterPtr
createStructureAdapter(::boost::python::object);

}   // namespace srreal
}   // namespace diffpy

namespace srrealmodule {

using diffpy::srreal::createStructureAdapter;

/// this macro creates a setter for overloaded method that can accept
/// either instance or a type string
#define DECLARE_BYTYPE_SETTER_WRAPPER(method, wrapper) \
    template <class T, class V> \
    void wrapper(T& obj, ::boost::python::object value) \
    { \
        using ::boost::python::extract; \
        extract<std::string> tp(value); \
        if (tp.check())  obj.method##ByType(tp()); \
        else \
        { \
            typename V::SharedPtr p = extract<typename V::SharedPtr>(value); \
            obj.method(p); \
        } \
    } \


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
/// that converts the result to numpy character array
#define DECLARE_PYCHARARRAY_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    ::boost::python::object wrapper(const T& obj) \
    { \
        ::boost::python::list lst = convertToPythonList(obj.method()); \
        ::boost::python::object tochararray = \
            ::boost::python::import("numpy").attr("char").attr("array"); \
        ::boost::python::object rv = tochararray(lst); \
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


/// this macro defines a wrapper function for a C++ method with one argument,
/// that converts the result to a python set
#define DECLARE_PYSET_METHOD_WRAPPER1(method, wrapper) \
    template <class T, class T1> \
    ::boost::python::object wrapper(const T& obj, const T1& a1) \
    { \
        ::boost::python::object rv = convertToPythonSet(obj.method(a1)); \
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


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to a python list
#define DECLARE_PYLIST_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    ::boost::python::object wrapper(const T& obj) \
    { \
        ::boost::python::object rv = convertToPythonList(obj.method()); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to a python list of NumPy arrays
#define DECLARE_PYLISTARRAY_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    ::boost::python::list wrapper(const T& obj) \
    { \
        ::boost::python::list rvlist; \
        fillPyListWithArrays(rvlist, obj.method()); \
        return rvlist; \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to a list of Python sets
#define DECLARE_PYLISTSET_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    ::boost::python::list wrapper(const T& obj) \
    { \
        ::boost::python::list rvlist; \
        fillPyListWithSets(rvlist, obj.method()); \
        return rvlist; \
    } \


/// this macro defines a wrapper function for a C++ method,
/// that converts the result to a python dict
#define DECLARE_PYDICT_METHOD_WRAPPER(method, wrapper) \
    template <class T> \
    ::boost::python::object wrapper(const T& obj) \
    { \
        ::boost::python::object rv = convertToPythonDict(obj.method()); \
        return rv; \
    } \


/// this macro defines a wrapper function for a C++ method with one argument,
/// that converts the result to a python dict
#define DECLARE_PYDICT_METHOD_WRAPPER1(method, wrapper) \
    template <class T, class T1> \
    ::boost::python::object wrapper(const T& obj, const T1& a1) \
    { \
        ::boost::python::object rv = convertToPythonDict(obj.method(a1)); \
        return rv; \
    } \


/// helper template function for DECLARE_PYLISTARRAY_METHOD_WRAPPER
template <class T>
void fillPyListWithArrays(::boost::python::list lst, const T& value)
{
    typename T::const_iterator v = value.begin();
    for (; v != value.end(); ++v)  lst.append(convertToNumPyArray(*v));
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


/// helper template function for DECLARE_PYLISTSET_METHOD_WRAPPER
template <class T>
void fillPyListWithSets(::boost::python::list lst, const T& value)
{
    typename T::const_iterator v = value.begin();
    for (; v != value.end(); ++v)  lst.append(convertToPythonSet(*v));
}


/// Type for numpy array object and a raw pointer to its double data
typedef std::pair<boost::python::object, double*> NumPyArray_DoublePtr;


/// helper for creating numpy array of doubles
NumPyArray_DoublePtr createNumPyDoubleArray(int dim, const int* sz);


/// helper for creating numpy array of the same shape as the argument
NumPyArray_DoublePtr createNumPyDoubleArrayLike(boost::python::object& obj);


/// helper for creating numpy views on existing double array
boost::python::object createNumPyDoubleView(double*, int dim, const int* sz);


/// template function for converting iterables to numpy array of doubles
template <class Iter>
::boost::python::object
convertToNumPyArray(Iter first, Iter last)
{
    int sz = last - first;
    NumPyArray_DoublePtr ap = createNumPyDoubleArray(1, &sz);
    std::copy(first, last, ap.second);
    return ap.first;
}


/// specialization for R3::Vector
inline ::boost::python::object
convertToNumPyArray(const ::diffpy::srreal::R3::Vector& value)
{
    return convertToNumPyArray(value.begin(), value.end());
}


/// specialization for R3::Matrix
inline ::boost::python::object
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
inline ::boost::python::object
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
inline ::boost::python::object
convertToNumPyArray(const ::diffpy::srreal::QuantityType& value)
{
    return convertToNumPyArray(value.begin(), value.end());
}


/// NumPy array view specializations for R3::Vector
boost::python::object
viewAsNumPyArray(::diffpy::srreal::R3::Vector&);


/// NumPy array view specializations for R3::Matrix
boost::python::object
viewAsNumPyArray(::diffpy::srreal::R3::Matrix&);


/// Copy possible NumPy array to R3::Vector
void assignR3Vector(
        ::diffpy::srreal::R3::Vector& dst, boost::python::object& value);


/// Copy possible NumPy array to R3::Matrix
void assignR3Matrix(
        ::diffpy::srreal::R3::Matrix& dst, boost::python::object& value);


/// Type for numpy array object and a raw pointer to its double data
typedef std::pair<boost::python::object, int*> NumPyArray_IntPtr;


/// helper for creating numpy array of integers
NumPyArray_IntPtr createNumPyIntArray(int dim, const int* sz);


/// specialization for a vector of integers
inline ::boost::python::object
convertToNumPyArray(const ::std::vector<int>& value)
{
    int sz = value.size();
    NumPyArray_IntPtr ap = createNumPyIntArray(1, &sz);
    std::copy(value.begin(), value.end(), ap.second);
    return ap.first;
}


/// template function for converting C++ STL container to a python list
template <class T>
::boost::python::list
convertToPythonList(const T& value)
{
    using namespace ::boost;
    python::list rvlist;
    typename T::const_iterator ii;
    for (ii = value.begin(); ii != value.end(); ++ii)  rvlist.append(*ii);
    return rvlist;
}


/// template converter of a C++ map-like container to a python dictionary
template <class T>
::boost::python::dict
convertToPythonDict(const T& value)
{
    ::boost::python::dict rv;
    typename T::const_iterator ii = value.begin();
    for (; ii != value.end(); ++ii)  rv[ii->first] = ii->second;
    return rv;
}


/// efficient conversion of Python object to a QuantityType
/// If obj wraps a QuantityType reference, return that reference.
/// Otherwise copy the obj values to rv and return rv.
::diffpy::srreal::QuantityType&
extractQuantityType(::boost::python::object obj,
        ::diffpy::srreal::QuantityType& rv);


/// efficient conversion of Python object to a numpy array of doubles
NumPyArray_DoublePtr extractNumPyDoubleArray(::boost::python::object& obj);


/// extract integer with a support for numpy.int types
int extractint(::boost::python::object obj);


/// extract a vector of integers from a numpy array, iterable or scalar
std::vector<int> extractintvector(::boost::python::object obj);


/// helper for raising RuntimeError on a call of pure virtual function
void throwPureVirtualCalled(const char* fncname);


/// template class for getting overrides to pure virtual method
template <class T>
class wrapper_srreal : public ::boost::python::wrapper<T>
{
    public:

        typedef T base;

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

// Include shared wrapper definitions ----------------------------------------

#include "srreal_converters.ipp"

#endif  // SRREAL_CONVERTERS_HPP_INCLUDED
