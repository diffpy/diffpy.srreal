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
*
*****************************************************************************/

#include <string>
#include <stdexcept>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <ranges>
#include <type_traits>
#include <utility>

#include <diffpy/Attributes.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>

#include "srreal_converters.hpp"
#include "srreal_validators.hpp"

namespace {

// exception translations ----------------------------------------------------

using diffpy::attributes::DoubleAttributeError;
using std::invalid_argument;

template <class Array>
bool native_array(nb::handle obj, Array& result)
{
    // Keep scalars on their conversion path. NumPy datetime/timedelta scalar
    // buffers omit strides, which nanobind's buffer importer dereferences.
    if (!PySequence_Check(obj.ptr())) return false;
    // The importer checks byte order, dtype and layout, but not alignment.
    return nb::try_cast<Array>(obj, result, false) &&
        reinterpret_cast<std::uintptr_t>(result.data()) %
            alignof(typename Array::Scalar) == 0;
}

using IndexArray = nb::ndarray<nb::numpy, nb::ro, nb::ndim<1>,
        nb::device::cpu>;

template <class T>
bool copy_index_array(const IndexArray& array, std::vector<int>& result)
{
    if (array.dtype() != nb::dtype<T>() ||
            reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0)
        return false;

    result.resize(array.size());
    if constexpr (std::is_same_v<T, int>)
    {
        if (array.stride(0) == 1)
        {
            if (!result.empty())
                std::copy_n(static_cast<const int*>(array.data()),
                        result.size(), result.begin());
            return true;
        }
    }
    // Typed views also handle slices without boxing each integer.
    auto values = array.view<const T>();
    for (size_t i = 0; i < result.size(); ++i)
    {
        T value = values(i);
        if (!std::in_range<int>(value))
        {
            PyErr_SetString(PyExc_OverflowError, "integer index out of range");
            nb::raise_python_error();
        }
        result[i] = static_cast<int>(value);
    }
    return true;
}

nb::object newNumPyArray(size_t dim, const size_t* shape,
        const char* dtype, size_t itemsize)
{
    // NumPy dimensions and byte counts must fit in Py_ssize_t, even though
    // ndarray metadata uses size_t. Check before multiplying or allocating.
    const size_t limit = static_cast<size_t>(PY_SSIZE_T_MAX);
    size_t count = 1;
    nb::tuple_builder dimensions(dim);
    for (size_t i = 0; i < dim; ++i)
    {
        if (shape[i] > limit || (count && shape[i] > limit / itemsize / count))
        {
            PyErr_SetString(PyExc_OverflowError, "array size is out of range");
            nb::raise_python_error();
        }
        count *= shape[i];
        dimensions.put(shape[i]);
    }
    return nb::module_::import_("numpy").attr("empty")(
            dimensions.commit(), nb::arg("dtype") = dtype);
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
            catch (const nb::cast_error&)
            {
                // Explicit nb::cast calls bypass nanobind's argument checking.
                // Match the TypeError raised by Boost.Python extraction.
                PyErr_SetString(PyExc_TypeError,
                        "Cannot convert this Python object to the requested C++ type.");
            }
        }
    );
}


NumPyOutput<double> createNumPyDoubleArray(size_t dim, const size_t* sz)
{
    nb::object array = newNumPyArray(dim, sz, "float64", sizeof(double));
    return {array, nb::cast<decltype(NumPyOutput<double>::array)>(array, false)};
}


NumPyOutput<double> createNumPyDoubleArrayLike(const NumPyDoubleInput& array)
{
    std::vector<size_t> shape(array.ndim());
    for (size_t i = 0; i < shape.size(); ++i) shape[i] = array.shape(i);
    return createNumPyDoubleArray(shape.size(), shape.data());
}


NumPyOutput<int> createNumPyIntArray(size_t dim, const size_t* sz)
{
    nb::object array = newNumPyArray(dim, sz, "intc", sizeof(int));
    return {array, nb::cast<decltype(NumPyOutput<int>::array)>(array, false)};
}


nb::object viewAsNumPyArray(::diffpy::srreal::R3::Vector& v, nb::handle owner)
{
    using View = nb::ndarray<nb::numpy, double, nb::shape<3>,
            nb::c_contig, nb::device::cpu>;
    return View(&v[0], {3}, owner).cast(nb::rv_policy::reference);
}


nb::object viewAsNumPyArray(::diffpy::srreal::R3::Matrix& mx, nb::handle owner)
{
    using View = nb::ndarray<nb::numpy, double, nb::shape<3, 3>,
            nb::c_contig, nb::device::cpu>;
    return View(&mx(0, 0), {3, 3}, owner).cast(nb::rv_policy::reference);
}


void assignR3Vector(::diffpy::srreal::R3::Vector& dst, nb::object& value)
{
    using VectorInput = nb::ndarray<nb::numpy, const double, nb::shape<3>,
            nb::c_contig, nb::device::cpu>;
    VectorInput source;
    if (native_array(value, source))
    {
        std::copy_n(source.data(), 3, dst.data().begin());
        return;
    }
    double scalar;
    if (nb::try_cast<double>(value, scalar))
    {
        std::ranges::fill(dst.data().begin(), dst.data().end(), scalar);
        return;
    }
    // Assign into temporary NumPy storage to preserve broadcasting, dtype
    // conversion and warnings without partially updating the atom on failure.
    const size_t shape[] = {3};
    auto temporary = createNumPyDoubleArray(1, shape);
    temporary.object[nb::slice(nb::none(), nb::none(), nb::none())] = value;
    source = nb::cast<VectorInput>(temporary.object, false);
    std::copy_n(source.data(), 3, dst.data().begin());
}


void assignR3Matrix(::diffpy::srreal::R3::Matrix& dst, nb::object& value)
{
    using MatrixInput = nb::ndarray<nb::numpy, const double, nb::shape<3, 3>,
            nb::c_contig, nb::device::cpu>;
    MatrixInput source;
    if (native_array(value, source))
    {
        std::copy_n(source.data(), 9, dst.data().begin());
        return;
    }
    double scalar;
    if (nb::try_cast<double>(value, scalar))
    {
        std::ranges::fill(dst.data().begin(), dst.data().end(), scalar);
        return;
    }
    const size_t shape[] = {3, 3};
    auto temporary = createNumPyDoubleArray(2, shape);
    temporary.object[nb::slice(nb::none(), nb::none(), nb::none())] = value;
    source = nb::cast<MatrixInput>(temporary.object, false);
    std::copy_n(source.data(), 9, dst.data().begin());
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
    // Normalize layout before typed access, including byte order and alignment.
    // Other dtypes retain the elementwise conversion used by this endpoint.
    using QuantityInput = nb::ndarray<nb::numpy, const double, nb::ndim<1>,
            nb::c_contig, nb::device::cpu>;
    QuantityInput direct;
    if (native_array(obj, direct))
    {
        rv.resize(direct.size());
        if (direct.size()) std::copy_n(direct.data(), direct.size(), rv.begin());
        return rv;
    }
    nb::module_ numpy = nb::module_::import_("numpy");
    if (nb::isinstance(obj, numpy.attr("ndarray")) &&
            nb::cast<int>(obj.attr("ndim")) == 1 &&
            nb::cast<std::string>(obj.attr("dtype").attr("kind")) == "f" &&
            nb::cast<size_t>(obj.attr("itemsize")) == sizeof(double))
    {
        nb::object normalized = numpy.attr("require")(obj,
                nb::arg("dtype") = "=f8",
                nb::arg("requirements") = nb::make_tuple("C", "A"));
        auto array = nb::cast<NumPyDoubleInput>(normalized, false);
        rv.resize(array.size());
        if (array.size()) std::copy_n(array.data(), array.size(), rv.begin());
        return rv;
    }
    // otherwise copy elementwise converting each element to a double
    std::vector<double> tmp;
    for (nb::handle item : obj)
        tmp.push_back(nb::cast<double>(item));

    rv.assign(tmp.begin(), tmp.end());
    return rv;
}


/// efficient conversion of Python object to a numpy array of doubles
NumPyDoubleInput extractNumPyDoubleArray(nb::object obj)
{
    NumPyDoubleInput direct;
    if (native_array(obj, direct)) return direct;
    nb::module_ numpy = nb::module_::import_("numpy");
    try
    {
        // NumPy's old FromAny path used safe casting for existing arrays,
        // but constructed Python sequences directly in the requested dtype.
        nb::object array;
        if (nb::isinstance(obj, numpy.attr("generic")))
        {
            // NumPy scalars historically use scalar casting, which permits
            // e.g. longdouble and complex-to-real with a ComplexWarning.
            array = numpy.attr("asarray")(obj).attr("astype")("float64");
        }
        else if (nb::isinstance<nb::bytes>(obj))
        {
            array = numpy.attr("asarray")(obj, nb::arg("dtype") = "float64");
        }
        else if (nb::isinstance(obj, numpy.attr("ndarray")) ||
                nb::hasattr(obj, "__array__") || PyObject_CheckBuffer(obj.ptr()))
        {
            array = numpy.attr("asarray")(obj).attr("astype")("float64",
                    nb::arg("casting") = "safe", nb::arg("copy") = false);
        }
        else
        {
            array = numpy.attr("asarray")(obj, nb::arg("dtype") = "float64");
        }
        array = numpy.attr("require")(array,
                nb::arg("dtype") = "=f8",
                nb::arg("requirements") = nb::make_tuple("C", "A"));
        return nb::cast<NumPyDoubleInput>(array, false);
    }
    catch (const nb::python_error&)
    {
        throw nb::type_error("Cannot convert this object to numpy array.");
    }
}


/// extract double with a support for numpy.int types
double extractdouble(nb::object obj)
{
    double x;
    if (nb::try_cast<double>(obj, x))
        return x;

    nb::module_ numpy = nb::module_::import_("numpy");
    if (nb::isinstance(obj, numpy.attr("generic")) ||
            (nb::isinstance(obj, numpy.attr("ndarray")) &&
             nb::cast<int>(obj.attr("ndim")) == 0))
    {
        // Preserve NumPy scalar casting (including its ComplexWarning),
        // without accepting Python strings as numeric arguments.
        nb::object scalar = numpy.attr("asarray")(obj)
                .attr("astype")("float64").attr("item")();
        return nb::cast<double>(scalar);
    }
    // nothing worked, call default behavior which will raise an exception
    return nb::cast<double>(obj);
}


/// extract integer with a support for numpy.int types
int extractint(nb::object obj)
{
    PyObject* pobj = obj.ptr();
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
    IndexArray array;
    // Import once, then dispatch on the native dtype without boxing elements.
    // As in native_array(), leave scalar buffers to the scalar converter.
    if (PySequence_Check(obj.ptr()) &&
            nb::try_cast<IndexArray>(obj, array, false) &&
            (copy_index_array<std::int32_t>(array, rv) ||
             copy_index_array<std::int64_t>(array, rv) ||
             copy_index_array<std::uint32_t>(array, rv) ||
             copy_index_array<std::uint64_t>(array, rv) ||
             copy_index_array<std::int16_t>(array, rv) ||
             copy_index_array<std::uint16_t>(array, rv) ||
             copy_index_array<std::int8_t>(array, rv) ||
             copy_index_array<std::uint8_t>(array, rv)))
        return rv;
    // iterable of integers
    if (isiterable(obj))
    {
        // Never narrow an array before checking each integer. Iterating also
        // handles non-native byte order and unaligned data.
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
