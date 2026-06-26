/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* Pickling support that uses serialization of the libdiffpy classes.
*
*****************************************************************************/

#ifndef SRREAL_PICKLING_HPP_INCLUDED
#define SRREAL_PICKLING_HPP_INCLUDED

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <string>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <memory>

#include <diffpy/serialization.hpp>
#include <diffpy/srreal/forwardtypes.hpp>

namespace nb = nanobind;

namespace srrealmodule {

struct PythonTrampolineTag
{
    virtual ~PythonTrampolineTag() = default;
};


inline
void ensure_tuple_length(nb::tuple state, const int statelen)
{
    if (state.size() == static_cast<size_t>(statelen))  return;
    PyErr_Format(
        PyExc_ValueError,
        "expected %d-item tuple in call to __setstate__; got %zd",
        statelen,
        state.size()
    );
    nb::raise_python_error();
}


template <typename T>
nb::bytes serialization_tobytes(const T& tobj)
{
    std::string s = diffpy::serialization_tostring(tobj);
    return nb::bytes(s.data(), s.size());
}

inline
std::string bytes_to_string(nb::handle h)
{
    if (!PyBytes_Check(h.ptr()))
    {
        PyErr_SetString(PyExc_TypeError, "expected bytes");
        nb::raise_python_error();
    }
    char *buffer = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(h.ptr(), &buffer, &size) < 0)
        nb::raise_python_error();
    return std::string(
        static_cast<const char *>(buffer),
        static_cast<size_t>(size)
    );
}


inline
nb::object runtime_type(nb::handle obj)
{
    return nb::borrow<nb::object>(reinterpret_cast<PyObject *>(Py_TYPE(obj.ptr())));
}


inline
nb::object get_instance_dict(nb::handle obj)
{
    PyObject *dict = PyObject_GetAttrString(obj.ptr(), "__dict__");
    if (!dict)
    {
        if (PyErr_ExceptionMatches(PyExc_AttributeError))
        {
            PyErr_Clear();
            return nb::none();
        }
        nb::raise_python_error();
    }
    return nb::steal<nb::object>(dict);
}


inline
bool state_manages_dict(nb::handle obj, bool default_policy)
{
    PyObject *flag =
        PyObject_GetAttrString(reinterpret_cast<PyObject *>(Py_TYPE(obj.ptr())),
                               "__getstate_manages_dict__");
    if (!flag)
    {
        if (PyErr_ExceptionMatches(PyExc_AttributeError))
        {
            PyErr_Clear();
            return default_policy;
        }
        nb::raise_python_error();
    }
    nb::object flag_obj = nb::steal<nb::object>(flag);
    if (flag_obj.is_none())  return false;
    int rv = PyObject_IsTrue(flag_obj.ptr());
    if (rv < 0)  nb::raise_python_error();
    return rv != 0;
}


inline
void ensure_dict_is_managed_or_empty(nb::handle obj, bool manages_dict)
{
    if (manages_dict)  return;
    nb::object dict_obj = get_instance_dict(obj);
    if (dict_obj.is_none())  return;
    Py_ssize_t n = PyMapping_Size(dict_obj.ptr());
    if (n < 0)  nb::raise_python_error();
    if (n == 0)  return;
    throw std::runtime_error(
        "Incomplete pickle support (__getstate_manages_dict__ not set)"
    );
}


inline
void ensure_mapping_state(nb::handle state)
{
    if (!PyMapping_Check(state.ptr()))
    {
        PyErr_SetString(PyExc_TypeError, "pickle __dict__ state must be a mapping");
        nb::raise_python_error();
    }

    PyObject *keys = PyMapping_Keys(state.ptr());
    if (!keys)
    {
        if (PyErr_ExceptionMatches(PyExc_AttributeError))
        {
            PyErr_Clear();
            PyErr_SetString(
                PyExc_TypeError,
                "pickle __dict__ state must be a mapping"
            );
        }
        nb::raise_python_error();
    }
    Py_DECREF(keys);
}


inline
void restore_instance_dict(nb::handle obj, nb::handle dict_state)
{
    if (dict_state.is_none())  return;
    ensure_mapping_state(dict_state);

    nb::object dict_obj = get_instance_dict(obj);
    if (dict_obj.is_none())
    {
        Py_ssize_t n = PyMapping_Size(dict_state.ptr());
        if (n < 0)  nb::raise_python_error();
        if (n == 0)  return;
        PyErr_SetString(PyExc_TypeError, "object has no __dict__ to restore");
        nb::raise_python_error();
    }
    if (!PyDict_Check(dict_obj.ptr()))
    {
        PyErr_SetString(PyExc_TypeError, "object __dict__ is not a dict");
        nb::raise_python_error();
    }
    if (PyDict_Update(dict_obj.ptr(), dict_state.ptr()) < 0)
        nb::raise_python_error();
}


template <class T>
bool is_python_trampoline(T *ptr)
{
    return dynamic_cast<PythonTrampolineTag *>(ptr) != nullptr;
}


template <class T>
bool is_python_trampoline(nb::handle obj)
{
    T *ptr = nullptr;
    return nb::try_cast<T *>(obj, ptr, false) &&
        ptr && is_python_trampoline(ptr);
}


template <class T>
void assign_pointer_state(std::shared_ptr<T>& slot, nb::handle value)
{
    nb::object value_obj = nb::borrow<nb::object>(value);
    if (value_obj.is_none())  return;
    slot = nb::cast<std::shared_ptr<T>>(value_obj);
}


template <class T>
nb::object resolve_state_pointer(const std::shared_ptr<T>& value)
{
    // Return None for native extension objects. Python-defined wrappers are
    // pickled independently so their dictionaries and virtual overrides survive.
    if (!value || !std::dynamic_pointer_cast<PythonTrampolineTag>(value))
        return nb::none();
    return nb::cast(value);
}


template <class T>
class ScopedSharedPtrReplacement
{
    public:

        ScopedSharedPtrReplacement(
                std::shared_ptr<T>& slot,
                std::shared_ptr<T> replacement) :
            mslot(slot),
            moriginal(slot),
            mstate(resolve_state_pointer<T>(moriginal)),
            mreplacement(std::move(replacement))
        {
            if (mstate.is_none())  return;
            mslot = mreplacement;
            mactive = true;
        }

        ~ScopedSharedPtrReplacement()
        {
            restore();
        }

        nb::object state_object() const
        {
            return mstate;
        }

        void restore() noexcept
        {
            if (!mactive)  return;
            mslot = moriginal;
            mactive = false;
        }

    private:

        std::shared_ptr<T>& mslot;
        std::shared_ptr<T> moriginal;
        nb::object mstate;
        std::shared_ptr<T> mreplacement;
        bool mactive = false;
};


enum PickleDictPolicy
{
    DICT_GUARD,
    DICT_PICKLE,
    DICT_DISCARD
};

template <class T, PickleDictPolicy dictpolicy=DICT_GUARD, class Storage=T>
class SerializationPickleSuite
{
    public:

        template <typename C>
        static void bind(C& cls)
        {
            cls
                .def("__getstate__", getstate)
                .def("__setstate__", setstate)
                .def("__reduce__", reduce)
                ;
            if constexpr (dictpolicy == DICT_PICKLE)
            {
                cls.attr("__getstate_manages_dict__") = nb::bool_(true);
            }
            else
            {
                cls.attr("__getstate_manages_dict__") = nb::none();
            }
        }

        static nb::tuple getstate(nb::object obj)
        {
            const T& tobj = nb::cast<const T&>(obj);
            nb::bytes content = serialization_tobytes(tobj);

            if constexpr (dictpolicy == DICT_PICKLE)
            {
                return nb::make_tuple(
                    content,
                    get_instance_dict(obj)
                );
            }
            if constexpr (dictpolicy == DICT_GUARD)
            {
                ensure_dict_is_managed_or_empty(obj, false);
            }
            return nb::make_tuple(content);
        }


        static void setstate(
                nb::object obj, nb::tuple state)
        {
            ensure_tuple_length(
                state,
                dictpolicy == DICT_PICKLE ? 2 : 1);
            // load the C++ object
            T& tobj = nb::cast<T&>(obj);
            diffpy::serialization_fromstring(tobj, bytes_to_string(state[0]));
            // restore the object's __dict__
            if constexpr (dictpolicy == DICT_PICKLE)
            {
                restore_instance_dict(obj, state[1]);
            }
        }

        static nb::tuple reduce(nb::object obj)
        {
            return nb::make_tuple(
                runtime_type(obj),
                nb::make_tuple(),
                obj.attr("__getstate__")()
            );
        }

        static bool getstate_manages_dict()
        {
            return dictpolicy == DICT_PICKLE;
        }

};  // class SerializationPickleSuite


template <class T, PickleDictPolicy dictpolicy=DICT_GUARD, class Storage=T>
class PairQuantityPickleSuite :
    public SerializationPickleSuite<T, dictpolicy, Storage>
{
    private:

        typedef SerializationPickleSuite<T, dictpolicy, Storage> Super;

    public:

        template <typename C>
        static void bind(C& cls)
        {
            cls
                .def("__getstate__", getstate)
                .def("__setstate__", setstate)
                .def("__reduce__", reduce)
                ;
            if constexpr (dictpolicy == DICT_PICKLE)
            {
                cls.attr("__getstate_manages_dict__") = nb::bool_(true);
            }
            else
            {
                cls.attr("__getstate_manages_dict__") = nb::none();
            }
        }

        static nb::tuple getstate(nb::object obj)
        {
            using namespace diffpy::srreal;
            T& pq = nb::cast<T&>(obj);
            // temporarily remove structure from the pair quantity
            StructureAdapterPtr& structure_slot = pq.getStructure();
            StructureAdapterPtr pstru = structure_slot;
            nb::object stru = pstru ? nb::cast(pstru) : nb::none();
            structure_slot.reset();
            struct RestoreStructure
            {
                StructureAdapterPtr& slot;
                StructureAdapterPtr pstru;
                bool active = true;

                ~RestoreStructure()
                {
                    restore();
                }

                void restore() noexcept
                {
                    if (!active)  return;
                    slot = pstru;
                    active = false;
                }
            } restore{structure_slot, pstru};

            nb::object state0;
            try
            {
                state0 = Super::getstate(obj);
            }
            catch (...)
            {
                restore.restore();
                throw;
            }
            restore.restore();
            return nb::make_tuple(state0, stru);
        }


        static void setstate(
                nb::object obj, nb::tuple state)
        {
            using namespace diffpy::srreal;
            ensure_tuple_length(state, 2);
            // restore the state using boost serialization
            nb::tuple state0(state[0]);
            Super::setstate(obj, state0);
            // restore the structure object
            StructureAdapterPtr pstru = nb::cast<StructureAdapterPtr>(state[1]);
            T& pq = nb::cast<T&>(obj);
            pq.getStructure() = pstru;
        }

        static nb::tuple reduce(nb::object obj)
        {
            return Super::reduce(obj);
        }

};  // class PairQuantityPickleSuite


template <class T, class Storage=T>
class StructureAdapterPickleSuite
{
    public:

        template <typename C>
        static void bind(C& cls)
        {
            cls
                .def("__getstate__", getstate)
                .def("__setstate__", setstate)
                .def("__reduce__", reduce)
                ;
            cls.attr("__getstate_manages_dict__") = nb::bool_(true);
        }


        static nb::tuple getstate(nb::object obj)
        {
            diffpy::srreal::StructureAdapterPtr adpt =
                nb::cast<diffpy::srreal::StructureAdapterPtr>(obj);
            nb::object content = nb::none();
            if (frompython(adpt))
            {
                const T& tobj = nb::cast<const T&>(obj);
                content = serialization_tobytes(tobj);
            }
            return nb::make_tuple(content, get_instance_dict(obj));
        }


        static void setstate(
                nb::object obj, nb::tuple state)
        {
            ensure_tuple_length(state, 2);
            // Restore the C++ data from state[0] for Python built-objects.
            // state[0] is None for C++ objects and there is no need to do
            // anything as those were already restored by string constructor.
            nb::object st0 = nb::borrow<nb::object>(state[0]);
            if (!st0.is_none())
            {
                T& tobj = nb::cast<T&>(obj);
                diffpy::serialization_fromstring(tobj, bytes_to_string(st0));
            }
            // restore the object's __dict__
            restore_instance_dict(obj, state[1]);
        }

        static nb::tuple reduce(nb::object obj)
        {
            diffpy::srreal::StructureAdapterPtr adpt =
                nb::cast<diffpy::srreal::StructureAdapterPtr>(obj);
            if (frompython(adpt))
            {
                return nb::make_tuple(
                    runtime_type(obj),
                    nb::make_tuple(),
                    obj.attr("__getstate__")()
                );
            }

            nb::object restore = nb::module_::import_(
                "diffpy.srreal.srreal_ext").attr("_restoreStructureAdapter");
            nb::bytes content = serialization_tobytes(adpt);
            return nb::make_tuple(
                restore,
                nb::make_tuple(content),
                nb::make_tuple(nb::none(), get_instance_dict(obj))
            );
        }

        static bool getstate_manages_dict()  { return true; }


    private:

        static bool frompython(
                diffpy::srreal::StructureAdapterPtr adpt)
        {
            return bool(std::dynamic_pointer_cast<PythonTrampolineTag>(adpt));
        }

};  // class StructureAdapterPickleSuite

/// Helper function for creating Python constructor from string
/// that restores c++ non-wrapped classes.
using diffpy::srreal::StructureAdapterPtr;

StructureAdapterPtr
createStructureAdapterFromString(const std::string &content);

template <class Adapter>
std::shared_ptr<Adapter>
createAdapterFromString(const std::string &content) 
{
    StructureAdapterPtr base = createStructureAdapterFromString(content);

    auto rv = std::dynamic_pointer_cast<Adapter>(base);
    if (!rv) 
    {
        throw nb::type_error(
            "serialized content does not contain the requested adapter type"
        );
    }

    return rv;
}

template <class Adapter>
void StructureAdapter_constructor(
        nb::pointer_and_handle<Adapter> self, nb::bytes content)
{
    if (!runtime_type(self.h).is(nb::type<Adapter>()))
    {
        throw nb::type_error(
            "serialized StructureAdapter constructor is only supported "
            "for native extension types"
        );
    }
    std::shared_ptr<Adapter> adpt =
        createAdapterFromString<Adapter>(bytes_to_string(content));
    new (self.p) Adapter(*adpt);
}

}   // namespace srrealmodule

#endif  // SRREAL_PICKLING_HPP_INCLUDED
