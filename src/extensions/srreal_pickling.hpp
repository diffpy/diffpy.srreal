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

#include <string>
#include <sstream>
#include <type_traits>

#include <diffpy/serialization.hpp>
#include <diffpy/srreal/forwardtypes.hpp>

namespace nb = nanobind;

namespace srrealmodule {

inline
void ensure_tuple_length(nb::tuple state, const int statelen)
{
    if (state.size() == statelen)  return;
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

template <typename H>
std::string bytes_to_string(const H& h)
{
    nb::bytes py_content(h);
    return std::string(
        static_cast<const char *>(py_content.data()),
        py_content.size()
    );
}


template <class T>
bool is_wrapper(nb::object& obj)
{
    T* p = nullptr;
    return nb::try_cast<T*>(obj, p, false) && p;
}


template <class T>
nb::object resolve_state_object(nb::object value)
{
    // return None if value holds a pristine C++ instance of T
    return is_wrapper<T>(value) ? value : nb::none();
}


template <class T>
void assign_state_object(T target, nb::object value)
{
    if (!value.is_none())  target = value;
}


template <class T>
void construct_for_unpickle(T* tobj)
{
    if constexpr (std::is_default_constructible_v<T> && !std::is_abstract_v<T>)
    {
        new (tobj) T();
    }
    else
    {
        throw nb::type_error(
            "cannot unpickle an uninitialized non-default-constructible "
            "C++ instance"
        );
    }
}


template <class T>
T& ensure_instance_ready(nb::handle obj)
{
    T* tobj = nb::inst_ptr<T>(obj);

    if (!nb::inst_ready(obj)) 
    {
        construct_for_unpickle(tobj);
        nb::inst_mark_ready(obj);
    }

    return *tobj;
}


enum {DICT_IGNORE=false, DICT_PICKLE=true};

template <class T, bool pickledict=DICT_IGNORE>
class SerializationPickleSuite
{
    public:

        template <typename C>
        static void bind(C& cls)
        {
            cls
                .def("__getstate__", getstate)
                .def("__setstate__", setstate)
                ;
        }

        static nb::tuple getstate(nb::object obj)
        {
            const T& tobj = nb::cast<const T&>(obj);
            nb::bytes content = serialization_tobytes(tobj);

            if constexpr (pickledict) 
            {
                return nb::make_tuple(
                    content,
                    nb::getattr(obj, "__dict__")
                );
            } 
            else 
            {
                return nb::make_tuple(content);
            }
        }


        static void setstate(
                nb::object obj, nb::tuple state)
        {
            constexpr int statelen = pickledict ? 2 : 1;
            ensure_tuple_length(state, statelen);
            // load the C++ object
            T& tobj = ensure_instance_ready<T>(obj);
            diffpy::serialization_fromstring(tobj, bytes_to_string(state[0]));
            // restore the object's __dict__
            if constexpr (pickledict)
            {
                nb::object dict_obj = nb::getattr(obj, "__dict__");
                nb::dict d = nb::borrow<nb::dict>(dict_obj);
                d.update(state[1]);
            }
        }

        static bool getstate_manages_dict()  { return pickledict; }

};  // class SerializationPickleSuite


template <class T, bool pickledict=DICT_IGNORE>
class PairQuantityPickleSuite :
    public SerializationPickleSuite<T, pickledict>
{
    private:

        typedef SerializationPickleSuite<T, pickledict> Super;

    public:

        template <typename C>
        static void bind(C& cls)
        {
            cls
                .def("__getstate__", getstate)
                .def("__setstate__", setstate)
                ;
        }

        static nb::tuple getstate(nb::object obj)
        {
            using namespace diffpy::srreal;
            // store the original structure object
            nb::object stru = obj.attr("getStructure")();
            // temporarily remove structure from the pair quantity
            T& pq = ensure_instance_ready<T>(obj);
            StructureAdapterPtr pstru =
                replacePairQuantityStructure(pq, StructureAdapterPtr());
            nb::object state0 = Super::getstate(obj);
            // restore the original structure
            replacePairQuantityStructure(pq, pstru);
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
            T& pq = ensure_instance_ready<T>(obj);
            replacePairQuantityStructure(pq, pstru);
        }

};  // class PairQuantityPickleSuite


template <class T>
class StructureAdapterPickleSuite
{
    public:

        template <typename C>
        static void bind(C& cls)
        {
            cls
                .def("__getstate__", getstate)
                .def("__setstate__", setstate)
                ;
        }


        static nb::tuple getstate(nb::object obj)
        {
            const T& tobj = nb::cast<const T&>(obj);
            nb::bytes content = serialization_tobytes(tobj);
            return nb::make_tuple(content, nb::getattr(obj, "__dict__"));
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
                T& tobj = ensure_instance_ready<T>(obj);
                diffpy::serialization_fromstring(tobj, bytes_to_string(st0));
            }
            // restore the object's __dict__
            nb::object dict_obj = nb::getattr(obj, "__dict__");
            nb::dict d = nb::borrow<nb::dict>(dict_obj);
            d.update(state[1]);
        }


        static bool getstate_manages_dict()  { return true; }


    private:

        static bool frompython(
                diffpy::srreal::StructureAdapterPtr adpt)
        {
            return bool(std::dynamic_pointer_cast<T>(adpt));
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
void StructureAdapter_constructor(Adapter* self, const std::string& content) 
{
    std::shared_ptr<Adapter> adpt = createAdapterFromString<Adapter>(content);
    new (self) Adapter(*adpt);
}

}   // namespace srrealmodule

#endif  // SRREAL_PICKLING_HPP_INCLUDED
