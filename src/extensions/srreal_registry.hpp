/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   (c) 2016 Brookhaven Science Associates,
*                   Brookhaven National Laboratory.
*                   All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Utilities for wrapping classes that derive from HasClassRegistry.
*
*****************************************************************************/

#ifndef SRREAL_REGISTRY_HPP_INCLUDED
#define SRREAL_REGISTRY_HPP_INCLUDED

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace srrealmodule {

/// helper for remembering instances of Python-derived class.
void register_for_cleanup(PyObject*);


/// template class for handling Python Wrapper classes in C++ class registry
template <class T>
class wrapper_registry_configurator
{
        typedef typename T::SharedPtr TSharedPtr;
        typedef typename T::SharedPtr::element_type* TPtr;

    public:

        // constructor
        wrapper_registry_configurator() : mcptr(0), mpyptr(0)  { }

        // methods
        /// the fetch method should be called only from the wrapped method
        /// create() to remember pointers to the last Python object and
        /// the C++ instance that it wraps.
        TSharedPtr fetch(nb::object& obj) const
        {
            TSharedPtr p = nb::cast<TSharedPtr>(obj);
            mcptr = p.get();
            mpyptr = obj.ptr();
            return p;
        }

        /// the setup function should be executed just once from the
        /// HasClassRegistry::setupRegisteredObject method in the wrapper
        /// class.  This ensures the Python object in a prototype factory
        /// stays alive when the process is shut down.
        void setup(TSharedPtr ptr) const
        {
            assert(mcptr && mcptr == ptr.get());
            register_for_cleanup(mpyptr);
            mcptr = 0;
            mpyptr = 0;
        }

    private:

        // data
        mutable TPtr mcptr;
        mutable PyObject* mpyptr;
};


/// retrieve a dictionary of Python-defined docstrings for the cls class.
nb::object get_registry_docstrings(nb::object& cls);


/// helper wrapper functions for return value conversions.

template <class W>
nb::object getAliasedTypes_asdict()
{
    return convertToPythonDict(W::getAliasedTypes());
}

template <class W>
nb::object getRegisteredTypes_asset()
{
    return convertToPythonSet(W::getRegisteredTypes());
}


/// template function that wraps HasClassRegistry methods
template <class C, class... Extra>
nb::class_<C, Extra...>& wrap_registry_methods(nb::class_<C, Extra...>& cls)
{
    // get docstrings for the class registry methods.
    nb::object d = get_registry_docstrings(cls);
    std::string doc_create = nb::cast<std::string>(d["create"]);
    std::string doc_clone = nb::cast<std::string>(d["clone"]);
    std::string doc_type = nb::cast<std::string>(d["type"]);
    std::string doc__registerThisType = nb::cast<std::string>(d["_registerThisType"]);
    std::string doc__aliasType = nb::cast<std::string>(d["_aliasType"]);
    std::string doc__deregisterType = nb::cast<std::string>(d["_deregisterType"]);
    std::string doc_createByType = nb::cast<std::string>(d["createByType"]);
    std::string doc_isRegisteredType = nb::cast<std::string>(d["isRegisteredType"]);
    std::string doc_getAliasedTypes = nb::cast<std::string>(d["getAliasedTypes"]);
    std::string doc_getRegisteredTypes = nb::cast<std::string>(d["getRegisteredTypes"]);
    // define the class registry related methods.
    cls
        .def("create", &C::create,
                doc_create.c_str())
        .def("clone", &C::clone,
                doc_clone.c_str())
        .def("type", &C::type,
                nb::rv_policy::copy,
                doc_type.c_str())
        .def("_registerThisType", &C::registerThisType,
                doc__registerThisType.c_str())
        .def_static("_aliasType", &C::aliasType,
                nb::arg("tp"), nb::arg("alias"),
                doc__aliasType.c_str())
        .def_static("_deregisterType", &C::deregisterType,
                nb::arg("tp"),
                doc__deregisterType.c_str())
        .def_static("createByType", &C::createByType,
                nb::arg("tp"),
                doc_createByType.c_str())
        .def_static("isRegisteredType", &C::isRegisteredType,
                nb::arg("tp"),
                doc_isRegisteredType.c_str())
        .def_static("getAliasedTypes", &getAliasedTypes_asdict<C>,
                doc_getAliasedTypes.c_str())
        .def_static("getRegisteredTypes", &getRegisteredTypes_asset<C>,
                doc_getRegisteredTypes.c_str())
        ;
    return cls;
}


}   // namespace srrealmodule

#endif  // SRREAL_REGISTRY_HPP_INCLUDED
