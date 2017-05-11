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

#include <boost/python/object.hpp>
#include <boost/python/extract.hpp>

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
        TSharedPtr fetch(::boost::python::object& obj) const
        {
            TSharedPtr p = ::boost::python::extract<TSharedPtr>(obj);
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
::boost::python::object get_registry_docstrings(::boost::python::object& cls);


/// helper wrapper functions for return value conversions.

template <class W>
::boost::python::object getAliasedTypes_asdict()
{
    return convertToPythonDict(W::getAliasedTypes());
}

template <class W>
::boost::python::object getRegisteredTypes_asset()
{
    return convertToPythonSet(W::getRegisteredTypes());
}


/// template function that wraps HasClassRegistry methods
template <class C>
C& wrap_registry_methods(C& boostpythonclass)
{
    namespace bp = boost::python;
    using namespace boost::python;
    typedef typename C::wrapped_type::base B;
    typedef extract<const char*> CString;
    // get docstrings for the class registry methods.
    object d = get_registry_docstrings(boostpythonclass);
    const char* doc_create = CString(d["create"]);
    const char* doc_clone = CString(d["clone"]);
    const char* doc_type = CString(d["type"]);
    const char* doc__registerThisType = CString(d["_registerThisType"]);
    const char* doc__aliasType = CString(d["_aliasType"]);
    const char* doc__deregisterType = CString(d["_deregisterType"]);
    const char* doc_createByType = CString(d["createByType"]);
    const char* doc_isRegisteredType = CString(d["isRegisteredType"]);
    const char* doc_getAliasedTypes = CString(d["getAliasedTypes"]);
    const char* doc_getRegisteredTypes = CString(d["getRegisteredTypes"]);
    // define the class registry related methods.
    boostpythonclass
        .def("create", &B::create, doc_create)
        .def("clone", &B::clone, doc_clone)
        .def("type", &B::type,
                return_value_policy<copy_const_reference>(),
                doc_type)
        .def("_registerThisType", &B::registerThisType,
                doc__registerThisType)
        .def("_aliasType", &B::aliasType,
                (bp::arg("tp"), bp::arg("alias")), doc__aliasType)
        .staticmethod("_aliasType")
        .def("_deregisterType", &B::deregisterType,
                bp::arg("tp"), doc__deregisterType)
        .staticmethod("_deregisterType")
        .def("createByType", &B::createByType,
                bp::arg("tp"), doc_createByType)
        .staticmethod("createByType")
        .def("isRegisteredType", &B::isRegisteredType,
                bp::arg("tp"), doc_isRegisteredType)
        .staticmethod("isRegisteredType")
        .def("getAliasedTypes", getAliasedTypes_asdict<B>,
                doc_getAliasedTypes)
        .staticmethod("getAliasedTypes")
        .def("getRegisteredTypes", getRegisteredTypes_asset<B>,
                doc_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;
    return boostpythonclass;
}


}   // namespace srrealmodule

#endif  // SRREAL_REGISTRY_HPP_INCLUDED
