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
            ::boost::python::incref(mpyptr);
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


/// helper wrapper function for return value conversion.
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
    const char* doc_createByType = CString(d["createByType"]);
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
        .def("createByType", &B::createByType,
                bp::arg("tp"), doc_createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getRegisteredTypes_asset<B>,
                doc_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        ;
    return boostpythonclass;
}


}   // namespace srrealmodule

#endif  // SRREAL_REGISTRY_HPP_INCLUDED
