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

}   // namespace srrealmodule

#endif  // SRREAL_REGISTRY_HPP_INCLUDED
