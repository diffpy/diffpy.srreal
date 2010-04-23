/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Bindings to the diffpy::Attributes class.
*
* $Id$
*
*****************************************************************************/

#include <cassert>
#include <sstream>
#include <boost/python.hpp>

#include <diffpy/Attributes.hpp>
#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_Attributes {

using std::string;
using namespace boost;
using namespace diffpy::attributes;


DECLARE_PYSET_METHOD_WRAPPER(namesOfDoubleAttributes,
        namesOfDoubleAttributes_asset)


const char* getattr_setattr_code = "\
def __getattr__(self, name):\n\
    try:\n\
        rv = self._getDoubleAttr(name)\n\
    except Exception, e:\n\
        raise AttributeError(e)\n\
    return rv\n\
\n\
\n\
def __setattr__(self, name, value):\n\
    if self._hasDoubleAttr(name):\n\
        self._setDoubleAttr(name, value)\n\
    else:\n\
        object.__setattr__(self, name, value)\n\
    return\n\
";

// Helper class to handle double attributes defined from Python

class PythonDoubleAttribute : public BaseDoubleAttribute
{
    public:

        // constructor
        PythonDoubleAttribute(python::object owner,
                python::object getter, python::object setter)
        {
            // PythonDoubleAttribute needs to know its Python owner, but it
            // has to use a borrowed reference otherwise the owner would be
            // never freed.  We store a pointer to the raw object and create
            // a borrowed boost python wrapper as necessary.
            mowner = owner.ptr();
            mgetter = getter;
            msetter = setter;
        }


        double getValue(const Attributes* obj) const
        {
            // verify that mowner is indeed the obj wrapper
            python::object owner(python::borrowed(mowner));
            assert(obj == python::extract<const Attributes*>(owner));
            python::object pyrv = mgetter(owner);
            double rv = python::extract<double>(pyrv);
            return rv;
        }


        void setValue(Attributes* obj, double value)
        {
            if (msetter.ptr() == Py_None)  throwDoubleAttributeReadOnly();
            // verify that mowner is indeed the obj wrapper
            python::object owner(python::borrowed(mowner));
            assert(obj == python::extract<Attributes*>(owner));
            msetter(owner, value);
        }

    private:

        // data
        PyObject* mowner;
        python::object mgetter;
        python::object msetter;

};  // class PythonDoubleAttribute


void registerPythonDoubleAttribute(python::object owner,
        const string& name, python::object g, python::object s)
{
    // when neither getter no setter are specified,
    // make it use normal python attribute access
    if (g.ptr() == Py_None && s.ptr() == Py_None)
    {
        python::object globals = python::import("__main__").attr("__dict__");
        python::dict locals;
        std::ostringstream gcode;
        gcode << "lambda obj : object.__getattribute__(obj, '" << name << "')";
        g = python::eval(gcode.str().c_str(), globals, locals);
        std::ostringstream scode;
        scode << "lambda obj, v : object.__setattr__(obj, '" << name << "', v)";
        s = python::eval(scode.str().c_str(), globals, locals);
    }
    Attributes* cowner = python::extract<Attributes*>(owner);
    BaseDoubleAttribute* pa = new PythonDoubleAttribute(owner, g, s);
    registerBaseDoubleAttribute(cowner, name, pa);
}

}   // namespace nswrap_Attributes

// Wrapper definition --------------------------------------------------------

void wrap_Attributes()
{
    using namespace nswrap_Attributes;
    using namespace boost::python;
    const python::object None;
    // store custom __getattr__ and __setattr__ in the locals dictionary
    object globals = import("__main__").attr("__dict__");
    dict locals;
    exec(getattr_setattr_code, globals, locals);
    // ready for class definition
    class_<Attributes>("Attributes")
        .def("_getDoubleAttr", &Attributes::getDoubleAttr)
        .def("_setDoubleAttr", &Attributes::setDoubleAttr)
        .def("_hasDoubleAttr", &Attributes::hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                namesOfDoubleAttributes_asset<Attributes>)
        .def("_registerDoubleAttribute",
                registerPythonDoubleAttribute,
                (python::arg("getter")=None, python::arg("setter")=None))
        .def("__getattr__", locals["__getattr__"])
        .def("__setattr__", locals["__setattr__"])
        ;
}

}   // namespace srrealmodule

// End of file
