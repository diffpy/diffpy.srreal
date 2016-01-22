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
* Bindings to the diffpy::Attributes class.
*
*****************************************************************************/

#include <boost/python/class.hpp>

#include <cassert>

#include <diffpy/Attributes.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_Attributes {

using std::string;
using namespace boost;
using namespace diffpy::attributes;

// docstrings ----------------------------------------------------------------

const char* doc_Attributes = "\
This class manages named C++ double attributes owned by an instance.\n\
The derived objects own double attributes that can be looked up from C++\n\
by name, without having to know a full C++ interface of their classes.\n\
";

const char* doc_Attributes__getDoubleAttr = "\
Return value of a named C++ double attribute owned by this object.\n\
\n\
name -- string name of a double attribute\n\
\n\
Return double.\n\
Raise AttributeError for invalid name.\n\
";

const char* doc_Attributes__setDoubleAttr = "\
Set named C++ double attribute to the specified value.\n\
\n\
name     -- string name of a double attribute\n\
value    -- new value of the attribute\n\
\n\
No return value.\n\
Raise AttributeError for invalid name or read-only attribute.\n\
";

const char* doc_Attributes__hasDoubleAttr = "\
Check if named C++ double attribute exists.\n\
\n\
name     -- string name of a double attribute\n\
\n\
Return bool.\n\
";

const char* doc_Attributes__namesOfDoubleAttributes = "\
Return set of C++ double attributes owned by this object.\n\
";

const char* doc_Attributes__namesOfWritableDoubleAttributes = "\
Return set of writable C++ double attributes related to this object.\n\
";

const char* doc_Attributes__registerDoubleAttribute = "\
Register a C++ double attribute that is defined in Python.\n\
This must be called from the __init__ method of a Python class\n\
that derives from the Attributes.\n\
\n\
name     -- string name of a double attribute, must be a unique\n\
            attribute name for this instance\n\
getter   -- optional function that returns the attribute value\n\
setter   -- optional function that sets the attribute value.\n\
            The attribute is read-only when None.\n\
\n\
When both getter and setter are None, register standard Python\n\
attribute access as a C++ double attribute.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYSET_METHOD_WRAPPER(namesOfDoubleAttributes,
        namesOfDoubleAttributes_asset)
DECLARE_PYSET_METHOD_WRAPPER(namesOfWritableDoubleAttributes,
        namesOfWritableDoubleAttributes_asset)

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
            if (this->isreadonly())  throwDoubleAttributeReadOnly();
            // verify that mowner is indeed the obj wrapper
            python::object owner(python::borrowed(mowner));
            assert(obj == python::extract<Attributes*>(owner));
            msetter(owner, value);
        }


        bool isreadonly() const
        {
            return (msetter.ptr() == Py_None);
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
        python::object mod = python::import("diffpy.srreal.attributes");
        g = mod.attr("_pyattrgetter")(name);
        s = mod.attr("_pyattrsetter")(name);
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
    // ready for class definition
    class_<Attributes>("Attributes", doc_Attributes)
        .def("_getDoubleAttr", &Attributes::getDoubleAttr,
                doc_Attributes__getDoubleAttr)
        .def("_setDoubleAttr", &Attributes::setDoubleAttr,
                doc_Attributes__setDoubleAttr)
        .def("_hasDoubleAttr", &Attributes::hasDoubleAttr,
                doc_Attributes__hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                namesOfDoubleAttributes_asset<Attributes>,
                doc_Attributes__namesOfDoubleAttributes)
        .def("_namesOfWritableDoubleAttributes",
                namesOfWritableDoubleAttributes_asset<Attributes>,
                doc_Attributes__namesOfWritableDoubleAttributes)
        .def("_registerDoubleAttribute",
                registerPythonDoubleAttribute,
                (python::arg("getter")=None, python::arg("setter")=None),
                doc_Attributes__registerDoubleAttribute)
        ;
}

}   // namespace srrealmodule

// End of file
