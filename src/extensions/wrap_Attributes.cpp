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

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>

#include <cassert>

#include <diffpy/Attributes.hpp>

#include "srreal_converters.hpp"

namespace nb = nanobind;

namespace srrealmodule {
namespace nswrap_Attributes {

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
        PythonDoubleAttribute(nb::object owner,
                nb::object getter, nb::object setter)
        {
            // PythonDoubleAttribute needs to know its Python owner, but it
            // has to use a borrowed reference otherwise the owner would be
            // never freed.  We store a pointer to the raw object and create
            // a borrowed boost python wrapper as necessary.
            mowner = owner.ptr();
            mgetter = getter;
            msetter = setter;
        }


        double getValue(const Attributes* obj) const override
        {
            nb::gil_scoped_acquire gil;
            // verify that mowner is indeed the obj wrapper
            nb::object owner = nb::borrow<nb::object>(mowner);
            assert(obj == nb::cast<const Attributes*>(owner));
            nb::object pyrv = mgetter(owner);
            return nb::cast<double>(pyrv);
        }


        void setValue(Attributes* obj, double value) override
        {
            nb::gil_scoped_acquire gil;
            if (this->isreadonly())  throwDoubleAttributeReadOnly();
            // verify that mowner is indeed the obj wrapper
            nb::object owner = nb::borrow<nb::object>(mowner);
            assert(obj == nb::cast<Attributes*>(owner));
            msetter(owner, value);
        }


        bool isreadonly() const override
        {
            return (msetter.is_none());
        }

    private:

        // data
        PyObject* mowner;
        nb::object mgetter;
        nb::object msetter;

};  // class PythonDoubleAttribute


void registerPythonDoubleAttribute(nb::object owner,
        const std::string& name, nb::object g, nb::object s)
{
    // when neither getter no setter are specified,
    // make it use normal python attribute access
    if (g.is_none() && s.is_none())
    {
        nb::object mod = nb::module_::import_("diffpy.srreal.attributes");
        g = mod.attr("_pyattrgetter")(name);
        s = mod.attr("_pyattrsetter")(name);
    }
    Attributes* cowner = nb::cast<Attributes*>(owner);
    BaseDoubleAttribute* pa = new PythonDoubleAttribute(owner, g, s);
    registerBaseDoubleAttribute(cowner, name, pa);
}

}   // namespace nswrap_Attributes

// Wrapper definition --------------------------------------------------------

void wrap_Attributes(nb::module_& m)
{
    using namespace nswrap_Attributes;
    // ready for class definition
    nb::class_<Attributes>(
            m, "Attributes",
            nb::dynamic_attr(),
            nb::is_weak_referenceable(),
            doc_Attributes)
        .def(nb::init<>())
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
                nb::arg("name"),
                nb::arg("getter") = nb::none(),
                nb::arg("setter") = nb::none(),
                doc_Attributes__registerDoubleAttribute)
        ;
}

}   // namespace srrealmodule

// End of file
