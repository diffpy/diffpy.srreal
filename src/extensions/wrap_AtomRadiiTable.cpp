/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2011 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* Bindings to the AtomRadiiTable class.
*
*****************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/operators.h>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"
#include "srreal_registry.hpp"

#include <diffpy/srreal/AtomRadiiTable.hpp>
#include <diffpy/srreal/ConstantRadiiTable.hpp>

namespace nb = nanobind;

namespace srrealmodule {
namespace nswrap_AtomRadiiTable {

using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_AtomRadiiTable = "\
Base class for looking up empirical atom radii.\n\
This class has virtual methods and cannot be used as is.\n\
\n\
A derived class has to overload the following methods:\n\
\n\
    create(self)\n\
    clone(self)\n\
    type(self)\n\
    _standardLookup(self, smbl)\n\
\n\
Derived class can be added to the global registry of AtomRadiiTable\n\
types by calling the _registerThisType method with any instance.\n\
";

const char* doc_AtomRadiiTable_lookup = "\
Return empirical radius of an atom in Angstroms.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
\n\
Return atom radius in Angstroms.\n\
This method cannot be overloaded in Python.\n\
";

const char* doc_AtomRadiiTable__standardLookup = "\
Standard lookup of empirical atom radius.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
\n\
Return atom radius in Angstroms.\n\
Raise ValueError for unknown atom symbol.\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_AtomRadiiTable_setCustom = "\
Define custom radius for a specified atom type.\n\
\n\
smbl     -- string symbol for atom, ion or isotope\n\
radius   -- custom radius that will be returned by the lookup method\n\
\n\
No return value.\n\
";

const char* doc_AtomRadiiTable_fromString = "\
Define custom radius for one or more atom types from string.\n\
\n\
s    -- string with custom atom radii in 'A1:r1, A2:r2, ...' format.\n\
\n\
No return value.\n\
Raise ValueError for an invalid string format.\n\
";

const char* doc_AtomRadiiTable_resetCustom = "\
Remove custom radius for the specified atom type.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
\n\
No return value.\n\
";

const char* doc_AtomRadiiTable_resetAll = "\
Reset all custom radii defined in this table.\n\
";

const char* doc_AtomRadiiTable_getAllCustom = "\
Return a dictionary of all custom atom radii defined in this table.\n\
";

const char* doc_AtomRadiiTable_toString = "\
Return string of all custom atom radii in 'A1:r1, A2:r2, ...' format.\n\
\n\
separator    -- string separator between 'A1:r1' entries, by default ','\n\
\n\
Return string.\n\
";

const char* doc_ConstantRadiiTable = "\
Atom radii table with the same radius for all atoms, by default 0.\n\
\n\
See setDefault() for changing the default radius or setCustom()\n\
and fromString() for setting a special radius for selected atoms.\n\
";

const char* doc_ConstantRadiiTable_create = "\
Return a new instance of ConstantRadiiTable.\n\
";

const char* doc_ConstantRadiiTable_clone = "\
Return a duplicate of this ConstantRadiiTable object\n\
";

const char* doc_ConstantRadiiTable__standardLookup = "\
Return empirical radius of the given atom in Angstroms.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
\n\
Return atom radius in Angstroms.\n\
This method cannot be overloaded in Python.\n\
";

const char* doc_ConstantRadiiTable_setDefault = "\
Set radius that is by default returned for all atoms.\n\
";

const char* doc_ConstantRadiiTable_getDefault = "\
Return the value of the default atom radius.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYDICT_METHOD_WRAPPER(getAllCustom, getAllCustom_asdict)

// Helper class for overloads of AtomRadiiTable methods from Python

class AtomRadiiTableWrap :
    public AtomRadiiTable,
    public PythonTrampolineTag
{
    public:

        NB_TRAMPOLINE(AtomRadiiTable, 4);

        // HasClassRegistry methods

        AtomRadiiTablePtr create() const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "create", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method AtomRadiiTable.create() called"
                );
            }
            
            nb::object rv = nb_trampoline.base().attr(ticket.key)();
            return mconfigurator.fetch(rv);
        }

        AtomRadiiTablePtr clone() const override
        {
            NB_OVERRIDE_PURE(clone);
        }

        const std::string& type() const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "type", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method AtomRadiiTable.type() called"
                );
            }

            nb::object tp = nb_trampoline.base().attr(ticket.key)();
            mtype = nb::cast<std::string>(tp);
            return mtype;
        }

        // own methods

        double standardLookup(const std::string& smbl) const override
        {
            NB_OVERRIDE_PURE_NAME("_standardLookup", standardLookup, smbl);
        }

    protected:

        // HasClassRegistry method

        void setupRegisteredObject(AtomRadiiTablePtr p) const override
        {
            mconfigurator.setup(p);
        }

    private:

        mutable std::string mtype;
        wrapper_registry_configurator<AtomRadiiTable> mconfigurator;

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
        {
            using boost::serialization::base_object;
            ar & base_object<AtomRadiiTable>(*this);
        }

};  // class AtomRadiiTableWrap

}   // namespace nswrap_AtomRadiiTable

// Wrapper definition --------------------------------------------------------

void wrap_AtomRadiiTable(nb::module_& m)
{
    using namespace nswrap_AtomRadiiTable;

    nb::class_<AtomRadiiTable, AtomRadiiTableWrap>
        atomradiitable(m, "AtomRadiiTable", doc_AtomRadiiTable,
                   nb::dynamic_attr());
    wrap_registry_methods(atomradiitable)
        .def(nb::init<>())
        .def("lookup",
                &AtomRadiiTable::lookup, nb::arg("smbl"),
                doc_AtomRadiiTable_lookup)
        .def("_standardLookup",
                &AtomRadiiTable::standardLookup,
                nb::arg("smbl"), doc_AtomRadiiTable__standardLookup)
        .def("setCustom",
                &AtomRadiiTable::setCustom,
                nb::arg("smbl"), nb::arg("radius"),
                doc_AtomRadiiTable_setCustom)
        .def("fromString",
                &AtomRadiiTable::fromString,
                nb::arg("s"),
                doc_AtomRadiiTable_fromString)
        .def("resetCustom",
                &AtomRadiiTable::resetCustom, nb::arg("smbl"),
                doc_AtomRadiiTable_resetCustom)
        .def("resetAll",
                &AtomRadiiTable::resetAll,
                doc_AtomRadiiTable_resetAll)
        .def("getAllCustom",
                getAllCustom_asdict<AtomRadiiTable>,
                doc_AtomRadiiTable_getAllCustom)
        .def("toString",
                &AtomRadiiTable::toString, nb::arg("separator")=",",
                doc_AtomRadiiTable_toString)
        ;
        SerializationPickleSuite<
            AtomRadiiTable,
            DICT_PICKLE,
            AtomRadiiTableWrap>::bind(atomradiitable);

    nb::class_<ConstantRadiiTable, AtomRadiiTable>
        constantradiitable(m, "ConstantRadiiTable", doc_ConstantRadiiTable);
        // docstring updates
    constantradiitable
        .def(nb::init<>())
        .def("create", &ConstantRadiiTable::create,
                doc_ConstantRadiiTable_create)
        .def("clone", &ConstantRadiiTable::clone,
                doc_ConstantRadiiTable_clone)
        .def("_standardLookup",
                &ConstantRadiiTable::standardLookup,
                nb::arg("smbl"), doc_ConstantRadiiTable__standardLookup)
        // own methods
        .def("setDefault",
                &ConstantRadiiTable::setDefault,
                nb::arg("radius"),
                doc_ConstantRadiiTable_setDefault)
        .def("getDefault",
                &ConstantRadiiTable::getDefault,
                doc_ConstantRadiiTable_getDefault)
        ;
        SerializationPickleSuite<ConstantRadiiTable, DICT_IGNORE>::bind(constantradiitable);

}

}   // namespace srrealmodule

// Serialization -------------------------------------------------------------

BOOST_CLASS_EXPORT(srrealmodule::nswrap_AtomRadiiTable::AtomRadiiTableWrap)

// End of file
