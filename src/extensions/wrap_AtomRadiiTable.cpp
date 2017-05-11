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

#include <boost/python/class.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/serialization/export.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"
#include "srreal_registry.hpp"

#include <diffpy/srreal/AtomRadiiTable.hpp>
#include <diffpy/srreal/ConstantRadiiTable.hpp>

namespace srrealmodule {
namespace nswrap_AtomRadiiTable {

using namespace boost::python;
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
    public wrapper_srreal<AtomRadiiTable>
{
    public:

        // HasClassRegistry methods

        AtomRadiiTablePtr create() const
        {
            object rv = this->get_pure_virtual_override("create")();
            return mconfigurator.fetch(rv);
        }

        AtomRadiiTablePtr clone() const
        {
            return this->get_pure_virtual_override("clone")();
        }

        const std::string& type() const
        {
            object tp = this->get_pure_virtual_override("type")();
            mtype = extract<std::string>(tp);
            return mtype;
        }

        // own methods

        double standardLookup(const std::string& smbl) const
        {
            return this->get_pure_virtual_override("_standardLookup")(smbl);
        }

    protected:

        // HasClassRegistry method

        void setupRegisteredObject(AtomRadiiTablePtr p) const
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

void wrap_AtomRadiiTable()
{
    using namespace nswrap_AtomRadiiTable;
    using boost::noncopyable;

    class_<AtomRadiiTableWrap, noncopyable>
        atomradiitable("AtomRadiiTable", doc_AtomRadiiTable);
    wrap_registry_methods(atomradiitable)
        .def("lookup",
                &AtomRadiiTable::lookup, arg("smbl"),
                doc_AtomRadiiTable_lookup)
        .def("_standardLookup",
                &AtomRadiiTable::standardLookup,
                arg("smbl"), doc_AtomRadiiTable__standardLookup)
        .def("setCustom",
                &AtomRadiiTable::setCustom,
                (arg("smbl"), arg("radius")),
                doc_AtomRadiiTable_setCustom)
        .def("fromString",
                &AtomRadiiTable::fromString,
                doc_AtomRadiiTable_fromString)
        .def("resetCustom",
                &AtomRadiiTable::resetCustom, arg("smbl"),
                doc_AtomRadiiTable_resetCustom)
        .def("resetAll",
                &AtomRadiiTable::resetAll,
                doc_AtomRadiiTable_resetAll)
        .def("getAllCustom",
                getAllCustom_asdict<AtomRadiiTable>,
                doc_AtomRadiiTable_getAllCustom)
        .def("toString",
                &AtomRadiiTable::toString, arg("separator")=",",
                doc_AtomRadiiTable_toString)
        .def_pickle(SerializationPickleSuite<AtomRadiiTable,DICT_PICKLE>())
        ;

    register_ptr_to_python<AtomRadiiTablePtr>();

    class_<ConstantRadiiTable, bases<AtomRadiiTable> >(
            "ConstantRadiiTable", doc_ConstantRadiiTable)
        // docstring updates
        .def("create", &ConstantRadiiTable::create,
                doc_ConstantRadiiTable_create)
        .def("clone", &ConstantRadiiTable::clone,
                doc_ConstantRadiiTable_clone)
        .def("_standardLookup",
                &ConstantRadiiTable::standardLookup,
                arg("smbl"), doc_ConstantRadiiTable__standardLookup)
        // own methods
        .def("setDefault",
                &ConstantRadiiTable::setDefault,
                arg("radius"),
                doc_ConstantRadiiTable_setDefault)
        .def("getDefault",
                &ConstantRadiiTable::getDefault,
                doc_ConstantRadiiTable_getDefault)
        .def_pickle(SerializationPickleSuite<ConstantRadiiTable,DICT_IGNORE>())
        ;

}

}   // namespace srrealmodule

// Serialization -------------------------------------------------------------

BOOST_CLASS_EXPORT(srrealmodule::nswrap_AtomRadiiTable::AtomRadiiTableWrap)

// End of file
