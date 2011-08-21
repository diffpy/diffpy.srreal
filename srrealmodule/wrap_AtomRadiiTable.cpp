/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2011 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Bindings to the AtomRadiiTable class.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>

#include <diffpy/srreal/AtomRadiiTable.hpp>
#include <diffpy/srreal/ZeroRadiiTable.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_AtomRadiiTable {

using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_AtomRadiiTable_create = "FIXME";
const char* doc_AtomRadiiTable_clone = "FIXME";
const char* doc_AtomRadiiTable_type = "FIXME";
const char* doc_AtomRadiiTable__registerThisType = "FIXME";
const char* doc_AtomRadiiTable_createByType = "FIXME";
const char* doc_AtomRadiiTable_getRegisteredTypes = "FIXME";

const char* doc_AtomRadiiTable = "\
Lookup table for empirical atom radii.\n\
";

const char* doc_AtomRadiiTable_lookup = "\
Return empirical radius of an atom in Angstroms.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
\n\
Return atom radius in Angstroms.\n\
";

const char* doc_AtomRadiiTable__tableLookup = "\
Standard lookup of empirical atom radius.\n\
This method can be overloaded in a derived class.\n\
This returns zero for the base-class _tableLookup method.\n\
\n\
smbl -- string symbol for atom, ion or isotope\n\
\n\
Return atom radius in Angstroms.\n\
";

const char* doc_AtomRadiiTable_setCustom = "\
Define custom radius for a specified atom type.\n\
\n\
smbl     -- string symbol for atom, ion or isotope\n\
radius   -- custon radius that will be returned by the lookup method\n\
\n\
No return value.\n\
";

const char* doc_AtomRadiiTable_fromString = "\
Define custom radius for one or more atom types from string.\n\
\n\
s    -- string with custom atom radii in (A1:r1, A2:r2, ...) format.\n\
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
Return string of all custom atom radii in (A1:r1, A2:r2, ...) format.\n\
\n\
separator    -- string separator between 'A1:r1' entries, by default ','\n\
\n\
Return string.\n\
";

const char* doc_ZeroRadiiTable = "FIXME";

// wrappers ------------------------------------------------------------------

DECLARE_PYDICT_METHOD_WRAPPER(getAllCustom, getAllCustom_asdict)
DECLARE_PYSET_FUNCTION_WRAPPER(AtomRadiiTable::getRegisteredTypes,
        getAtomRadiiTableTypes_asset)


// Helper class for overloads of AtomRadiiTable methods from Python

class AtomRadiiTableWrap :
    public AtomRadiiTable,
    public wrapper_srreal<AtomRadiiTable>
{
    public:

        // HasClassRegistry methods

        AtomRadiiTablePtr create() const
        {
            return this->get_pure_virtual_override("create")();
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

        double tableLookup(const std::string& smbl) const
        {
            return this->get_pure_virtual_override("_tableLookup")(smbl);
        }

    private:

        mutable std::string mtype;

};  // class AtomRadiiTableWrap


std::string atomradiitable_tostring(AtomRadiiTablePtr obj)
{
    return serialization_tostring(obj);
}


AtomRadiiTablePtr atomradiitable_fromstring(std::string content)
{
    AtomRadiiTablePtr rv;
    serialization_fromstring(rv, content);
    return rv;
}

}   // namespace nswrap_AtomRadiiTable

// Wrapper definition --------------------------------------------------------

void wrap_AtomRadiiTable()
{
    using namespace nswrap_AtomRadiiTable;
    using boost::noncopyable;

    class_<AtomRadiiTableWrap, noncopyable>(
            "AtomRadiiTable", doc_AtomRadiiTable)
        .def("create", &AtomRadiiTable::create,
                doc_AtomRadiiTable_create)
        .def("clone", &AtomRadiiTable::clone,
                doc_AtomRadiiTable_clone)
        .def("type", &AtomRadiiTable::type,
                return_value_policy<copy_const_reference>(),
                doc_AtomRadiiTable_type)
        .def("_registerThisType", &AtomRadiiTable::registerThisType,
                doc_AtomRadiiTable__registerThisType)
        .def("createByType", &AtomRadiiTable::createByType,
                arg("tp"), doc_AtomRadiiTable_createByType)
        .staticmethod("createByType")
        .def("getRegisteredTypes", getAtomRadiiTableTypes_asset,
                doc_AtomRadiiTable_getRegisteredTypes)
        .staticmethod("getRegisteredTypes")
        .def("lookup",
                &AtomRadiiTable::lookup, arg("smbl"),
                doc_AtomRadiiTable_lookup)
        .def("_tableLookup",
                &AtomRadiiTable::tableLookup,
                arg("smbl"), doc_AtomRadiiTable__tableLookup)
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
        .def_pickle(SerializationPickleSuite<AtomRadiiTable>())
        ;

    register_ptr_to_python<AtomRadiiTablePtr>();

    // pickling support functions
    def("_AtomRadiiTable_tostring", atomradiitable_tostring);
    def("_AtomRadiiTable_fromstring", atomradiitable_fromstring);

    class_<ZeroRadiiTable, bases<AtomRadiiTable> >(
            "ZeroRadiiTable", doc_ZeroRadiiTable);

}

}   // namespace srrealmodule

// End of file
