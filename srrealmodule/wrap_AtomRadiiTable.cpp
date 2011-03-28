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

#include <string>
#include <boost/python.hpp>
#include <diffpy/srreal/AtomRadiiTable.hpp>
#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_AtomRadiiTable {

using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

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

// wrappers ------------------------------------------------------------------

// Helper class for overloads of AtomRadiiTable methods from Python

class AtomRadiiTableWrap :
    public AtomRadiiTable,
    public wrapper<AtomRadiiTable>
{
    public:

        double tableLookup(const std::string& smbl) const
        {
            override f = this->get_override("_tableLookup");
            if (f)  return f(smbl);
            return this->default_tableLookup(smbl);
        }

        double default_tableLookup(const std::string& smbl) const
        {
            return this->AtomRadiiTable::tableLookup(smbl);
        }

};  // class AtomRadiiTableWrap

}   // namespace nswrap_AtomRadiiTable

// Wrapper definition --------------------------------------------------------

void wrap_AtomRadiiTable()
{
    using namespace nswrap_AtomRadiiTable;
    using boost::noncopyable;

    class_<AtomRadiiTableWrap, noncopyable>(
            "AtomRadiiTable", doc_AtomRadiiTable)
        .def("lookup",
                &AtomRadiiTable::lookup, arg("smbl"),
                doc_AtomRadiiTable_lookup)
        .def("_tableLookup",
                &AtomRadiiTable::tableLookup,
                &AtomRadiiTableWrap::default_tableLookup,
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
}

}   // namespace srrealmodule

// End of file
