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
* Bindings to the BVParametersTable and BVParam classes.
*
*****************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>

#include <string>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

#include <diffpy/srreal/BVParam.hpp>
#include <diffpy/srreal/BVParametersTable.hpp>

#define BVPARMCIF "bvparm2011.cif"

namespace nb = nanobind;

namespace srrealmodule {
namespace nswrap_BVParametersTable {

using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_BVParam = "\
Storage of bond valence parameters for a given cation-anion pair.\n\
";

const char* doc_BVParam___init__ = "\
Initialize new instance of the BVParam class.\n\
\n\
atom0    -- symbol of the cation atom, no charge specification\n\
valence0 -- integer cation valence, must be positive\n\
atom1    -- symbol of the anion atom, no charge specification\n\
valence1 -- integer anion valence, must be negative\n\
Ro       -- valence parameter Ro\n\
B        -- valence parameter B\n\
ref_id   -- optional reference code in " BVPARMCIF "\n\
";

const char* doc_BVParam___repr__ = "\
String representation of the BVParam object\n\
";

const char* doc_BVParam_bondvalence = "\
Bond valence of the specified distance in Angstroms.\n\
";

const char* doc_BVParam_bondvalenceToDistance = "\
Distance in Angstroms corresponding to specified bond valence.\n\
";

const char* doc_BVParam_setFromCifLine = "\
Update bond valence data from a string formatted as in " BVPARMCIF ".\n\
";

const char* doc_BVParam_atom0 = "\
Bare symbol of the cation atom without charge specification.\n\
";

const char* doc_BVParam_valence0 = "\
Positive integer valence of the cation.\n\
";

const char* doc_BVParam_atom1 = "\
Bare symbol of the anion atom without charge specification.\n\
";

const char* doc_BVParam_valence1 = "\
Negative integer valence of the anion.\n\
";

const char* doc_BVParam_Ro = "\
Bond valence parameter Ro in Angstroms.\n\
";

const char* doc_BVParam_B = "\
Bond valence parameter B in Angstroms.\n\
";

const char* doc_BVParam_ref_id = "\
code of the reference paper in " BVPARMCIF ".\n\
";

const char* doc_BVParametersTable = "\
Lookup table for bond valence parameters of a cation-anion pairs.\n\
";

const char* doc_BVParametersTable_none = "\
Singleton instance of void bond valence parameters.\n\
Also returned by 'lookup' when valence data do not exist.\n\
";

const char* doc_BVParametersTable_getAtomValence = "\
Return signed valence for the specified atom or ion symbol.\n\
\n\
Return valence previously defined by setAtomValence or\n\
interpret the charge suffix, for example, use -2 for \"S2-\".\n\
";

const char* doc_BVParametersTable_setAtomValence = "\
Define custom valence for the specified atom or ion symbol.\n\
\n\
smbl     -- string symbol of atom or ion, for example \"F\".\n\
value    -- signed valence to be used for the symbol, e.g., -1.\n\
\n\
No return value.\n\
";

const char* doc_BVParametersTable_resetAtomValences = "\
Unset any custom valences defined by setAtomValence.\n\
\n\
Valences are thereafter obtained from charge suffixes only.\n\
";

const char* doc_BVParametersTable_lookup1 = "\
Lookup bond valence parameters by a BVParam instance.\n\
\n\
bvparam  -- BVParam object.  The only attributes considered for\n\
            lookup are atom0, valence0, atom1, valence1.\n\
\n\
Return a BVParam object with the looked up data.\n\
Return BVParametersTable.none() if bond valence data do not exist.\n\
";

const char* doc_BVParametersTable_lookup2 = "\
Lookup bond valence parameters by cation-anion pair.\n\
The cation-anion order may be flipped.\n\
\n\
smbl0    -- symbol of the first ion with charge, e.g., \"Na+\"\n\
smbl1    -- symbol of the second ion with charge, e.g., \"O2-\"\n\
\n\
Return a BVParam object with the looked up data.\n\
Return BVParametersTable.none() if bond valence data do not exist.\n\
";

const char* doc_BVParametersTable_lookup4 = "\
Lookup bond valence parameters by cation-anion pair.\n\
The cation-anion order may be flipped.\n\
\n\
atom0    -- bare symbol of the cation atom\n\
valence0 -- positive integer cation valence\n\
atom1    -- bare symbol of the anion atom\n\
valence1 -- negative integer anion valence\n\
\n\
Return a BVParam object with the looked up data.\n\
Return BVParametersTable.none() if bond valence data do not exist.\n\
";

const char* doc_BVParametersTable_setCustom1 = "\
Insert custom bond valence data to the table.\n\
\n\
bvparam  -- BVParam object with the custom bond valence data.\n\
\n\
No return value.\n\
";

const char* doc_BVParametersTable_setCustom6 = "\
Insert custom bond valence data to the table.\n\
\n\
atom0    -- bare symbol of the cation atom\n\
valence0 -- positive integer cation valence\n\
atom1    -- bare symbol of the anion atom\n\
valence1 -- negative integer anion valence\n\
Ro       -- valence parameter Ro in Angstroms\n\
B        -- valence parameter B in Angstroms\n\
ref_id   -- optional reference code in " BVPARMCIF "\n\
\n\
No return value.\n\
";

const char* doc_BVParametersTable_resetCustom1 = "\
Remove custom bond valence data for the specified cation-anion pair.\n\
\n\
bvparam  -- BVParam object.  The only attributes considered for\n\
            custom entry lookup are atom0, valence0, atom1, valence1.\n\
\n\
No return value.\n\
";

const char* doc_BVParametersTable_resetCustom4 = "\
Remove custom bond valence data for the specified cation-anion pair.\n\
The cation-anion order may be flipped.\n\
\n\
atom0    -- bare symbol of the cation atom\n\
valence0 -- positive integer cation valence\n\
atom1    -- bare symbol of the anion atom\n\
valence1 -- negative integer anion valence\n\
\n\
No return value.\n\
";
const char* doc_BVParametersTable_resetAll = "\
Remove all custom bond valence data defined in this table.\n\
";

const char* doc_BVParametersTable_getAll = "\
Return all bond valence parameter values in this table.\n\
\n\
Return a set of BVParam objects.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYSET_METHOD_WRAPPER(getAll, getAll_asset)
void setCustom6(BVParametersTable& obj,
                const std::string& atom0,
                int valence0,
                const std::string& atom1,
                int valence1,
                double Ro,
                double B,
                const std::string& ref_id) {
    obj.setCustom(atom0, valence0, atom1, valence1, Ro, B, ref_id);
}

nb::object repr_BVParam(const BVParam& bp)
{
    if (bp == BVParametersTable::none())  return nb::str("BVParam()");
    return nb::str(
        "BVParam({!r}, {:d}, {!r}, {:d}, Ro={}, B={}, ref_id={!r})"
    ).attr("format")(
        bp.matom0,
        bp.mvalence0,
        bp.matom1,
        bp.mvalence1,
        bp.mRo,
        bp.mB,
        bp.mref_id
    );
}


nb::object singleton_none()
{
    const char* nameofnone = "__BVParam_singleton_none";
    nb::module_ mod = nb::module_::import_("diffpy.srreal.srreal_ext");
    static bool noneassigned = false;
    if (!noneassigned)
    {
        mod.attr(nameofnone) = nb::cast(BVParametersTable::none(), nb::rv_policy::copy);
        noneassigned = true;
    }
    return mod.attr(nameofnone);
}

}   // namespace nswrap_BVParametersTable

// Wrapper definition --------------------------------------------------------

void wrap_BVParametersTable(nb::module_& m)
{
    using namespace nswrap_BVParametersTable;

    nb::class_<BVParam> bvparam(m, "BVParam", doc_BVParam);
    bvparam
        .def(nb::init<>())
        .def(nb::init<const std::string&, int,
                      const std::string&, int,
                      double, double, std::string>(),
             nb::arg("atom0"),
             nb::arg("valence0"),
             nb::arg("atom1"),
             nb::arg("valence1"),
             nb::arg("Ro") = 0.0,
             nb::arg("B") = 0.0,
             nb::arg("ref_id") = "",
             doc_BVParam___init__)
        .def("__repr__", repr_BVParam, doc_BVParam___repr__)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("bondvalence", &BVParam::bondvalence,
                nb::arg("distance"), doc_BVParam_bondvalence)
        .def("bondvalenceToDistance", &BVParam::bondvalenceToDistance,
                nb::arg("valence"), doc_BVParam_bondvalenceToDistance)
        .def("setFromCifLine", &BVParam::setFromCifLine,
                doc_BVParam_setFromCifLine)
        .def_ro("atom0", &BVParam::matom0, doc_BVParam_atom0)
        .def_ro("valence0", &BVParam::mvalence0, doc_BVParam_valence0)
        .def_ro("atom1", &BVParam::matom1, doc_BVParam_atom1)
        .def_ro("valence1", &BVParam::mvalence1, doc_BVParam_valence1)
        .def_rw("Ro", &BVParam::mRo, doc_BVParam_Ro)
        .def_rw("B", &BVParam::mB, doc_BVParam_B)
        .def_rw("ref_id", &BVParam::mref_id, doc_BVParam_ref_id)
        ;
        SerializationPickleSuite<BVParam, DICT_IGNORE>::bind(bvparam);

    nb::class_<BVParametersTable>
        bvtable(m, "BVParametersTable", doc_BVParametersTable);
    bvtable
        .def(nb::init<>())
        .def_static("none", singleton_none, doc_BVParametersTable_none)
        .def("getAtomValence", &BVParametersTable::getAtomValence,
                nb::arg("smbl"),
                doc_BVParametersTable_getAtomValence)
        .def("setAtomValence", &BVParametersTable::setAtomValence,
                nb::arg("smbl"), nb::arg("value"),
                doc_BVParametersTable_setAtomValence)
        .def("resetAtomValences", &BVParametersTable::resetAtomValences,
                doc_BVParametersTable_resetAtomValences)
        .def("lookup",
             [](const BVParametersTable &obj,
                const BVParam &bvparam) -> BVParam {
                 return obj.lookup(bvparam);
             },
             nb::arg("bvparam"),
             doc_BVParametersTable_lookup1)

        .def("lookup",
             [](const BVParametersTable &obj,
                const std::string &smbl0,
                const std::string &smbl1) -> BVParam {
                 return obj.lookup(smbl0, smbl1);
             },
             nb::arg("smbl0"),
             nb::arg("smbl1"),
             doc_BVParametersTable_lookup2)

        .def("lookup",
             [](const BVParametersTable &obj,
                const std::string &atom0,
                int valence0,
                const std::string &atom1,
                int valence1) -> BVParam {
                 return obj.lookup(atom0, valence0, atom1, valence1);
             },
             nb::arg("atom0"),
             nb::arg("valence0"),
             nb::arg("atom1"),
             nb::arg("valence1"),
             doc_BVParametersTable_lookup4)

        .def("setCustom",
             [](BVParametersTable &obj, const BVParam &bvparam) {
                 obj.setCustom(bvparam);
             },
             nb::arg("bvparam"),
             doc_BVParametersTable_setCustom1)

        .def("setCustom",
             &setCustom6,
             nb::arg("atom0"),
             nb::arg("valence0"),
             nb::arg("atom1"),
             nb::arg("valence1"),
             nb::arg("Ro"),
             nb::arg("B"),
             nb::arg("ref_id") = "",
             doc_BVParametersTable_setCustom6)
        .def("resetCustom",
             [](BVParametersTable &obj, const BVParam &bvparam) {
                 obj.resetCustom(bvparam);
             },
             nb::arg("bvparam"),
             doc_BVParametersTable_resetCustom1)

        .def("resetCustom",
             [](BVParametersTable &obj,
                const std::string &atom0,
                int valence0,
                const std::string &atom1,
                int valence1) {
                 obj.resetCustom(atom0, valence0, atom1, valence1);
             },
             nb::arg("atom0"),
             nb::arg("valence0"),
             nb::arg("atom1"),
             nb::arg("valence1"),
             doc_BVParametersTable_resetCustom4)
        .def("resetAll", &BVParametersTable::resetAll,
                doc_BVParametersTable_resetAll)
        .def("getAll", getAll_asset<BVParametersTable>,
                doc_BVParametersTable_getAll)
        ;
        SerializationPickleSuite<BVParametersTable, DICT_IGNORE>::bind(bvtable);
        
}

}   // namespace srrealmodule

// End of file
