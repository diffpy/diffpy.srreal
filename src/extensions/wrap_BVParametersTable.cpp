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

#include <boost/python/class.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/register_ptr_to_python.hpp>

#include <string>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

#include <diffpy/srreal/BVParam.hpp>
#include <diffpy/srreal/BVParametersTable.hpp>

#define BVPARMCIF "bvparm2011.cif"

namespace srrealmodule {
namespace nswrap_BVParametersTable {

using namespace boost::python;
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

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(setcustom6, setCustom, 6, 7)
DECLARE_PYSET_METHOD_WRAPPER(getAll, getAll_asset)

object repr_BVParam(const BVParam& bp)
{
    if (bp == BVParametersTable::none())  return object("BVParam()");
    object rv = ("BVParam(%r, %i, %r, %i, Ro=%s, B=%s, ref_id=%r)" %
        make_tuple(bp.matom0, bp.mvalence0, bp.matom1, bp.mvalence1,
            bp.mRo, bp.mB, bp.mref_id));
    return rv;
}


object singleton_none()
{
    const char* nameofnone = "__BVParam_singleton_none";
    object mod = import("diffpy.srreal.srreal_ext");
    static bool noneassigned = false;
    if (!noneassigned)
    {
        mod.attr(nameofnone) = object(BVParametersTable::none());
        noneassigned = true;
    }
    return mod.attr(nameofnone);
}

}   // namespace nswrap_BVParametersTable

// Wrapper definition --------------------------------------------------------

void wrap_BVParametersTable()
{
    using namespace nswrap_BVParametersTable;
    using diffpy::srreal::hash_value;
    using std::string;

    class_<BVParam>("BVParam", doc_BVParam)
        .def(init<const string&, int, const string&, int,
                double, double, string>(doc_BVParam___init__,
                    (arg("atom0"), arg("valence0"),
                    arg("atom1"), arg("valence1"), arg("Ro")=0.0, arg("B")=0.0,
                    arg("ref_id")="")))
        .def("__repr__", repr_BVParam, doc_BVParam___repr__)
        .def(self == self)
        .def(self != self)
        .def("__hash__", hash_value)
        .def("bondvalence", &BVParam::bondvalence,
                arg("distance"), doc_BVParam_bondvalence)
        .def("bondvalenceToDistance", &BVParam::bondvalenceToDistance,
                arg("valence"), doc_BVParam_bondvalenceToDistance)
        .def("setFromCifLine", &BVParam::setFromCifLine,
                doc_BVParam_setFromCifLine)
        .def_readonly("atom0", &BVParam::matom0, doc_BVParam_atom0)
        .def_readonly("valence0", &BVParam::mvalence0, doc_BVParam_valence0)
        .def_readonly("atom1", &BVParam::matom1, doc_BVParam_atom1)
        .def_readonly("valence1", &BVParam::mvalence1, doc_BVParam_valence1)
        .def_readwrite("Ro", &BVParam::mRo, doc_BVParam_Ro)
        .def_readwrite("B", &BVParam::mB, doc_BVParam_B)
        .def_readwrite("ref_id", &BVParam::mref_id, doc_BVParam_ref_id)
        .def_pickle(SerializationPickleSuite<BVParam>())
        ;

    typedef const BVParam&(BVParametersTable::*bptb_bvparam_1)(
            const BVParam&) const;
    typedef const BVParam&(BVParametersTable::*bptb_bvparam_2)(
            const string&, const string&) const;
    typedef const BVParam&(BVParametersTable::*bptb_bvparam_4)(
            const string&, int, const string&, int) const;
    typedef void(BVParametersTable::*bptb_void_1)(
            const BVParam&);
    typedef void(BVParametersTable::*bptb_void_4)(
            const string&, int, const string&, int);

    class_<BVParametersTable>("BVParametersTable", doc_BVParametersTable)
        .def("none", singleton_none, doc_BVParametersTable_none)
        .staticmethod("none")
        .def("getAtomValence", &BVParametersTable::getAtomValence,
                arg("smbl"),
                doc_BVParametersTable_getAtomValence)
        .def("setAtomValence", &BVParametersTable::setAtomValence,
                (arg("smbl"), arg("value")),
                doc_BVParametersTable_setAtomValence)
        .def("resetAtomValences", &BVParametersTable::resetAtomValences,
                doc_BVParametersTable_resetAtomValences)
        .def("lookup", bptb_bvparam_1(&BVParametersTable::lookup),
                arg("bvparam"), doc_BVParametersTable_lookup1,
                return_value_policy<copy_const_reference>())
        .def("lookup", bptb_bvparam_2(&BVParametersTable::lookup),
                (arg("smbl0"), arg("smbl1")),
                doc_BVParametersTable_lookup2,
                return_value_policy<copy_const_reference>())
        .def("lookup", bptb_bvparam_4(&BVParametersTable::lookup),
                (arg("atom0"), arg("valence0"), arg("atom1"), arg("valence1")),
                doc_BVParametersTable_lookup4,
                return_value_policy<copy_const_reference>())
        .def("setCustom", bptb_void_1(&BVParametersTable::setCustom),
                arg("bvparm"), doc_BVParametersTable_setCustom1)
        .def("setCustom", (void(BVParametersTable::*)(const string&, int,
                    const string&, int, double, double, string)) NULL,
                setcustom6((arg("atom0"), arg("valence0"), arg("atom1"), arg("valence1"),
                     arg("Ro"), arg("B"), arg("ref_id")=""),
                    doc_BVParametersTable_setCustom6))
        .def("resetCustom", bptb_void_1(&BVParametersTable::resetCustom),
                doc_BVParametersTable_resetCustom1)
        .def("resetCustom", bptb_void_4(&BVParametersTable::resetCustom),
                (arg("atom0"), arg("valence0"), arg("atom1"), arg("valence1")),
                doc_BVParametersTable_resetCustom4)
        .def("resetAll", &BVParametersTable::resetAll,
                doc_BVParametersTable_resetAll)
        .def("getAll", getAll_asset<BVParametersTable>,
                doc_BVParametersTable_getAll)
        .def_pickle(SerializationPickleSuite<BVParametersTable>())
        ;

    register_ptr_to_python<BVParametersTablePtr>();
}

}   // namespace srrealmodule

// End of file
