/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   Pavol Juhas
*                   (c) 2013 Brookhaven National Laboratory,
*                   Upton, New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Bindings to the Atom, AtomicStructureAdapter, PeriodicStructureAdapter
* and CrystalStructureAdapter classes.
* only for accessing the C++ created StructureAdapter instances and there
* is no support for method overrides from Python.
*
*****************************************************************************/

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/slice.hpp>

#include <diffpy/srreal/AtomicStructureAdapter.hpp>
#include <diffpy/srreal/PeriodicStructureAdapter.hpp>
#include <diffpy/srreal/CrystalStructureAdapter.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"
#include "srreal_validators.hpp"

namespace srrealmodule {
namespace nswrap_AtomicStructureAdapter {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

// class Atom

const char* doc_Atom = "\
Data for one atom site in AtomicStructureAdapter.\n\
\n\
Instance data:\n\
\n\
atomtype     -- string symbol for element, ion or isotope\n\
xyz_cartn    -- Cartesian coordinates viewed as NumPy array\n\
occupancy    -- fractional occupancy of this atom site\n\
anisotropy   -- boolean flag for anisotropic displacements at this site\n\
uij_cartn    -- matrix of anisotropic displacements parameters viewed\n\
                as NumPy array\n\
\n\
Note xyz_cartn and uij_cartn are NumPy arrays with a direct\n\
view to the data in C++ class.  Do not resize or reshape.\n\
";

const char* doc_Atom_init_copy = "\
Make a deep copy of an existing Atom.\n\
";
const char* doc_Atom_xic = "Vector element in xyz_cartn";
const char* doc_Atom_uijc = "Matrix element in uij_cartn";

// class AtomicStructureAdapter

const char* doc_AtomicStructureAdapter = "";
const char* doc_AtomicStructureAdapter_init_copy = "FIXME";
const char* doc_AtomicStructureAdapter_insert = "FIXME";
const char* doc_AtomicStructureAdapter_append = "FIXME";
const char* doc_AtomicStructureAdapter_pop = "FIXME";
const char* doc_AtomicStructureAdapter_reserve = "FIXME";

// class PeriodicStructureAdapter

const char* doc_PeriodicStructureAdapter = "FIXME";
const char* doc_PeriodicStructureAdapter_init_copy = "FIXME";
const char* doc_PeriodicStructureAdapter_getLatPar = "FIXME";
const char* doc_PeriodicStructureAdapter_setLatPar = "FIXME";
const char* doc_PeriodicStructureAdapter_toCartesian = "FIXME";
const char* doc_PeriodicStructureAdapter_toFractional = "FIXME";

// class CrystalStructureAdapter

const char* doc_CrystalStructureAdapter = "FIXME";
const char* doc_CrystalStructureAdapter_init_copy = "FIXME";
const char* doc_CrystalStructureAdapter_countSymOps = "FIXME";
const char* doc_CrystalStructureAdapter_clearSymOps = "FIXME";
const char* doc_CrystalStructureAdapter_addSymOp = "FIXME";
const char* doc_CrystalStructureAdapter_getSymOp = "FIXME";
const char* doc_CrystalStructureAdapter_getEquivalentAtoms = "FIXME";
const char* doc_CrystalStructureAdapter_expandLatticeAtom = "FIXME";
const char* doc_CrystalStructureAdapter_updateSymmetryPositions = "FIXME";

// wrappers ------------------------------------------------------------------

// Wrapper helpers for the class Atom

object get_xyz_cartn(Atom& a)
{
    return viewAsNumPyArray(a.xyz_cartn);
}

void set_xyz_cartn(Atom& a, object value)
{
    object xyzc = get_xyz_cartn(a);
    xyzc[slice()] = value;
}


object get_uij_cartn(Atom& a)
{
    return viewAsNumPyArray(a.uij_cartn);
}

void set_uij_cartn(Atom& a, object value)
{
    object uijc = get_uij_cartn(a);
    uijc[slice()] = value;
}


double get_xc(const Atom& a)  { return a.xyz_cartn[0]; }
void set_xc(Atom& a, double value)  { a.xyz_cartn[0] = value; }
double get_yc(const Atom& a)  { return a.xyz_cartn[1]; }
void set_yc(Atom& a, double value)  { a.xyz_cartn[1] = value; }
double get_zc(const Atom& a)  { return a.xyz_cartn[2]; }
void set_zc(Atom& a, double value)  { a.xyz_cartn[2] = value; }

double get_uc11(const Atom& a)  { return a.uij_cartn(0, 0); }
void set_uc11(Atom& a, double value)  { a.uij_cartn(0, 0) = value; }
double get_uc22(const Atom& a)  { return a.uij_cartn(1, 1); }
void set_uc22(Atom& a, double value)  { a.uij_cartn(1, 1) = value; }
double get_uc33(const Atom& a)  { return a.uij_cartn(2, 2); }
void set_uc33(Atom& a, double value)  { a.uij_cartn(2, 2) = value; }
double get_uc12(const Atom& a)  { return a.uij_cartn(0, 1); }
void set_uc12(Atom& a, double value) {
    a.uij_cartn(0, 1) = a.uij_cartn(1, 0) = value; }
double get_uc13(const Atom& a)  { return a.uij_cartn(0, 2); }
void set_uc13(Atom& a, double value) {
    a.uij_cartn(0, 2) = a.uij_cartn(2, 0) = value; }
double get_uc23(const Atom& a)  { return a.uij_cartn(1, 2); }
void set_uc23(Atom& a, double value) {
    a.uij_cartn(1, 2) = a.uij_cartn(2, 1) = value; }

// Wrapper helpers for class AtomicStructureAdapter

AtomicStructureAdapterPtr atomadapter_create()
{
    return AtomicStructureAdapterPtr(new AtomicStructureAdapter);
}


AtomicStructureAdapterPtr atomadapter_copy(const AtomicStructureAdapter& adpt)
{
    return AtomicStructureAdapterPtr(new AtomicStructureAdapter(adpt));
}


class atomadapter_indexing : public vector_indexing_suite<
                         AtomicStructureAdapter, false, atomadapter_indexing>
{
    public:

        typedef AtomicStructureAdapter Container;

        static object
        get_slice(Container& container, index_type from, index_type to)
        {
            Container rv;
            if (from <= to)
            {
                rv.assign(container.begin() + from, container.begin() + to);
            }
            return object(rv);
        }


        static void
        append(Container& container, data_type const& v)
        {
            container.append(v);
        }

};


void atomadapter_insert(AtomicStructureAdapter& adpt, const Atom& a, int idx)
{
    ensure_index_bounds(idx, -int(adpt.size()), adpt.size() + 1);
    int idx1 = (idx >= 0) ? idx : int(adpt.size()) - idx;
    adpt.insert(idx1, a);
}


Atom atomadapter_pop(AtomicStructureAdapter& adpt, int idx)
{
    ensure_index_bounds(idx, -int(adpt.size()), adpt.size());
    int idx1 = (idx >= 0) ? idx : int(adpt.size()) - idx;
    Atom a = adpt[idx1];
    adpt.erase(idx1);
    return a;
}


void atomadapter_reserve(AtomicStructureAdapter& adpt, int sz)
{
    ensure_non_negative(sz);
    adpt.reserve(sz);
}

// Wrapper helpers for class PeriodicStructureAdapter

PeriodicStructureAdapterPtr periodicadapter_create()
{
    return PeriodicStructureAdapterPtr(new PeriodicStructureAdapter);
}


PeriodicStructureAdapterPtr
periodicadapter_copy(const PeriodicStructureAdapter& adpt)
{
    return PeriodicStructureAdapterPtr(new PeriodicStructureAdapter(adpt));
}


python::tuple periodicadapter_getlatpar(const PeriodicStructureAdapter& adpt)
{
    const Lattice& L = adpt.getLattice();
    python::tuple rv = python::make_tuple(
            L.a(), L.b(), L.c(), L.alpha(), L.beta(), L.gamma());
    return rv;
}

// Wrapper helpers for class CrystalStructureAdapter


CrystalStructureAdapterPtr crystaladapter_create()
{
    return CrystalStructureAdapterPtr(new CrystalStructureAdapter);
}


CrystalStructureAdapterPtr
crystaladapter_copy(const CrystalStructureAdapter& adpt)
{
    return CrystalStructureAdapterPtr(new CrystalStructureAdapter(adpt));
}


void crystaladapter_addsymop(CrystalStructureAdapter& adpt,
        python::object R, python::object t)
{
    static SymOpRotTrans op;
    object Rview = viewAsNumPyArray(op.R);
    Rview[slice()] = R;
    object tview = viewAsNumPyArray(op.t);
    tview[slice()] = t;
    adpt.addSymOp(op);
}


python::tuple
crystaladapter_getsymop(const CrystalStructureAdapter& adpt, int idx)
{
    ensure_index_bounds(idx, 0, adpt.countSymOps());
    const SymOpRotTrans& op = adpt.getSymOp(idx);
    python::tuple rv = python::make_tuple(
            convertToNumPyArray(op.R), convertToNumPyArray(op.t));
    return rv;
}


AtomicStructureAdapterPtr
crystaladapter_getequivalentatoms(const CrystalStructureAdapter& adpt, int idx)
{
    ensure_index_bounds(idx, 0, adpt.countSymOps());
    const CrystalStructureAdapter::AtomVector& av =
        adpt.getEquivalentAtoms(idx);
    AtomicStructureAdapterPtr rv = atomadapter_create();
    rv->assign(av.begin(), av.end());
    return rv;
}


AtomicStructureAdapterPtr crystaladapter_expandlatticeatom(
        const CrystalStructureAdapter& adpt, const Atom& a)
{
    CrystalStructureAdapter::AtomVector av = adpt.expandLatticeAtom(a);
    AtomicStructureAdapterPtr rv = atomadapter_create();
    rv->assign(av.begin(), av.end());
    return rv;
}

}   // namespace nswrap_AtomicStructureAdapter

// Wrapper definitions -------------------------------------------------------

void wrap_AtomicStructureAdapter()
{
    namespace bp = boost::python;
    using namespace nswrap_AtomicStructureAdapter;

    // class Atom
    class_<Atom> atom_class("Atom", doc_Atom);
    // first define copy constructor and property helper methods
    atom_class
        .def(init<const Atom&>(bp::arg("atom"), doc_Atom_init_copy))
        .def(self == self)
        .def(self != self)
        .def("_get_xyz_cartn",
                get_xyz_cartn,
                with_custodian_and_ward_postcall<0,1>())
        .def("_get_uij_cartn",
                get_uij_cartn,
                with_custodian_and_ward_postcall<0,1>())
        ;
    // now we can finalize the Atom class interface
    atom_class
        .def_readwrite("atomtype", &Atom::atomtype)
        .add_property("xyz_cartn",
                atom_class.attr("_get_xyz_cartn"),
                set_xyz_cartn)
        .add_property("xc", get_xc, set_xc, doc_Atom_xic)
        .add_property("yc", get_yc, set_yc, doc_Atom_xic)
        .add_property("zc", get_zc, set_zc, doc_Atom_xic)
        .def_readwrite("occupancy", &Atom::occupancy)
        .def_readwrite("anisotropy", &Atom::anisotropy)
        .add_property("uij_cartn",
                atom_class.attr("_get_uij_cartn"),
                set_uij_cartn)
        .add_property("uc11", get_uc11, set_uc11, doc_Atom_uijc)
        .add_property("uc22", get_uc22, set_uc22, doc_Atom_uijc)
        .add_property("uc33", get_uc33, set_uc33, doc_Atom_uijc)
        .add_property("uc12", get_uc12, set_uc12, doc_Atom_uijc)
        .add_property("uc13", get_uc13, set_uc13, doc_Atom_uijc)
        .add_property("uc23", get_uc23, set_uc23, doc_Atom_uijc)
        ;

    // class AtomicStructureAdapter
    class_<AtomicStructureAdapter, bases<StructureAdapter> >(
            "AtomicStructureAdapter", doc_AtomicStructureAdapter)
        // object from default constructor would throw tr1::bad_weak_ptr
        // when calling shared_from_this, but it seems to work well
        // if constructed with a factory function.
        .def("__init__", make_constructor(atomadapter_create))
        .def("__init__", make_constructor(atomadapter_copy),
                doc_AtomicStructureAdapter_init_copy)
        .def(atomadapter_indexing())
        .def(self == self)
        .def(self != self)
        .def("insert", atomadapter_insert,
                (bp::arg("index"), bp::arg("atom")),
                doc_AtomicStructureAdapter_insert)
        .def("append", &AtomicStructureAdapter::append,
                doc_AtomicStructureAdapter_append)
        .def("pop", atomadapter_pop,
                bp::arg("index"), doc_AtomicStructureAdapter_pop)
        .def("reserve", atomadapter_reserve,
                bp::arg("sz"), doc_AtomicStructureAdapter_reserve)
        ;

    // class PeriodicStructureAdapter
    class_<PeriodicStructureAdapter, bases<AtomicStructureAdapter> >(
            "PeriodicStructureAdapter", doc_PeriodicStructureAdapter)
        .def("__init__", make_constructor(periodicadapter_create))
        .def("__init__", make_constructor(periodicadapter_copy),
                doc_PeriodicStructureAdapter_init_copy)
        .def(self == self)
        .def(self != self)
        .def("getLatPar", periodicadapter_getlatpar,
                doc_PeriodicStructureAdapter_getLatPar)
        .def("setLatPar", &PeriodicStructureAdapter::setLatPar,
                (bp::arg("a"), bp::arg("b"), bp::arg("c"),
                 bp::arg("alphadeg"), bp::arg("betadeg"), bp::arg("gammadeg")),
                doc_PeriodicStructureAdapter_setLatPar)
        .def("toCartesian", &PeriodicStructureAdapter::toCartesian,
                bp::arg("atom"), doc_PeriodicStructureAdapter_toCartesian)
        .def("toFractional", &PeriodicStructureAdapter::toFractional,
                bp::arg("atom"), doc_PeriodicStructureAdapter_toFractional)
        ;

    // class CrystalStructureAdapter
    class_<CrystalStructureAdapter, bases<PeriodicStructureAdapter> >(
            "CrystalStructureAdapter", doc_CrystalStructureAdapter)
        .def("__init__", make_constructor(crystaladapter_create))
        .def("__init__", make_constructor(crystaladapter_copy),
                doc_CrystalStructureAdapter_init_copy)
        .def(self == self)
        .def(self != self)
        .def("countSymOps", &CrystalStructureAdapter::countSymOps,
                doc_CrystalStructureAdapter_countSymOps)
        .def("clearSymOps", &CrystalStructureAdapter::clearSymOps,
                doc_CrystalStructureAdapter_clearSymOps)
        .def("addSymOp", crystaladapter_addsymop,
                (bp::arg("R"), bp::arg("t")),
                doc_CrystalStructureAdapter_addSymOp)
        .def("getSymOp", crystaladapter_getsymop, bp::arg("index"),
                doc_CrystalStructureAdapter_getSymOp)
        .def("getEquivalentAtoms",
                crystaladapter_getequivalentatoms, bp::arg("index"),
                doc_CrystalStructureAdapter_getEquivalentAtoms)
        .def("expandLatticeAtom",
                crystaladapter_expandlatticeatom, bp::arg("atom"),
                doc_CrystalStructureAdapter_expandLatticeAtom)
        .def("updateSymmetryPositions",
                &CrystalStructureAdapter::updateSymmetryPositions,
                doc_CrystalStructureAdapter_updateSymmetryPositions)
        ;

}

}   // namespace srrealmodule

// End of file
