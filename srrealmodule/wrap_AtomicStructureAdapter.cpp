/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   (c) 2013 Brookhaven Science Associates,
*                   Brookhaven National Laboratory.
*                   All rights reserved.
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

#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <diffpy/srreal/AtomicStructureAdapter.hpp>
#include <diffpy/srreal/PeriodicStructureAdapter.hpp>
#include <diffpy/srreal/CrystalStructureAdapter.hpp>
#include <diffpy/srreal/PairQuantity.hpp>
#include <diffpy/srreal/StructureDifference.hpp>
#include <diffpy/serialization.ipp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"
#include "srreal_validators.hpp"

namespace srrealmodule {

// declarations
void sync_StructureDifference(boost::python::object obj);

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

const char* doc_AtomicStructureAdapter = "\
Structure adapter for a non-periodic group of atoms.\n\
\n\
This class supports indexing and iteration similar to Python list.\n\
";

const char* doc_AtomicStructureAdapter_clone = "\
Return a deep copy of this AtomicStructureAdapter.\n\
\n\
This method can be overloaded in a derived class.\n\
";

const char* doc_AtomicStructureAdapter_insert = "\
Insert atom at a specified site index in this adapter.\n\
\n\
index    -- integer index for adding the new atom.  May be negative\n\
            as per Python indexing conventions.\n\
atom     -- Atom object to be inserted.\n\
\n\
No return value.\n\
";

const char* doc_AtomicStructureAdapter_append = "\
Add atom to the end of this adapter.\n\
\n\
atom     -- Atom object to be inserted.\n\
\n\
No return value.\n\
";

const char* doc_AtomicStructureAdapter_pop = "\
Remove and return Atom at the specified site index.\n\
\n\
index    -- integer index of the removed atom.  May be negative\n\
            as per Python indexing conventions.\n\
\n\
Return the removed Atom.\n\
";

const char* doc_AtomicStructureAdapter_clear = "\
Remove all atoms from the structure.\n\
";

const char* doc_AtomicStructureAdapter_reserve = "\
Reserve memory for a specified number of atoms.\n\
Although not required, calling this method can save memory and\n\
avoid reallocation of Atom instances.\n\
\n\
sz   -- expected number of atoms in this adapter.\n\
\n\
No return value.\n\
";

// class PeriodicStructureAdapter

const char* doc_PeriodicStructureAdapter = "\
Group of atoms with periodic boundary conditions, but no\n\
space group symmetry.\n\
";

const char* doc_PeriodicStructureAdapter_clone = "\
Return a deep copy of this PeriodicStructureAdapter.\n\
\n\
This method can be overloaded in a derived class.\n\
";

const char* doc_PeriodicStructureAdapter_getLatPar = "\
Get lattice parameters for the periodic unit cell.\n\
\n\
Return a tuple of (a, b, c, alpha, beta, gamma), where cell\n\
angles are in degrees.\n\
";

const char* doc_PeriodicStructureAdapter_setLatPar = "\
Set lattice parameters of the periodic unit cell.\n\
\n\
a, b, c  -- cell lengths in Angstroms.\n\
alphadeg, betadeg, gammadeg  -- cell angles in degrees.\n\
\n\
No return value.\n\
";

const char* doc_PeriodicStructureAdapter_toCartesian = "\
Convert atom position and displacement parameters to Cartesian coordinates.\n\
\n\
atom -- Atom object to be converted to Cartesian coordinates.\n\
\n\
No return value.  This updates the xyz_cartn and uij_cartn\n\
attributes of the passed atom inplace.\n\
";

const char* doc_PeriodicStructureAdapter_toFractional = "\
Convert atom position and displacement parameters to fractional coordinates.\n\
\n\
atom -- Atom object to be converted to fractional coordinates.\n\
\n\
No return value.  This updates the xyz_cartn and uij_cartn\n\
attributes of the passed atom inplace.\n\
";

// class CrystalStructureAdapter

const char* doc_CrystalStructureAdapter = "\
Structure with asymmetric unit cell and a list of space group symmetry\n\
operations.  The indexed atoms relate to the asymmetric unit cell.\n\
";

const char* doc_CrystalStructureAdapter_clone = "\
Return a deep copy of this CrystalStructureAdapter.\n\
\n\
This method can be overloaded in a derived class.\n\
";

const char* doc_CrystalStructureAdapter_symmetryprecision = "\n\
Distance threshold for assuming symmetry generated sites equal.\n\
";

const char* doc_CrystalStructureAdapter_countSymOps = "\
Return number of space group symmetry operations stored in the adapter.\n\
";

const char* doc_CrystalStructureAdapter_clearSymOps = "\
Clear all symmetry operations from the adapter.\n\
";

const char* doc_CrystalStructureAdapter_addSymOp = "\
Add one space group symmetry operation to the adapter.\n\
\n\
R    -- rotation matrix for the symmetry operation.\n\
t    -- translation vector in the symmetry operation.\n\
\n\
No return value.  R and t are in fractional coordinates.\n\
";

const char* doc_CrystalStructureAdapter_getSymOp = "\
Get rotation and translation for the specified symmetry operation.\n\
\n\
index    -- zero based index of a previously defined symmetry operation.\n\
\n\
Return a tuple (R, t) of symmetry rotation matrix and translation\n\
vector in fractional coordinate system.\n\
";

const char* doc_CrystalStructureAdapter_getEquivalentAtoms = "\
Return symmetry equivalent atoms for a site in the asymmetric unit.\n\
\n\
index    -- zero-based index of an atom in the asymmetric unit.\n\
\n\
Return all symmetry equivalent atoms in the periodic unit cell\n\
as an AtomicStructureAdapter type.  Atom positions and displacement\n\
parameters are in Cartesian coordinates.\n\
";

const char* doc_CrystalStructureAdapter_expandLatticeAtom = "\
Perform symmetry expansion for an Atom in fractional coordinates.\n\
\n\
atom -- Atom object with xyz_cartn and uij_cartn referring to position\n\
        and displacement parameters in fractional coordinates\n\
\n\
Return all symmetry equivalent atoms in the periodic unit cell\n\
as an AtomicStructureAdapter type.  Positions and displacement\n\
parameters are all in fractional coordinates.\n\
";

const char* doc_CrystalStructureAdapter_updateSymmetryPositions = "\
Force update of symmetry equivalent positions for the asymmetric unit.\n\
\n\
The getEquivalentAtoms function calls this automatically if internal\n\
symmetry operations changed or if size of the asymmetric unit changed.\n\
An explicit call may be necessary for a more subtle changes such as\n\
moving one asymmetric site.  The updateSymmetryPositions is always\n\
implicitly called from createBondGenerator.\n\
";

// wrappers ------------------------------------------------------------------

// Wrapper helpers for the class Atom

object get_xyz_cartn(Atom& a)
{
    return viewAsNumPyArray(a.xyz_cartn);
}

void set_xyz_cartn(Atom& a, object value)
{
    assignR3Vector(a.xyz_cartn, value);
}


object get_uij_cartn(Atom& a)
{
    return viewAsNumPyArray(a.uij_cartn);
}

void set_uij_cartn(Atom& a, object& value)
{
    assignR3Matrix(a.uij_cartn, value);
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

// template wrapper class for overloading of clone and _customPQConfig

template <class T>
class MakeWrapper : public T, public wrapper_srreal<T>
{
    public:

        StructureAdapterPtr clone() const
        {
            override f = this->get_override("clone");
            if (f)  return f();
            else  return this->default_clone();
        }

        StructureAdapterPtr default_clone() const
        {
            return this->T::clone();
        }


        void customPQConfig(PairQuantity* pq) const
        {
            override f = this->get_override("_customPQConfig");
            if (f)  f(ptr(pq));
            else    this->default_customPQConfig(pq);
        }

        void default_customPQConfig(PairQuantity* pq) const
        {
            this->T::customPQConfig(pq);
        }


        StructureDifference diff(StructureAdapterConstPtr other) const
        {
            override f = this->get_override("diff");
            if (f)
            {
                python::object sdobj = f(other);
                sync_StructureDifference(sdobj);
                StructureDifference& sd =
                    python::extract<StructureDifference&>(sdobj);
                return sd;
            }
            return this->default_diff(other);
        }

        StructureDifference default_diff(StructureAdapterConstPtr other) const
        {
            return this->T::diff(other);
        }

    private:

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<T>(*this);
        }

};  // class MakeWrapper

// Wrapper helpers for class AtomicStructureAdapter

typedef MakeWrapper<AtomicStructureAdapter> AtomicStructureAdapterWrap;
typedef boost::shared_ptr<AtomicStructureAdapterWrap> AtomicStructureAdapterWrapPtr;

class atomadapter_indexing : public vector_indexing_suite<
                         AtomicStructureAdapter, false, atomadapter_indexing>
{
    public:

        typedef AtomicStructureAdapter Container;

        static object
        get_slice(Container& container, index_type from, index_type to)
        {
            // make sure slice is of a correct type and has a copy
            // of any additional structure data.
            StructureAdapterPtr rv = container.clone();
            AtomicStructureAdapterPtr rva;
            rva = boost::static_pointer_cast<AtomicStructureAdapter>(rv);
            // handle index ranges for a valid and empty slice
            if (from <= to)
            {
                rva->erase(rva->begin() + to, rva->end());
                rva->erase(rva->begin(), rva->begin() + from);
            }
            else  rva->clear();
            // save memory by making a new copy for short slices
            const bool longslice = ((to - from) > rva->countSites() / 2);
            object pyrv(longslice ? rv : rv->clone());
            return pyrv;
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

typedef MakeWrapper<PeriodicStructureAdapter> PeriodicStructureAdapterWrap;
typedef boost::shared_ptr<PeriodicStructureAdapterWrap> PeriodicStructureAdapterWrapPtr;

python::tuple periodicadapter_getlatpar(const PeriodicStructureAdapter& adpt)
{
    const Lattice& L = adpt.getLattice();
    python::tuple rv = python::make_tuple(
            L.a(), L.b(), L.c(), L.alpha(), L.beta(), L.gamma());
    return rv;
}

// Wrapper helpers for class CrystalStructureAdapter

typedef MakeWrapper<CrystalStructureAdapter> CrystalStructureAdapterWrap;
typedef boost::shared_ptr<CrystalStructureAdapterWrap> CrystalStructureAdapterWrapPtr;

double
crystaladapter_getsymmetryprecision(const CrystalStructureAdapter& adpt)
{
    return adpt.getSymmetryPrecision();
}


void crystaladapter_addsymop(CrystalStructureAdapter& adpt,
        python::object R, python::object t)
{
    static SymOpRotTrans op;
    assignR3Matrix(op.R, R);
    assignR3Vector(op.t, t);
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
    AtomicStructureAdapterPtr rv(new AtomicStructureAdapter);
    rv->assign(av.begin(), av.end());
    return rv;
}


AtomicStructureAdapterPtr crystaladapter_expandlatticeatom(
        const CrystalStructureAdapter& adpt, const Atom& a)
{
    CrystalStructureAdapter::AtomVector av = adpt.expandLatticeAtom(a);
    AtomicStructureAdapterPtr rv(new AtomicStructureAdapter);
    rv->assign(av.begin(), av.end());
    return rv;
}

}   // namespace nswrap_AtomicStructureAdapter

// declare shared docstrings from wrap_StructureAdapter.cpp

extern const char* doc_StructureAdapter___init__fromstring;
extern const char* doc_StructureAdapter__customPQConfig;
extern const char* doc_StructureAdapter_diff;

// Wrapper definitions -------------------------------------------------------

void wrap_AtomicStructureAdapter()
{
    namespace bp = boost::python;
    using namespace nswrap_AtomicStructureAdapter;
    using diffpy::srreal::hash_value;

    // class Atom
    class_<Atom> atom_class("Atom", doc_Atom);
    // first define copy constructor and property helper methods
    atom_class
        .def(init<const Atom&>(bp::arg("atom"), doc_Atom_init_copy))
        .def(self == self)
        .def(self != self)
        .def(self < self)
        .def(self > self)
        .def(self <= self)
        .def(self >= self)
        .def("__hash__", hash_value)
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
        .def_pickle(SerializationPickleSuite<Atom,DICT_IGNORE>())
        ;

    // class AtomicStructureAdapter
    class_<AtomicStructureAdapterWrap, bases<StructureAdapter>,
        noncopyable, AtomicStructureAdapterWrapPtr>(
            "AtomicStructureAdapter", doc_AtomicStructureAdapter)
        .def("__init__", StructureAdapter_constructor(),
                doc_StructureAdapter___init__fromstring)
        .def(atomadapter_indexing())
        .def(self == self)
        .def(self != self)
        .def("clone",
                &AtomicStructureAdapter::clone,
                &AtomicStructureAdapterWrap::default_clone,
                doc_AtomicStructureAdapter_clone)
        .def("_customPQConfig",
                &AtomicStructureAdapter::customPQConfig,
                &AtomicStructureAdapterWrap::default_customPQConfig,
                python::arg("pqobj"),
                doc_StructureAdapter__customPQConfig)
        .def("diff",
                &AtomicStructureAdapter::diff,
                &AtomicStructureAdapterWrap::default_diff,
                python::arg("other"),
                doc_StructureAdapter_diff)
        .def("insert", atomadapter_insert,
                (bp::arg("index"), bp::arg("atom")),
                doc_AtomicStructureAdapter_insert)
        .def("append", &AtomicStructureAdapter::append,
                doc_AtomicStructureAdapter_append)
        .def("pop", atomadapter_pop,
                bp::arg("index"), doc_AtomicStructureAdapter_pop)
        .def("clear", &AtomicStructureAdapter::clear,
                doc_AtomicStructureAdapter_clear)
        .def("reserve", atomadapter_reserve,
                bp::arg("sz"), doc_AtomicStructureAdapter_reserve)
        .def_pickle(StructureAdapterPickleSuite<AtomicStructureAdapterWrap>())
        ;

    // class PeriodicStructureAdapter
    class_<PeriodicStructureAdapterWrap, bases<AtomicStructureAdapter>,
        noncopyable, PeriodicStructureAdapterWrapPtr>(
            "PeriodicStructureAdapter", doc_PeriodicStructureAdapter)
        .def("__init__", StructureAdapter_constructor(),
                doc_StructureAdapter___init__fromstring)
        .def(self == self)
        .def(self != self)
        .def("clone",
                &PeriodicStructureAdapter::clone,
                &PeriodicStructureAdapterWrap::default_clone,
                doc_PeriodicStructureAdapter_clone)
        .def("_customPQConfig",
                &PeriodicStructureAdapter::customPQConfig,
                &PeriodicStructureAdapterWrap::default_customPQConfig,
                python::arg("pqobj"),
                doc_StructureAdapter__customPQConfig)
        .def("diff",
                &PeriodicStructureAdapter::diff,
                &PeriodicStructureAdapterWrap::default_diff,
                python::arg("other"),
                doc_StructureAdapter_diff)
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
        .def_pickle(StructureAdapterPickleSuite<PeriodicStructureAdapterWrap>())
        ;

    // class CrystalStructureAdapter
    class_<CrystalStructureAdapterWrap, bases<PeriodicStructureAdapter>,
        noncopyable, CrystalStructureAdapterWrapPtr>(
            "CrystalStructureAdapter", doc_CrystalStructureAdapter)
        .def("__init__", StructureAdapter_constructor(),
                doc_StructureAdapter___init__fromstring)
        .def(self == self)
        .def(self != self)
        .def("clone",
                &CrystalStructureAdapter::clone,
                &CrystalStructureAdapterWrap::default_clone,
                doc_CrystalStructureAdapter_clone)
        .def("_customPQConfig",
                &CrystalStructureAdapter::customPQConfig,
                &CrystalStructureAdapterWrap::default_customPQConfig,
                python::arg("pqobj"),
                doc_StructureAdapter__customPQConfig)
        .def("diff",
                &CrystalStructureAdapter::diff,
                &CrystalStructureAdapterWrap::default_diff,
                python::arg("other"),
                doc_StructureAdapter_diff)
        .add_property("symmetryprecision",
            crystaladapter_getsymmetryprecision,
            &CrystalStructureAdapter::setSymmetryPrecision,
            doc_CrystalStructureAdapter_symmetryprecision)
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
        .def_pickle(StructureAdapterPickleSuite<CrystalStructureAdapterWrap>())
        ;

}

}   // namespace srrealmodule

using srrealmodule::nswrap_AtomicStructureAdapter::AtomicStructureAdapterWrap;
BOOST_CLASS_EXPORT(AtomicStructureAdapterWrap)

using srrealmodule::nswrap_AtomicStructureAdapter::PeriodicStructureAdapterWrap;
BOOST_CLASS_EXPORT(PeriodicStructureAdapterWrap)

using srrealmodule::nswrap_AtomicStructureAdapter::CrystalStructureAdapterWrap;
BOOST_CLASS_EXPORT(CrystalStructureAdapterWrap)

// End of file
