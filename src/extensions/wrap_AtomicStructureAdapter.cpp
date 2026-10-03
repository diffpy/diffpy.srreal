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

#include <nanobind/nanobind.h>
#include "srreal_trampoline.hpp"
#include <nanobind/operators.h>

#include <diffpy/srreal/AtomicStructureAdapter.hpp>
#include <diffpy/srreal/PeriodicStructureAdapter.hpp>
#include <diffpy/srreal/CrystalStructureAdapter.hpp>
#include <diffpy/srreal/PairQuantity.hpp>
#include <diffpy/srreal/StructureDifference.hpp>
#include <diffpy/serialization.ipp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"
#include "srreal_validators.hpp"
#include "srreal_atom.hpp"

namespace srrealmodule {

// declarations
void sync_StructureDifference(nb::object obj);

namespace nswrap_AtomicStructureAdapter {

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
const char* doc_Atom_xic = "Vector element in xyz_cartn.";
const char* doc_Atom_occ = "Fractional occupancy of this atom.";
const char* doc_Atom_anisotropy =
    "Boolean flag for anisotropic displacements.";
const char* doc_Atom_uijc = "Matrix element in uij_cartn.";

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
Return a list of all symmetry equivalent atoms in the periodic unit cell\n\
Atom positions and displacement parameters are in Cartesian coordinates.\n\
";

const char* doc_CrystalStructureAdapter_expandLatticeAtom = "\
Perform symmetry expansion for an Atom in fractional coordinates.\n\
\n\
atom -- Atom object with xyz_cartn and uij_cartn referring to position\n\
        and displacement parameters in fractional coordinates\n\
\n\
Return a list of all symmetry equivalent atoms in the periodic unit cell\n\
Positions and displacement parameters are in fractional coordinates.\n\
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

nb::object get_xyz_cartn(PythonAtom& a)
{
    return pin_atom_view(a, viewAsNumPyArray(a.value().xyz_cartn));
}

void set_xyz_cartn(PythonAtom& a, nb::object value)
{
    assignR3Vector(a.value().xyz_cartn, value);
}


nb::object get_uij_cartn(PythonAtom& a)
{
    return pin_atom_view(a, viewAsNumPyArray(a.value().uij_cartn));
}

void set_uij_cartn(PythonAtom& a, nb::object& value)
{
    assignR3Matrix(a.value().uij_cartn, value);
}


template <const int i>
double get_xyz(const PythonAtom& a)
{
    return a.value().xyz_cartn[i];
}

template <const int i>
void set_xyz(PythonAtom& a, nb::object value)
{
    a.value().xyz_cartn[i] = extractdouble(value);
}


double get_occ(const PythonAtom& a)
{
    return a.value().occupancy;
}

void set_occ(PythonAtom& a, nb::object value)
{
    a.value().occupancy = extractdouble(value);
}


bool get_anisotropy(const PythonAtom& a)
{
    return a.value().anisotropy;
}

void set_anisotropy(PythonAtom& a, nb::object value)
{
    int truth = PyObject_IsTrue(value.ptr());
    if (truth < 0)
        nb::raise_python_error();
    a.value().anisotropy = truth != 0;
}


template <const int i, const int j>
double get_uc(const PythonAtom& a)
{
    assert(i <= j);
    return a.value().uij_cartn(i, j);
}

template <const int i, const int j>
void set_uc(PythonAtom& a, nb::object value)
{
    assert(i <= j);
    a.value().uij_cartn(i, j) = extractdouble(value);
    if (i != j)  a.value().uij_cartn(j, i) = a.value().uij_cartn(i, j);
}

class AtomPickleSuite : public SerializationPickleSuite<Atom>
{
    public:
        template <class C>
        static void bind(C& cls)
        {
            cls.def("__getstate__", [](nb::object obj) {
                ensure_dict_is_managed_or_empty(obj, state_manages_dict(obj, false));
                return nb::make_tuple(serialization_tobytes(nb::cast<PythonAtom&>(obj).value()));
            })
            .def("__setstate__", [](nb::object obj, nb::tuple state) {
                ensure_tuple_length(state, 1);
                PythonAtom& atom = nb::cast<PythonAtom&>(obj);
                diffpy::serialization_fromstring(atom.value(), bytes_to_string(state[0]));
            })
            .def("__reduce__", reduce);
            cls.attr("__getstate_manages_dict__") = nb::none();
        }
};

// template wrapper class for overloading of clone and _customPQConfig

template <class T>
class MakeWrapper :
    public T,
    public PythonTrampolineTag
{
    public:

        NB_TRAMPOLINE(T);

        StructureAdapterPtr clone() const override
        {
            NB_OVERRIDE(clone);
        }

        StructureAdapterPtr default_clone() const
        {
            return this->T::clone();
        }


        void customPQConfig(PairQuantity* pq) const override
        {
            NB_OVERRIDE_NAME("_customPQConfig", customPQConfig, pq);
        }

        void default_customPQConfig(PairQuantity* pq) const
        {
            this->T::customPQConfig(pq);
        }


        StructureDifference diff(StructureAdapterConstPtr other) const override
        {
            nb::gil_scoped_acquire gil;
            OverrideTicket ticket(nb_trampoline, "diff", false);
            if (ticket.key.is_valid())
            {
                nb::object sdobj = nb_trampoline.base().attr(ticket.key)(other);
                sync_StructureDifference(sdobj);
                StructureDifference& sd =
                    nb::cast<StructureDifference&>(sdobj);
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

template <class T>
StructureDifference atomadapter_diff(std::shared_ptr<T> self,
        StructureAdapterConstPtr other)
{
    // Establish shared_from_this() ownership, and allow Python overrides to
    // call super().diff() without dispatching back into the same override.
    auto* wrapper = dynamic_cast<MakeWrapper<T>*>(self.get());
    return wrapper ? wrapper->default_diff(other) : self->diff(other);
}

template <class T>
StructureAdapterPtr atomadapter_clone(const T& self)
{
    auto* wrapper = dynamic_cast<const MakeWrapper<T>*>(&self);
    return wrapper ? wrapper->default_clone() : self.clone();
}

template <class T>
void atomadapter_customPQConfig(const T& self, PairQuantity* pq)
{
    auto* wrapper = dynamic_cast<const MakeWrapper<T>*>(&self);
    if (wrapper) wrapper->default_customPQConfig(pq);
    else self.customPQConfig(pq);
}

// Wrapper helpers for class AtomicStructureAdapter

typedef MakeWrapper<AtomicStructureAdapter> AtomicStructureAdapterWrap;
typedef std::shared_ptr<AtomicStructureAdapterWrap> AtomicStructureAdapterWrapPtr;
typedef AtomicStructureAdapter::value_type data_type;

class AtomAdapterIterator
{
    public:

        explicit AtomAdapterIterator(nb::object container) :
            mcontainer(std::move(container)), midx(0)
        { }

        nb::object next()
        {
            if (midx >= nb::cast<AtomicStructureAdapter&>(mcontainer).size())
            {
                throw nb::stop_iteration();
            }
            return atom_reference(mcontainer, midx++);
        }

    private:

        nb::object mcontainer;
        size_t midx;

};

class atomadapter_indexing : public nb::def_visitor<atomadapter_indexing>
{
    public:

        typedef AtomicStructureAdapter Container;


        template <typename Class, typename... Extra>
        void execute(Class& cls, const Extra&...)
        {
            cls
                .def("__len__", [](const Container& container) {
                    return container.size();
                })
                .def("__getitem__",
                    [](nb::object owner, int idx) {
                        auto& container = nb::cast<Container&>(owner);
                        return atom_reference(owner, normalize_index(container, idx));
                    })
                .def("__getitem__", get_slice)
                .def("__iter__",
                    [](nb::object container) {
                        return AtomAdapterIterator(container);
                    })
                .def("__setitem__",
                    [](Container& container, int idx, const PythonAtom& atom) {
                        const int i = normalize_index(container, idx);
                        Atom value = atom.value();
                        replace_atom_references(container, i, i + 1, 1);
                        container[i] = value;
                    })
                .def("__setitem__", set_slice)
                .def("__delitem__",
                    [](Container& container, int idx) {
                        const int i = normalize_index(container, idx);
                        replace_atom_references(container, i, i + 1, 0);
                        container.erase(i);
                    })
                .def("__delitem__", delete_slice)
                .def("extend", extend, nb::arg("atoms"))
                ;
        }


        static nb::object
        get_slice(Container& container, nb::slice slice)
        {
            const size_t n = container.countSites();
            auto [from, to, step, slice_len] = slice.compute(n);
            // make sure slice is of a correct type and has a copy
            // of any additional structure data.
            StructureAdapterPtr rv = container.clone();
            AtomicStructureAdapterPtr rva =
                std::static_pointer_cast<AtomicStructureAdapter>(rv);
            rva->clear();
            // handle index ranges for a valid and empty slice
            Py_ssize_t idx = from;
            for (size_t i = 0; i < slice_len; ++i, idx += step)
            {
                rva->append(container[static_cast<int>(idx)]);
            }

            // save memory by making a new copy for short slices
            const bool longslice = slice_len > n / 2;
            AtomicStructureAdapterPtr out =
                longslice
                    ? rva
                    : std::static_pointer_cast<AtomicStructureAdapter>(rva->clone());

            return nb::cast(out);
        }


        static std::vector<Atom> extract_atoms(nb::handle values)
        {
            std::vector<Atom> atoms;
            if (nb::isinstance<PythonAtom>(values))
                atoms.push_back(nb::cast<const PythonAtom&>(values).value());
            else
            {
                for (nb::handle value : nb::borrow<nb::object>(values))
                {
                    const PythonAtom* atom = nullptr;
                    if (!nb::try_cast(value, atom, false) || !atom)
                        throw nb::type_error("expected an Atom in the replacement sequence");
                    atoms.push_back(atom->value());
                }
            }
            return atoms;
        }

        static void extend(Container& container, nb::object values)
        {
            auto atoms = extract_atoms(values);
            if (atoms.empty()) return;
            prepare_atom_resize(container);
            container.insert(container.end(), atoms.begin(), atoms.end());
        }

        static void set_slice(Container& container, nb::slice slice, nb::object values)
        {
            auto [first, last, step, count] = slice.compute(container.size());
            auto atoms = extract_atoms(values);
            if (step != 1)
            {
                if (atoms.size() != count)
                    throw nb::value_error("extended slice and replacement have different sizes");
                Py_ssize_t index = first;
                for (size_t i = 0; i < count; ++i, index += step)
                {
                    replace_atom_references(container, index, index + 1, 1);
                    container[static_cast<int>(index)] = atoms[i];
                }
                return;
            }
            last = first + count;
            replace_atom_references(container, first, last, atoms.size());
            container.erase(container.begin() + first, container.begin() + last);
            container.insert(container.begin() + first, atoms.begin(), atoms.end());
        }

        static void delete_slice(Container& container, nb::slice slice)
        {
            auto [first, last, step, count] = slice.compute(container.size());
            if (!count) return;
            if (step == 1)
            {
                last = first + count;
                replace_atom_references(container, first, last, 0);
                container.erase(container.begin() + first, container.begin() + last);
                return;
            }
            std::vector<int> indices;
            Py_ssize_t index = first;
            for (size_t i = 0; i < count; ++i, index += step)
                indices.push_back(static_cast<int>(index));
            std::sort(indices.rbegin(), indices.rend());
            for (int i : indices)
            {
                replace_atom_references(container, i, i + 1, 0);
                container.erase(i);
            }
        }

    private:


        static int normalize_index(const Container& container, int idx)
        {
            ensure_index_bounds(idx, -int(container.size()), container.size());
            return (idx >= 0) ? idx : int(container.size()) + idx;
        }
};


void atomadapter_insert(AtomicStructureAdapter& adpt, const PythonAtom& atom, int idx)
{
    ensure_index_bounds(idx, -int(adpt.size()), adpt.size() + 1);
    const int index = idx >= 0 ? idx : int(adpt.size()) + idx;
    Atom value = atom.value();
    replace_atom_references(adpt, index, index, 1);
    adpt.insert(index, value);
}


PythonAtom atomadapter_pop(AtomicStructureAdapter& adpt, int idx)
{
    ensure_index_bounds(idx, -int(adpt.size()), adpt.size());
    const int index = idx >= 0 ? idx : int(adpt.size()) + idx;
    Atom value = adpt[index];
    replace_atom_references(adpt, index, index + 1, 0);
    adpt.erase(index);
    return PythonAtom(value);
}


void atomadapter_reserve(AtomicStructureAdapter& adpt, int sz)
{
    ensure_non_negative(sz);
    if (static_cast<size_t>(sz) <= adpt.size()) return;
    prepare_atom_resize(adpt);
    adpt.reserve(sz);
}

// Wrapper helpers for class PeriodicStructureAdapter

typedef MakeWrapper<PeriodicStructureAdapter> PeriodicStructureAdapterWrap;
typedef std::shared_ptr<PeriodicStructureAdapterWrap> PeriodicStructureAdapterWrapPtr;

nb::tuple periodicadapter_getlatpar(const PeriodicStructureAdapter& adpt)
{
    const Lattice& L = adpt.getLattice();
    nb::tuple rv = nb::make_tuple(
            L.a(), L.b(), L.c(), L.alpha(), L.beta(), L.gamma());
    return rv;
}

// Wrapper helpers for class CrystalStructureAdapter

typedef MakeWrapper<CrystalStructureAdapter> CrystalStructureAdapterWrap;
typedef std::shared_ptr<CrystalStructureAdapterWrap> CrystalStructureAdapterWrapPtr;

double
crystaladapter_getsymmetryprecision(const CrystalStructureAdapter& adpt)
{
    return adpt.getSymmetryPrecision();
}


void crystaladapter_addsymop(CrystalStructureAdapter& adpt,
        nb::object R, nb::object t)
{
    static SymOpRotTrans op;
    assignR3Matrix(op.R, R);
    assignR3Vector(op.t, t);
    adpt.addSymOp(op);
}


nb::tuple
crystaladapter_getsymop(const CrystalStructureAdapter& adpt, int idx)
{
    ensure_index_bounds(idx, 0, adpt.countSymOps());
    const SymOpRotTrans& op = adpt.getSymOp(idx);
    return nb::make_tuple(
            convertToNumPyArray(op.R), convertToNumPyArray(op.t));
}


template <class Range>
nb::list atoms_aslist(const Range& atoms)
{
    nb::list result;
    for (const Atom& atom : atoms) result.append(PythonAtom(atom));
    return result;
}

nb::object crystaladapter_getequivalentatoms(
        const CrystalStructureAdapter& adpt, int idx)
{
    ensure_index_bounds(idx, 0, adpt.countSymOps());
    return atoms_aslist(adpt.getEquivalentAtoms(idx));
}


nb::list expandLatticeAtom_aslist(const CrystalStructureAdapter& adpt,
        const PythonAtom& atom)
{
    return atoms_aslist(adpt.expandLatticeAtom(atom.value()));
}

}   // namespace nswrap_AtomicStructureAdapter

void prepare_structure_load(diffpy::srreal::StructureAdapter& adapter)
{
    if (auto* atoms = dynamic_cast<AtomicStructureAdapter*>(&adapter))
    {
        replace_atom_references(*atoms, 0, atoms->size(), 0);
    }
}

// declare shared docstrings from wrap_StructureAdapter.cpp

extern const char* doc_StructureAdapter___init__fromstring;
extern const char* doc_StructureAdapter__customPQConfig;
extern const char* doc_StructureAdapter_diff;

// Wrapper definitions -------------------------------------------------------

void wrap_AtomicStructureAdapter(nb::module_& m)
{
    using namespace nswrap_AtomicStructureAdapter;
    using diffpy::srreal::hash_value;

    // class Atom
    nb::class_<PythonAtom> atom_class(m, "Atom", doc_Atom,
            nb::dynamic_attr(), nb::is_weak_referenceable());
    // first define copy constructor and property helper methods
    atom_class
        .def(nb::init<>())
        .def(nb::init<const PythonAtom&>(), nb::arg("atom"), doc_Atom_init_copy)
        .def("__eq__", [](const PythonAtom& a, const PythonAtom& b) {
            return a.value() == b.value();
        }, nb::is_operator())
        .def("__ne__", [](const PythonAtom& a, const PythonAtom& b) {
            return a.value() != b.value();
        }, nb::is_operator())
        .def("__lt__", [](const PythonAtom& a, const PythonAtom& b) {
            return a.value() < b.value();
        }, nb::is_operator())
        .def("__gt__", [](const PythonAtom& a, const PythonAtom& b) {
            return a.value() > b.value();
        }, nb::is_operator())
        .def("__le__", [](const PythonAtom& a, const PythonAtom& b) {
            return a.value() <= b.value();
        }, nb::is_operator())
        .def("__ge__", [](const PythonAtom& a, const PythonAtom& b) {
            return a.value() >= b.value();
        }, nb::is_operator())
        .def("__hash__", [](const PythonAtom& a) { return hash_value(a.value()); })
        .def("_get_xyz_cartn",
                get_xyz_cartn,
                nb::keep_alive<0,1>())
        .def("_get_uij_cartn",
                get_uij_cartn,
                nb::keep_alive<0,1>())
        ;
    // now we can finalize the Atom class interface
    atom_class
        .def_prop_rw("atomtype",
            [](const PythonAtom& a) { return a.value().atomtype; },
            [](PythonAtom& a, const std::string& value) { a.value().atomtype = value; })
        .def_prop_rw("xyz_cartn",
                get_xyz_cartn,
                set_xyz_cartn,
                nb::keep_alive<0,1>())
        .def_prop_rw("xc", get_xyz<0>, set_xyz<0>, doc_Atom_xic)
        .def_prop_rw("yc", get_xyz<1>, set_xyz<1>, doc_Atom_xic)
        .def_prop_rw("zc", get_xyz<2>, set_xyz<2>, doc_Atom_xic)
        .def_prop_rw("occupancy", get_occ, set_occ, doc_Atom_occ)
        .def_prop_rw("anisotropy", get_anisotropy, set_anisotropy,
                nb::for_setter(nb::arg("value").none()),
                doc_Atom_anisotropy)
        .def_prop_rw("uij_cartn",
                get_uij_cartn,
                set_uij_cartn,
                nb::keep_alive<0,1>())
        .def_prop_rw("uc11", get_uc<0, 0>, set_uc<0, 0>, doc_Atom_uijc)
        .def_prop_rw("uc22", get_uc<1, 1>, set_uc<1, 1>, doc_Atom_uijc)
        .def_prop_rw("uc33", get_uc<2, 2>, set_uc<2, 2>, doc_Atom_uijc)
        .def_prop_rw("uc12", get_uc<0, 1>, set_uc<0, 1>, doc_Atom_uijc)
        .def_prop_rw("uc13", get_uc<0, 2>, set_uc<0, 2>, doc_Atom_uijc)
        .def_prop_rw("uc23", get_uc<1, 2>, set_uc<1, 2>, doc_Atom_uijc)
        ;
        AtomPickleSuite::bind(atom_class);

    nb::class_<AtomAdapterIterator>(m, "_AtomicStructureAdapterIterator")
        .def("__iter__", [](AtomAdapterIterator& it) -> AtomAdapterIterator& {
            return it;
        }, nb::rv_policy::reference_internal)
        .def("__next__", &AtomAdapterIterator::next)
        ;

    // class AtomicStructureAdapter
    nb::class_<AtomicStructureAdapter,
            StructureAdapter,
            AtomicStructureAdapterWrap>
    adapter_class(m, "AtomicStructureAdapter", doc_AtomicStructureAdapter,
            nb::dynamic_attr());
    adapter_class
        .def(nb::init<>())
        .def("__init__",
                StructureAdapter_constructor<AtomicStructureAdapter>,
                nb::arg("content"),
                doc_StructureAdapter___init__fromstring)
        .def(atomadapter_indexing())
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("clone",
                atomadapter_clone<AtomicStructureAdapter>,
                doc_AtomicStructureAdapter_clone)
        .def("_customPQConfig",
                atomadapter_customPQConfig<AtomicStructureAdapter>,
                nb::arg("pqobj"),
                doc_StructureAdapter__customPQConfig)
        .def("diff",
                atomadapter_diff<AtomicStructureAdapter>,
                nb::arg("other").none(),
                doc_StructureAdapter_diff)
        .def("insert", atomadapter_insert,
                nb::arg("atom"), nb::arg("index"),
                doc_AtomicStructureAdapter_insert)
        .def("insert", [](AtomicStructureAdapter& adpt, int index, const PythonAtom& atom) {
                    atomadapter_insert(adpt, atom, index);
                }, nb::arg("index"), nb::arg("atom"))
        .def("append", [](AtomicStructureAdapter& adpt, const PythonAtom& atom) {
                    Atom value = atom.value();
                    prepare_atom_resize(adpt);
                    adpt.append(value);
                },
                nb::arg("atom"),
                doc_AtomicStructureAdapter_append)
        .def("pop", atomadapter_pop,
                nb::arg("index"), doc_AtomicStructureAdapter_pop)
        .def("clear", [](AtomicStructureAdapter& adpt) {
                    replace_atom_references(adpt, 0, adpt.size(), 0);
                    adpt.clear();
                },
                doc_AtomicStructureAdapter_clear)
        .def("reserve", atomadapter_reserve,
                nb::arg("sz"), doc_AtomicStructureAdapter_reserve)
        ;
    StructureAdapterPickleSuite<AtomicStructureAdapter>::bind(adapter_class);

    // class PeriodicStructureAdapter
    nb::class_<PeriodicStructureAdapter,
            AtomicStructureAdapter,
            PeriodicStructureAdapterWrap>
    periodic_class(m, "PeriodicStructureAdapter", doc_PeriodicStructureAdapter,
            nb::dynamic_attr());
    periodic_class
        .def(nb::init<>())
        .def("__init__",
                StructureAdapter_constructor<PeriodicStructureAdapter>,
                nb::arg("content"),
                doc_StructureAdapter___init__fromstring)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("clone",
                atomadapter_clone<PeriodicStructureAdapter>,
                doc_PeriodicStructureAdapter_clone)
        .def("_customPQConfig",
                atomadapter_customPQConfig<PeriodicStructureAdapter>,
                nb::arg("pqobj"),
                doc_StructureAdapter__customPQConfig)
        .def("diff",
                atomadapter_diff<PeriodicStructureAdapter>,
                nb::arg("other").none(),
                doc_StructureAdapter_diff)
        .def("getLatPar", periodicadapter_getlatpar,
                doc_PeriodicStructureAdapter_getLatPar)
        .def("setLatPar", &PeriodicStructureAdapter::setLatPar,
                nb::arg("a"), nb::arg("b"), nb::arg("c"),
                nb::arg("alphadeg"), nb::arg("betadeg"),
                nb::arg("gammadeg"),
                doc_PeriodicStructureAdapter_setLatPar)
        .def("toCartesian", [](const PeriodicStructureAdapter& adpt, PythonAtom& atom) {
                    adpt.toCartesian(atom.value());
                },
                nb::arg("atom"), doc_PeriodicStructureAdapter_toCartesian)
        .def("toFractional", [](const PeriodicStructureAdapter& adpt, PythonAtom& atom) {
                    adpt.toFractional(atom.value());
                },
                nb::arg("atom"), doc_PeriodicStructureAdapter_toFractional)
        ;
    StructureAdapterPickleSuite<PeriodicStructureAdapter>::bind(periodic_class);

    // class CrystalStructureAdapter
    nb::class_<CrystalStructureAdapter,
            PeriodicStructureAdapter,
            CrystalStructureAdapterWrap>
    crystal_class(m, "CrystalStructureAdapter", doc_CrystalStructureAdapter,
            nb::dynamic_attr());
    crystal_class
        .def(nb::init<>())
        .def("__init__",
                StructureAdapter_constructor<CrystalStructureAdapter>,
                nb::arg("content"),
                doc_StructureAdapter___init__fromstring)
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("clone",
                atomadapter_clone<CrystalStructureAdapter>,
                doc_CrystalStructureAdapter_clone)
        .def("_customPQConfig",
                atomadapter_customPQConfig<CrystalStructureAdapter>,
                nb::arg("pqobj"),
                doc_StructureAdapter__customPQConfig)
        .def("diff",
                atomadapter_diff<CrystalStructureAdapter>,
                nb::arg("other").none(),
                doc_StructureAdapter_diff)
        .def_prop_rw("symmetryprecision",
            crystaladapter_getsymmetryprecision,
            &CrystalStructureAdapter::setSymmetryPrecision,
            doc_CrystalStructureAdapter_symmetryprecision)
        .def("countSymOps", &CrystalStructureAdapter::countSymOps,
                doc_CrystalStructureAdapter_countSymOps)
        .def("clearSymOps", &CrystalStructureAdapter::clearSymOps,
                doc_CrystalStructureAdapter_clearSymOps)
        .def("addSymOp", crystaladapter_addsymop,
                nb::arg("R"), nb::arg("t"),
                doc_CrystalStructureAdapter_addSymOp)
        .def("getSymOp", crystaladapter_getsymop, nb::arg("index"),
                doc_CrystalStructureAdapter_getSymOp)
        .def("getEquivalentAtoms",
                crystaladapter_getequivalentatoms, nb::arg("index"),
                doc_CrystalStructureAdapter_getEquivalentAtoms)
        .def("expandLatticeAtom",
                expandLatticeAtom_aslist,
                nb::arg("atom"),
                doc_CrystalStructureAdapter_expandLatticeAtom)
        .def("updateSymmetryPositions",
                &CrystalStructureAdapter::updateSymmetryPositions,
                doc_CrystalStructureAdapter_updateSymmetryPositions)
        ;
    StructureAdapterPickleSuite<CrystalStructureAdapter>::bind(crystal_class);

}

}   // namespace srrealmodule

using srrealmodule::nswrap_AtomicStructureAdapter::AtomicStructureAdapterWrap;
BOOST_CLASS_EXPORT(AtomicStructureAdapterWrap)

using srrealmodule::nswrap_AtomicStructureAdapter::PeriodicStructureAdapterWrap;
BOOST_CLASS_EXPORT(PeriodicStructureAdapterWrap)

using srrealmodule::nswrap_AtomicStructureAdapter::CrystalStructureAdapterWrap;
BOOST_CLASS_EXPORT(CrystalStructureAdapterWrap)

// End of file
