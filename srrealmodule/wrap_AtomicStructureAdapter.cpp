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
cartesianposition -- cartesian coordinates viewed as NumPy array\n\
occupancy    -- fractional occupancy of this atom site\n\
anisotropy   -- boolean flag for anisotropic displacements at this site\n\
cartesianuij -- matrix of anisotropic displacements parameters viewed\n\
                as NumPy array\n\
\n\
Note cartesianposition and cartesianuij are NumPy arrays with a direct\n\
view to the data in C++ class.  Do not resize or reshape.\n\
";

const char* doc_AtomicStructureAdapter = "";
const char* doc_AtomicStructureAdapter_insert = "FIXME";
const char* doc_AtomicStructureAdapter_append = "FIXME";
const char* doc_AtomicStructureAdapter_pop = "FIXME";
const char* doc_AtomicStructureAdapter_reserve = "FIXME";

// wrappers ------------------------------------------------------------------

// Wrapper helpers for the class Atom

object get_cartesianposition(Atom& a)
{
    return viewAsNumPyArray(a.cartesianposition);
}

void set_cartesianposition(Atom& a, object value)
{
    object xyzc = get_cartesianposition(a);
    xyzc[slice()] = value;
}


object get_cartesianuij(Atom& a)
{
    return viewAsNumPyArray(a.cartesianuij);
}

void set_cartesianuij(Atom& a, object value)
{
    object uijc = get_cartesianuij(a);
    uijc[slice()] = value;
}

// Wrapper helpers for class AtomicStructureAdapter

AtomicStructureAdapterPtr atomadapter_create()
{
    return AtomicStructureAdapterPtr(new AtomicStructureAdapter);
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

}   // namespace nswrap_AtomicStructureAdapter

// Wrapper definitions -------------------------------------------------------

void wrap_AtomicStructureAdapter()
{
    namespace bp = boost::python;
    using namespace nswrap_AtomicStructureAdapter;

    // class Atom
    class_<Atom> atom_class("Atom", doc_Atom);
    // first define the property helper methods
    atom_class
        .def("_get_cartesianposition",
                get_cartesianposition,
                with_custodian_and_ward_postcall<0,1>())
        .def("_get_cartesianuij",
                get_cartesianuij,
                with_custodian_and_ward_postcall<0,1>())
        ;
    // now we can finalize the Atom class interface
    atom_class
        .def_readwrite("atomtype", &Atom::atomtype)
        .add_property("cartesianposition",
                atom_class.attr("_get_cartesianposition"),
                set_cartesianposition)
        .def_readwrite("occupancy", &Atom::occupancy)
        .def_readwrite("anisotropy", &Atom::anisotropy)
        .add_property("cartesianuij",
                atom_class.attr("_get_cartesianuij"),
                set_cartesianuij)
        ;

    // class AtomicStructureAdapter
    class_<AtomicStructureAdapter, bases<StructureAdapter> >(
            "AtomicStructureAdapter", doc_AtomicStructureAdapter)
        // object from default constructor would throw tr1::bad_weak_ptr
        // when calling shared_from_this, but it seems to work well
        // if constructed with a factory function.
        .def("__init__", make_constructor(atomadapter_create))
        .def(atomadapter_indexing())
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

}

}   // namespace srrealmodule

// End of file
