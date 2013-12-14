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
#include <boost/python/slice.hpp>

#include <diffpy/srreal/AtomicStructureAdapter.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

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

}   // namespace nswrap_AtomicStructureAdapter

// Wrapper definitions -------------------------------------------------------

void wrap_AtomicStructureAdapter()
{
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

}

}   // namespace srrealmodule

// End of file
