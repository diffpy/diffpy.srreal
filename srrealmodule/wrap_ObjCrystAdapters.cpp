/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   (c) 2014 Brookhaven Science Associates,
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
* Support Crystal and Molecule objects from pyobjcryst if libdiffpy
* has been built with ObjCryst support.
*
*****************************************************************************/

#include <boost/python/def.hpp>

#include <diffpy/features.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>

#ifdef DIFFPY_HAS_OBJCRYST
#include <diffpy/srreal/ObjCrystStructureAdapter.hpp>
#endif

namespace srrealmodule {
namespace nswrap_ObjCrystAdapters {

using namespace boost;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

// class Atom

const char* doc_convertObjCrystMolecule = "\
Convert pyobjcryst Molecule object to AtomicStructureAdapter.\n\
Instance data:\n\
\n\
molecule     -- instance of pyobjcryst Molecule object\n\
\n\
Return AtomicStructureAdapter.\n\
Raise TypeError if ObjCryst was not available at compile time.\n\
";

const char* doc_convertObjCrystCrystal = "\
Convert pyobjcryst Crystal object to PeriodicStructureAdapter.\n\
Instance data:\n\
\n\
molecule     -- instance of pyobjcryst Crystal object\n\
\n\
Return PeriodicStructureAdapter.\n\
Raise TypeError if ObjCryst was not available at compile time.\n\
";

// ObjCryst supported --------------------------------------------------------

#ifdef DIFFPY_HAS_OBJCRYST

using ObjCryst::Molecule;
using ObjCryst::Crystal;

StructureAdapterPtr convertObjCrystMolecule(const Molecule& mol)
{
    return createStructureAdapter(mol);
}

StructureAdapterPtr convertObjCrystCrystal(const Crystal& mol)
{
    return createStructureAdapter(mol);
}

// ObjCryst not available ----------------------------------------------------

#else

StructureAdapterPtr convertObjCrystMolecule(python::object mol)
{
    std::string emsg = "ObjCryst support not available.";
    PyErr_SetString(PyExc_TypeError, emsg.c_str());
    boost::python::throw_error_already_set();
    assert(false);
    return StructureAdapterPtr();
}

StructureAdapterPtr convertObjCrystCrystal(python::object cryst)
{
    // raise the same exception as for the Molecule
    return convertObjCrystMolecule(cryst);
}

#endif  // DIFFPY_HAS_OBJCRYST

}   // namespace nswrap_ObjCrystAdapters

// Wrapper definitions -------------------------------------------------------

void wrap_ObjCrystAdapters()
{
    using namespace nswrap_ObjCrystAdapters;
    using namespace boost::python;

    def("convertObjCrystMolecule",
            convertObjCrystMolecule, doc_convertObjCrystMolecule);
    def("convertObjCrystCrystal",
            convertObjCrystCrystal, doc_convertObjCrystCrystal);

}

}   // namespace srrealmodule

// End of file
