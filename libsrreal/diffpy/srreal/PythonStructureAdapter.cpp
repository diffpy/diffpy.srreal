/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* createPQAdapter - a factory for creating StructureAdapter for recognized
*     Python objects
*
* $Id$
*
*****************************************************************************/

#include <stdexcept>
#include <sstream>
#include <string>

#include <diffpy/srreal/PythonStructureAdapter.hpp>

using namespace boost;

namespace diffpy {
namespace srreal {

// Local Helpers -------------------------------------------------------------

namespace {

// diffpy.Structure.Structure and derived classes

StructureAdapter* createDiffPyStructureAdapter(const python::object& stru)
{
    static python::object cls_Structure;
    static bool loaded_cls_Structure;
    if (!loaded_cls_Structure)
    {
        loaded_cls_Structure = true;
        try {
            cls_Structure =
                python::import("diffpy.Structure").attr("Structure");
        }
        // Ignore import errors when diffpy.Structure is not installed
        catch (python::error_already_set e) {
            PyErr_Clear();
        }
    }
    if (cls_Structure.ptr() &&
        PyObject_IsInstance(stru.ptr(), cls_Structure.ptr()))
    {
        StructureAdapter* rv = new DiffPyStructureAdapter(stru);
        return rv;
    }
    return NULL;
}

}   // namespace

StructureAdapter* createPyObjCrystStructureAdapter(const python::object& cryst)
{
    static python::object cls_Crystal;
    static bool loaded_cls_Crystal;
    if (!loaded_cls_Crystal)
    {
        loaded_cls_Crystal = true;
        try {
            cls_Crystal =
                python::import("pyobjcryst.crystal").attr("Crystal");
        }
        // Ignore import errors when pyobjcryst is not installed
        catch (python::error_already_set e) {
            PyErr_Clear();
        }
    }
    if (cls_Crystal.ptr() &&
        PyObject_IsInstance(cryst.ptr(), cls_Crystal.ptr()));
    {
        StructureAdapter* rv = createPQAdapter( 
                python::extract< ObjCryst::Crystal* >(cryst) );
        return rv;
    }
    return NULL;
}




// Routines ------------------------------------------------------------------

StructureAdapter* createPQAdapter(const boost::python::object& stru)
{
    using namespace std;
    // NOTE: This may be later replaced with a Boost Python converter.
    StructureAdapter* rv = NULL;
    // Check if stru is a diffpy.Structure or derived object
    rv = createDiffPyStructureAdapter(stru);
    if (rv)  return rv;
    // Check if stru is a pyobjcryst.Crystal or derived object
    rv = createPyObjCrystStructureAdapter(stru);
    if (rv)  return rv;
    // add other python adapters here ...
    //
    // We get here only if nothing worked.
    python::object pytype = python::import("__builtin__").attr("type");
    python::object tp = python::str(pytype(stru));
    ostringstream emsg;
    emsg << "Cannot create structure adapter for Python " <<
        string(python::extract<string>(tp)) << ".";
    throw invalid_argument(emsg.str());
}


}   // namespace srreal
}   // namespace diffpy
