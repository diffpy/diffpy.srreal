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
#include <map>

#include <diffpy/srreal/PythonStructureAdapter.hpp>

using namespace boost;
using namespace std;

namespace diffpy {
namespace srreal {

// Local Helpers -------------------------------------------------------------

namespace {

/// Obtain object from specified Python module or None when not available

python::object importFromPyModule(const string& modname, const string& item)
{
    static map<string, python::object> cacheditems;
    string fullname = modname + "." + item;
    // perform import when not in the cache
    if (!cacheditems.count(fullname))
    {
        try {
            cacheditems[fullname] =
                python::import(modname.c_str()).attr(item.c_str());
        }
        // Ignore import errors when item could not be recovered.
        catch (python::error_already_set e) {
            PyErr_Clear();
            cacheditems[fullname] = python::object();
        }
    }
    return cacheditems[fullname];
}


// diffpy.Structure.Structure and derived classes

StructureAdapter* createDiffPyStructureAdapter(const python::object& stru)
{
    python::object cls_Structure;
    cls_Structure = importFromPyModule("diffpy.Structure", "Structure");
    StructureAdapter* rv = NULL;
    if (cls_Structure.ptr() &&
        PyObject_IsInstance(stru.ptr(), cls_Structure.ptr()))
    {
        rv = new DiffPyStructureAdapter(stru);
    }
    return rv;
}

// pyobjcryst.crystal.Crystal and derived classes

StructureAdapter* createPyObjCrystStructureAdapter(const python::object& stru)
{
    python::object cls_Crystal;
    cls_Crystal = importFromPyModule("pyobjcryst.crystal", "Crystal");
    StructureAdapter* rv = NULL;
    if (cls_Crystal.ptr() &&
        PyObject_IsInstance(stru.ptr(), cls_Crystal.ptr()));
    {
        const ObjCryst::Crystal* pcryst =
            python::extract<ObjCryst::Crystal*>(stru);
        rv = createPQAdapter(*pcryst);
    }
    return rv;
}

}   // namespace

// Routines ------------------------------------------------------------------

StructureAdapter* createPQAdapter(const boost::python::object& stru)
{
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
