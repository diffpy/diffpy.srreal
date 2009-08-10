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

#ifndef PYTHONSTRUCTUREADAPTER_HPP_INCLUDED
#define PYTHONSTRUCTUREADAPTER_HPP_INCLUDED

#include <boost/python.hpp>

#include <diffpy/srreal/DiffPyStructureAdapter.hpp>
#include <diffpy/srreal/ObjCrystStructureAdapter.hpp>

namespace diffpy {
namespace srreal {

class StructureAdapter;

/// Factory for constructing appropriate StructureAdapter for a Python object.
StructureAdapter* createPQAdapter(const boost::python::object& stru);

}   // namespace srreal
}   // namespace diffpy

#endif  // PYTHONSTRUCTUREADAPTER_HPP_INCLUDED
