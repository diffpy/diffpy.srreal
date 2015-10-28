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
* Pickling support that uses serialization of the libdiffpy classes.
*
*****************************************************************************/

#include "srreal_pickling.hpp"
#include <boost/python/make_constructor.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>

namespace srrealmodule {
namespace {

using diffpy::srreal::StructureAdapterPtr;

StructureAdapterPtr
createStructureAdapterFromString(const std::string& content)
{
    StructureAdapterPtr adpt;
    diffpy::serialization_fromstring(adpt, content);
    return adpt;
}

}   // namespace

// Non-member functions for StructureAdapterPickleSuite ----------------------

boost::python::object
StructureAdapter_constructor()
{
    return boost::python::make_constructor(createStructureAdapterFromString);
}

}   // namespace srrealmodule

// End of file
