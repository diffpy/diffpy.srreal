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
* Pickling support that uses serialization of the libdiffpy classes.
*
*****************************************************************************/

#include <diffpy/srreal/StructureAdapter.hpp>
#include "srreal_pickling.hpp"

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

// class StructureAdapterPickleSuite -----------------------------------------

boost::python::tuple
StructureAdapterPickleSuite::getinitargs(StructureAdapterPtr adpt)
{
    std::string content = diffpy::serialization_tostring(adpt);
    return boost::python::make_tuple(content);
}


boost::python::object
StructureAdapterPickleSuite::constructor()
{
    return boost::python::make_constructor(createStructureAdapterFromString);
}

// shared docstring for StructureAdapterPickleSuite clients

const char* doc_StructureAdapter___init__fromstring = "\
Construct StructureAdapter object from a string.  This is used\n\
internally by the pickle protocol and should not be called directly.\n\
";

}   // namespace srrealmodule

// End of file
