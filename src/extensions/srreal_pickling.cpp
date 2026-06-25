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


StructureAdapterPtr
createStructureAdapterFromString(const std::string& content)
{
    StructureAdapterPtr adpt;
    diffpy::serialization_fromstring(adpt, content);
    return adpt;
}

}   // namespace srrealmodule

// End of file
