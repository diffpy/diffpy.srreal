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
* class PeakProfile -- implementation of concrete class registry and
*     of a factory for borrowed pointers
*
* $Id$
*
*****************************************************************************/

#ifndef PEAKPROFILE_IPP_INCLUDED
#define PEAKPROFILE_IPP_INCLUDED

#include <sstream>
#include <stdexcept>

namespace diffpy {
namespace srreal {

// class PeakProfile ---------------------------------------------------------

inline
PeakProfile::RegistryType& PeakProfile::getRegistry()
{
    static RegistryType the_registry;
    return the_registry;
}

// Factory Functions ---------------------------------------------------------

inline
const PeakProfile* borrowPeakProfile(const std::string& tp)
{
    using namespace std;
    PeakProfile::RegistryType& reg = PeakProfile::getRegistry();
    PeakProfile::RegistryType::iterator iprfl;
    iprfl = reg.find(tp);
    if (iprfl == reg.end())
    {
        ostringstream emsg;
        emsg << "Unknown type of PeakProfile '" << tp << "'.";
        throw invalid_argument(emsg.str());
    }
    const PeakProfile* rv = iprfl->second;
    return rv;
}


inline
bool registerPeakProfile(const PeakProfile* prfl)
{
    using namespace std;
    PeakProfile::RegistryType& reg = PeakProfile::getRegistry();
    if (reg.count(prfl->type()))
    {
        ostringstream emsg;
        emsg << "PeakProfile type '" << prfl->type() <<
            "' is already registered.";
        throw logic_error(emsg.str());
    }
    reg[prfl->type()] = prfl;
    return true;
}


}   // namespace srreal
}   // namespace diffpy

// vim:ft=cpp

#endif  // PEAKPROFILE_IPP_INCLUDED
