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
* class PeakWidthModel -- base class for calculation of peak widths.
*     Implementation of inline functions.
*
* $Id$
*
*****************************************************************************/

#ifndef PEAKWIDTHMODEL_IPP_INCLUDED
#define PEAKWIDTHMODEL_IPP_INCLUDED

#include <sstream>
#include <stdexcept>

namespace diffpy {
namespace srreal {

// class PeakWidthModel ------------------------------------------------------

inline
std::map<std::string, PeakWidthModel*>& PeakWidthModel::getRegistry()
{
    static std::map<std::string, PeakWidthModel*> the_registry;
    return the_registry;
}

// Factory Functions ---------------------------------------------------------

inline
PeakWidthModel* createPeakWidthModel(const std::string& tp)
{
    using namespace std;
    map<string, PeakWidthModel*>& reg = PeakWidthModel::getRegistry();
    map<string, PeakWidthModel*>::iterator ipwm;
    ipwm = reg.find(tp);
    if (ipwm == reg.end())
    {
        stringstream emsg;
        emsg << "Unknown type of PeakWidthModel '" << tp << "'.";
        throw invalid_argument(emsg.str());
    }
    PeakWidthModel* rv = ipwm->second->create();
    return rv;
}


inline
bool registerPeakWidthModel(const PeakWidthModel& pwm)
{
    using namespace std;
    map<string, PeakWidthModel*>& reg = PeakWidthModel::getRegistry();
    if (reg.count(pwm.type()))
    {
        stringstream emsg;
        emsg << "PeakWidthModel type '" << pwm.type() <<
            "' is already registered.";
        throw logic_error(emsg.str());
    }
    reg[pwm.type()] = pwm.create();
    return true;
}


}   // namespace srreal
}   // namespace diffpy

// vim:ft=cpp

#endif  // PEAKWIDTHMODEL_IPP_INCLUDED
