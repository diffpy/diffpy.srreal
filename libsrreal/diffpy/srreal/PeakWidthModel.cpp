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
*
* $Id$
*
*****************************************************************************/

#include <sstream>
#include <stdexcept>
#include <memory>
#include <diffpy/srreal/PeakWidthModel.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

// class PeakWidthModel ------------------------------------------------------

PeakWidthModel::RegistryType& PeakWidthModel::getRegistry()
{
    static auto_ptr<RegistryType> the_registry;
    if (!the_registry.get())  the_registry.reset(new RegistryType());
    return *the_registry;
}

// Factory Functions ---------------------------------------------------------

PeakWidthModel* createPeakWidthModel(const std::string& tp)
{
    using namespace std;
    PeakWidthModel::RegistryType& reg = PeakWidthModel::getRegistry();
    PeakWidthModel::RegistryType::iterator ipwm;
    ipwm = reg.find(tp);
    if (ipwm == reg.end())
    {
        ostringstream emsg;
        emsg << "Unknown type of PeakWidthModel '" << tp << "'.";
        throw invalid_argument(emsg.str());
    }
    PeakWidthModel* rv = ipwm->second->copy();
    return rv;
}


bool registerPeakWidthModel(const PeakWidthModel& pwm)
{
    using namespace std;
    PeakWidthModel::RegistryType& reg = PeakWidthModel::getRegistry();
    if (reg.count(pwm.type()))
    {
        ostringstream emsg;
        emsg << "PeakWidthModel type '" << pwm.type() <<
            "' is already registered.";
        throw logic_error(emsg.str());
    }
    reg[pwm.type()] = pwm.create();
    return true;
}

}   // namespace srreal
}   // namespace diffpy

// End of file.
