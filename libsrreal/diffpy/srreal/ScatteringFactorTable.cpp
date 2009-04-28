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
* class ScatteringFactorTable -- base class for looking up scattering factors
*
* $Id$
*
*****************************************************************************/

#include <sstream>
#include <stdexcept>
#include <memory>
#include <diffpy/srreal/ScatteringFactorTable.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

// class ScatteringFactorTable -----------------------------------------------

// public methods

const double& ScatteringFactorTable::lookup(const std::string& smbl) const
{
    using namespace std;
    map<string, double>::const_iterator isft;
    isft = mtable.find(smbl);
    if (isft == mtable.end())
    {
        double value = this->fetch(smbl);
        mtable[smbl] = value;
        isft = mtable.find(smbl);
    }
    return isft->second;
}


void ScatteringFactorTable::setCustom(const std::string& smbl, double value)
{
    mtable[smbl] = value;
}


void ScatteringFactorTable::resetCustom(const std::string& smbl)
{
    mtable.erase(smbl);
}


void ScatteringFactorTable::resetAll()
{
    mtable.clear();
}

// private methods

ScatteringFactorTable::RegistryType&
ScatteringFactorTable::getRegistry()
{
    static auto_ptr<RegistryType> the_registry;
    if (!the_registry.get())  the_registry.reset(new RegistryType());
    return *the_registry;
}

// Factory Functions ---------------------------------------------------------

ScatteringFactorTable* createScatteringFactorTable(const std::string& tp)
{
    using namespace std;
    ScatteringFactorTable::RegistryType& reg =
        ScatteringFactorTable::getRegistry();
    ScatteringFactorTable::RegistryType::iterator isft;
    isft = reg.find(tp);
    if (isft == reg.end())
    {
        ostringstream emsg;
        emsg << "Unknown type of ScatteringFactorTable '" << tp << "'.";
        throw invalid_argument(emsg.str());
    }
    ScatteringFactorTable* rv = isft->second->create();
    return rv;
}


bool registerScatteringFactorTable(const ScatteringFactorTable& sft)
{
    using namespace std;
    ScatteringFactorTable::RegistryType&
        reg = ScatteringFactorTable::getRegistry();
    if (reg.count(sft.type()))
    {
        ostringstream emsg;
        emsg << "ScatteringFactorTable type '" << sft.type() <<
            "' is already registered.";
        throw logic_error(emsg.str());
    }
    reg[sft.type()] = sft.copy();
    return true;
}


bool aliasScatteringFactorTable(const std::string& tp, const std::string& al)
{
    using namespace std;
    map<string, const ScatteringFactorTable*>& reg =
        ScatteringFactorTable::getRegistry();
    if (!reg.count(tp))
    {
        ostringstream emsg;
        emsg << "Unknown ScatteringFactorTable '" << tp << "'.";
        throw logic_error(emsg.str());
    }
    if (reg.count(al) && reg[al] != reg[tp])
    {
        ostringstream emsg;
        emsg << "ScatteringFactorTable type '" << al <<
            "' is already registered.";
        throw logic_error(emsg.str());
    }
    reg[al] = reg[tp];
    return true;
}

}   // namespace srreal
}   // namespace diffpy

// End of file.
