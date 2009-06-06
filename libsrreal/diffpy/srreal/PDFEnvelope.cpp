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
* class PDFEnvelope -- abstract base class for PDF envelope functions
*
* $Id$
*
*****************************************************************************/

#include <sstream>
#include <stdexcept>
#include <memory>

#include <diffpy/srreal/PDFEnvelope.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

// class PDFEnvelope ---------------------------------------------------------

// private methods

PDFEnvelope::RegistryType&
PDFEnvelope::getRegistry()
{
    static auto_ptr<RegistryType> the_registry;
    if (!the_registry.get())  the_registry.reset(new RegistryType());
    return *the_registry;
}

// Factory Functions ---------------------------------------------------------

PDFEnvelope* createPDFEnvelope(const std::string& tp)
{
    using namespace std;
    PDFEnvelope::RegistryType& reg = PDFEnvelope::getRegistry();
    PDFEnvelope::RegistryType::iterator isft;
    isft = reg.find(tp);
    if (isft == reg.end())
    {
        ostringstream emsg;
        emsg << "Unknown type of PDFEnvelope '" << tp << "'.";
        throw invalid_argument(emsg.str());
    }
    PDFEnvelope* rv = isft->second->create();
    return rv;
}


bool registerPDFEnvelope(const PDFEnvelope& sft)
{
    using namespace std;
    PDFEnvelope::RegistryType&
        reg = PDFEnvelope::getRegistry();
    if (reg.count(sft.type()))
    {
        ostringstream emsg;
        emsg << "PDFEnvelope type '" << sft.type() <<
            "' is already registered.";
        throw logic_error(emsg.str());
    }
    reg[sft.type()] = sft.copy();
    return true;
}


bool aliasPDFEnvelope(const std::string& tp, const std::string& al)
{
    using namespace std;
    map<string, const PDFEnvelope*>& reg =
        PDFEnvelope::getRegistry();
    if (!reg.count(tp))
    {
        ostringstream emsg;
        emsg << "Unknown PDFEnvelope '" << tp << "'.";
        throw logic_error(emsg.str());
    }
    if (reg.count(al) && reg[al] != reg[tp])
    {
        ostringstream emsg;
        emsg << "PDFEnvelope type '" << al <<
            "' is already registered.";
        throw logic_error(emsg.str());
    }
    reg[al] = reg[tp];
    return true;
}

}   // namespace srreal
}   // namespace diffpy

// End of file
