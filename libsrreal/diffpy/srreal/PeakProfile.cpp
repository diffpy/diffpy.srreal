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
* Concrete implementations of the abstract PeakProfile class:
*
* class GaussPeakProfile -- registered as "gauss"
*
* $Id$
*
*****************************************************************************/

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <diffpy/srreal/PeakProfile.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

//////////////////////////////////////////////////////////////////////////////
// class PeakProfile
//////////////////////////////////////////////////////////////////////////////

PeakProfile::RegistryType& PeakProfile::getRegistry()
{
    static auto_ptr<RegistryType> the_registry;
    if (!the_registry.get())  the_registry.reset(new RegistryType());
    return *the_registry;
}

// Factory Functions ---------------------------------------------------------

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


PeakProfile* createPeakProfile(const std::string& tp)
{
    const PeakProfile* ppf = borrowPeakProfile(tp);
    PeakProfile* rv = ppf->copy();
    return rv;
}


bool registerPeakProfile(const PeakProfile& prfl)
{
    using namespace std;
    PeakProfile::RegistryType& reg = PeakProfile::getRegistry();
    if (reg.count(prfl.type()))
    {
        ostringstream emsg;
        emsg << "PeakProfile type '" << prfl.type() <<
            "' is already registered.";
        throw logic_error(emsg.str());
    }
    reg[prfl.type()] = prfl.copy();
    return true;
}

//////////////////////////////////////////////////////////////////////////////
// class GaussPeakProfile
//////////////////////////////////////////////////////////////////////////////

class GaussPeakProfile : public PeakProfile
{
    public:

        // constructors
        PeakProfile* create() const;
        PeakProfile* copy() const;

        // methods
        const std::string& type() const;
        double y(double x, double fwhm) const;
        double xboundlo(double eps_y, double fwhm) const;
        double xboundhi(double eps_y, double fwhm) const;

};

// Implementation ------------------------------------------------------------

PeakProfile* GaussPeakProfile::create() const
{
    PeakProfile* rv = new GaussPeakProfile();
    return rv;
}


PeakProfile* GaussPeakProfile::copy() const
{
    PeakProfile* rv = new GaussPeakProfile(*this);
    return rv;
}


const string& GaussPeakProfile::type() const
{
    static string rv = "gauss";
    return rv;
}


double GaussPeakProfile::y(double x, double fwhm) const
{
    double xrel = x / fwhm;
    double rv = 2 * sqrt(M_LN2 / M_PI) / fwhm * exp(-4 * M_LN2 * xrel * xrel);
    return rv;
}


double GaussPeakProfile::xboundlo(double eps_y, double fwhm) const
{
    return -1 * this->xboundhi(eps_y, fwhm);
}


double GaussPeakProfile::xboundhi(double eps_y, double fwhm) const
{
    double rv = (eps_y >= 1.0 || fwhm <= 0.0) ? 0.0 :
        fwhm * sqrt(-log(eps_y) / (4 * M_LN2));
    return rv;
}

// Registration --------------------------------------------------------------

bool reg_GaussPeakProfile = registerPeakProfile(GaussPeakProfile());

}   // namespace srreal
}   // namespace diffpy

// End of file
