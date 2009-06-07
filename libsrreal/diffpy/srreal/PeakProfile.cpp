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
#include <diffpy/ClassRegistry.hpp>

using namespace std;
using diffpy::ClassRegistry;

namespace diffpy {
namespace srreal {

//////////////////////////////////////////////////////////////////////////////
// class PeakProfile
//////////////////////////////////////////////////////////////////////////////

// Factory Functions ---------------------------------------------------------

PeakProfile* createPeakProfile(const string& tp)
{
    return ClassRegistry<PeakProfile>::create(tp);
}


bool registerPeakProfile(const PeakProfile& ref)
{
    return ClassRegistry<PeakProfile>::add(ref);
}


bool aliasPeakProfile(const string& tp, const string& al)
{
    return ClassRegistry<PeakProfile>::alias(tp, al);
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
        const string& type() const;
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
