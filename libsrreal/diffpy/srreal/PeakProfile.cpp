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
#include <diffpy/mathutils.hpp>

using namespace std;
using diffpy::ClassRegistry;

namespace diffpy {
namespace srreal {

//////////////////////////////////////////////////////////////////////////////
// class PeakProfile
//////////////////////////////////////////////////////////////////////////////

// Public Methods ------------------------------------------------------------

void PeakProfile::setPrecision(double eps)
{
    mprecision = eps;
}


double PeakProfile::getPrecision() const
{
    return mprecision;
}

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
        GaussPeakProfile() : PeakProfile(), mhalfboundrel(0.0)  { }

        // methods
        const string& type() const;
        double y(double x, double fwhm) const;
        double xboundlo(double fwhm) const;
        double xboundhi(double fwhm) const;
        void setPrecision(double eps);

    private:

        // data
        double mhalfboundrel;

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


double GaussPeakProfile::xboundlo(double fwhm) const
{
    return -1 * this->xboundhi(fwhm);
}


double GaussPeakProfile::xboundhi(double fwhm) const
{
    double rv = (fwhm <= 0.0) ? 0.0 : (mhalfboundrel * fwhm);
    return rv;
}


void GaussPeakProfile::setPrecision(double eps)
{
    using diffpy::mathutils::DOUBLE_MAX;
    this->PeakProfile::setPrecision(eps);
    if (eps <= 0.0)  mhalfboundrel = DOUBLE_MAX;
    else if (eps < 1.0)  mhalfboundrel = sqrt(-log(eps) / (4 * M_LN2));
    else  mhalfboundrel = 0.0;
}


// Registration --------------------------------------------------------------

bool reg_GaussPeakProfile = registerPeakProfile(GaussPeakProfile());

}   // namespace srreal
}   // namespace diffpy

// End of file
