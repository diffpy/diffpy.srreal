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
* class GaussianProfile -- concrete implementation of the PeakProfile class. 
*     GaussianProfile is registered as "gaussian".
*
* $Id$
*
*****************************************************************************/

#include <cmath>

#include <diffpy/srreal/GaussianProfile.hpp>
#include <diffpy/mathutils.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

// Constructors --------------------------------------------------------------

GaussianProfile::GaussianProfile() : mhalfboundrel(0.0)
{ }


PeakProfile* GaussianProfile::create() const
{
    PeakProfile* rv = new GaussianProfile();
    return rv;
}


PeakProfile* GaussianProfile::copy() const
{
    PeakProfile* rv = new GaussianProfile(*this);
    return rv;
}

// Public Methods ------------------------------------------------------------

const string& GaussianProfile::type() const
{
    static string rv = "gaussian";
    return rv;
}


double GaussianProfile::yvalue(double x, double fwhm) const
{
    double xrel = x / fwhm;
    double rv = 2 * sqrt(M_LN2 / M_PI) / fwhm * exp(-4 * M_LN2 * xrel * xrel);
    return rv;
}


double GaussianProfile::xboundlo(double fwhm) const
{
    return -1 * this->GaussianProfile::xboundhi(fwhm);
}


double GaussianProfile::xboundhi(double fwhm) const
{
    double rv = (fwhm <= 0.0) ? 0.0 : (mhalfboundrel * fwhm);
    return rv;
}


void GaussianProfile::setPrecision(double eps)
{
    using diffpy::mathutils::DOUBLE_EPS;
    using diffpy::mathutils::DOUBLE_MAX;
    // correct any settings below DOUBLE_EPS
    double eps1 = max(eps, DOUBLE_EPS);
    this->PeakProfile::setPrecision(eps1);
    if (eps1 < 1.0)  mhalfboundrel = sqrt(-log(eps1) / (4 * M_LN2));
    else  mhalfboundrel = 0.0;
}

// Registration --------------------------------------------------------------

bool reg_GaussianProfile = registerPeakProfile(GaussianProfile());

}   // namespace srreal
}   // namespace diffpy

// End of file
