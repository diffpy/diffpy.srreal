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
* class GaussPeakProfile -- full scale concrete implementation of the
*     PeakProfile class.  GaussPeakProfile is registered as "gauss".
*
* $Id$
*
*****************************************************************************/

#include <cmath>

#include <diffpy/srreal/GaussPeakProfile.hpp>
#include <diffpy/mathutils.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

// Constructors --------------------------------------------------------------

GaussPeakProfile::GaussPeakProfile() : mhalfboundrel(0.0)
{ }


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

// Public Methods ------------------------------------------------------------

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
