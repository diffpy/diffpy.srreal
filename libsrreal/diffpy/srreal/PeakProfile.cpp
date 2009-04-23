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
#include <diffpy/srreal/PeakProfile.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

//////////////////////////////////////////////////////////////////////////////
// class GaussPeakProfile
//////////////////////////////////////////////////////////////////////////////

class GaussPeakProfile : public PeakProfile
{
    public:

        // methods
        const std::string& type() const;
        double y(double x, double fwhm) const;
        double xboundlo(double eps_y, double fwhm) const;
        double xboundhi(double eps_y, double fwhm) const;

};

// Implementation ------------------------------------------------------------

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

bool reg_GaussPeakProfile = registerPeakProfile(new GaussPeakProfile());

}   // namespace srreal
}   // namespace diffpy

// End of file.
