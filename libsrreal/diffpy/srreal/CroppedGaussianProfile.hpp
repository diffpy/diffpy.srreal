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
* class CroppedGaussianProfile -- Gaussian profile cropped to zero beyond
*     xboundhi and scaled so that its integrated area equals 1.
*     Registered as "croppedgaussian".
*
* $Id$
*
*****************************************************************************/

#ifndef CROPPEDGAUSSIANPROFILE_HPP_INCLUDED
#define CROPPEDGAUSSIANPROFILE_HPP_INCLUDED

#include <diffpy/srreal/GaussianProfile.hpp>

namespace diffpy {
namespace srreal {

class CroppedGaussianProfile : public GaussianProfile
{
    public:

        // constructors
        CroppedGaussianProfile();
        PeakProfile* create() const;
        PeakProfile* copy() const;

        // methods
        const std::string& type() const;
        double yvalue(double x, double fwhm) const;
        void setPrecision(double eps);

    private:

        // data
        double mscale;

};

}   // namespace srreal
}   // namespace diffpy

#endif  // CROPPEDGAUSSIANPROFILE_HPP_INCLUDED
