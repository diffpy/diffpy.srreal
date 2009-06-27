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

#ifndef GAUSSIANPROFILE_HPP_INCLUDED
#define GAUSSIANPROFILE_HPP_INCLUDED

#include <diffpy/srreal/PeakProfile.hpp>

namespace diffpy {
namespace srreal {

class GaussianProfile : public PeakProfile
{
    public:

        // constructors
        GaussianProfile();
        PeakProfile* create() const;
        PeakProfile* copy() const;

        // methods
        const std::string& type() const;
        double yvalue(double x, double fwhm) const;
        double xboundlo(double fwhm) const;
        double xboundhi(double fwhm) const;
        void setPrecision(double eps);

    protected:

        // data
        double mhalfboundrel;

};

}   // namespace srreal
}   // namespace diffpy

#endif  // GAUSSIANPROFILE_HPP_INCLUDED
