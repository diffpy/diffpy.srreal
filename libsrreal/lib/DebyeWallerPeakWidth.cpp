/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Christopher Farrow, Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* class DebyeWallerPeakWidth -- peak width model based on FIXME paper by Jeong
*
* $Id$
*
*****************************************************************************/

#include <cmath>

#include <diffpy/srreal/DebyeWallerPeakWidth.hpp>
#include <diffpy/srreal/BaseBondGenerator.hpp>

using namespace std;
using namespace diffpy::srreal;

// Constructors --------------------------------------------------------------

BasePeakWidthModel* DebyeWallerPeakWidth::create() const
{
    BasePeakWidthModel* rv = new DebyeWallerPeakWidth();
    return rv;
}


BasePeakWidthModel* DebyeWallerPeakWidth::copy() const
{
    BasePeakWidthModel* rv = new DebyeWallerPeakWidth(*this);
    return rv;
}

// Public Methods ------------------------------------------------------------

bool DebyeWallerPeakWidth::operator==(const BasePeakWidthModel& other) const
{
    // This peak width model has no parameters, therefore we just need
    // to check if other is of the same type.
    if (this == &other)  return true;
    bool rv = dynamic_cast<const DebyeWallerPeakWidth*>(&other);
    return rv;
}


const string& DebyeWallerPeakWidth::type() const
{
    static const string rv = "debye-waller";
    return rv;
}


double DebyeWallerPeakWidth::calculate(const BaseBondGenerator& bnds) const
{
    double gauss_sigma_square = bnds.msd();
    double fwhm = 2 * sqrt(2 * M_LN2 * gauss_sigma_square);
    return fwhm;
}

// Registration --------------------------------------------------------------

bool reg_DebyeWallerPeakWidth = registerPeakWidthModel(DebyeWallerPeakWidth());

// End of file
