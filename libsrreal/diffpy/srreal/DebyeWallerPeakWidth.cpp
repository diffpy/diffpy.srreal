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
* class DebyeWallerPeakWidth -- peak width calculated assuming independent
*     thermal vibrations of atoms forming a pair.
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

PeakWidthModel* DebyeWallerPeakWidth::create() const
{
    PeakWidthModel* rv = new DebyeWallerPeakWidth();
    return rv;
}


PeakWidthModel* DebyeWallerPeakWidth::copy() const
{
    PeakWidthModel* rv = new DebyeWallerPeakWidth(*this);
    return rv;
}

// Public Methods ------------------------------------------------------------

bool DebyeWallerPeakWidth::operator==(const PeakWidthModel& other) const
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
    double msdval = bnds.msd();
    return this->DebyeWallerPeakWidth::calculateFromMSD(msdval);
}


double DebyeWallerPeakWidth::calculateFromMSD(double msdval) const
{
    const double tofwhm = 2 * sqrt(2 * M_LN2);
    double fwhm = tofwhm * sqrt(msdval);
    return fwhm;
}

// Registration --------------------------------------------------------------

bool reg_DebyeWallerPeakWidth = registerPeakWidthModel(DebyeWallerPeakWidth());

// End of file
