/***********************************************************************
* $Id$
***********************************************************************/

#include "bondwidthcalculator.h"

/* BondWidthCalculator */
SrReal::BondWidthCalculator::
BondWidthCalculator() {}

SrReal::BondWidthCalculator::
~BondWidthCalculator() {}

float
SrReal::BondWidthCalculator::
calculate(SrReal::BondPair &bp) 
{

    static float sigma;
    sigma = bp.getSC1()->mpScattPow->GetBiso();
    sigma += bp.getSC2()->mpScattPow->GetBiso();

    return sqrt(sigma/(8*M_PI*M_PI));

}

/* JeongBWCalculator */

SrReal::JeongBWCalculator::
JeongBWCalculator()
{
    delta1 = delta2 = 0.0;
}

SrReal::JeongBWCalculator::
~JeongBWCalculator() {}

float
SrReal::JeongBWCalculator::
calculate(SrReal::BondPair &bp)
{

    // Only isotropic scattering factors are supported right now.  Only one of
    // delta1 or delta2 should be used. This is not enforced.
    static float r, sigma, corr;
    sigma = SrReal::BondWidthCalculator::calculate(bp);
    r = bp.getDistance();
    corr = 1.0 - delta1/r - delta2/(r*r);
    if(corr > 0)
    {
        sigma *= sqrt(corr);
    }
    return sigma;
}
