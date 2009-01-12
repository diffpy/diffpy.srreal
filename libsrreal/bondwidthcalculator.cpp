/***********************************************************************
* $Id$
***********************************************************************/

#include <string>
#include "RefinableObj/RefinableObj.h" // From ObjCryst
#include "bondwidthcalculator.h"

/* BondWidthCalculator */
SrReal::BondWidthCalculator::
BondWidthCalculator() : ObjCryst::RefinableObj() {}

SrReal::BondWidthCalculator::
~BondWidthCalculator() {}

float
SrReal::BondWidthCalculator::
calculate(SrReal::BondPair& bp) 
{

    float sigma;
    sigma = bp.getSC1()->mpScattPow->GetBiso();
    sigma += bp.getSC2()->mpScattPow->GetBiso();

    return sqrt(sigma/(8*M_PI*M_PI));

}

/* JeongBWCalculator */

SrReal::JeongBWCalculator::
JeongBWCalculator()
{
    delta1 = delta2 = 0.0;

    ResetParList();

    // Delete the reference pars explicitly in the destructor since qbroad is
    // "borrowed".
    SetDeleteRefParInDestructor(false);

    /* Create the RefinablePar objects for delta1 and delta2 */
    // delta1
    {
    ObjCryst::RefinablePar* tmp = new ObjCryst::RefinablePar("delta1", &delta1, 0.0, 1.0, 
        &SrReal::bwrefpartype, ObjCryst::REFPAR_DERIV_STEP_ABSOLUTE, 
        false, false, true, false, 1.0, 1);
    tmp->AssignClock(mClockMaster);
    AddPar(tmp);
    }

    // delta2
    {
    ObjCryst::RefinablePar* tmp = new ObjCryst::RefinablePar("delta2", &delta2, 0.0, 1.0, 
        &SrReal::bwrefpartype, ObjCryst::REFPAR_DERIV_STEP_ABSOLUTE, 
        false, false, true, false, 1.0, 1);
    tmp->AssignClock(mClockMaster);
    AddPar(tmp);
    }
}

SrReal::JeongBWCalculator::
~JeongBWCalculator() 
{

    // Delete the "Owned" refinable parameters explicitly.
    {
    ObjCryst::RefinablePar* tmp = &GetPar(&delta1);
    delete tmp;
    }
    {
    ObjCryst::RefinablePar* tmp = &GetPar(&delta2);
    delete tmp;
    }

}

float
SrReal::JeongBWCalculator::
calculate(SrReal::BondPair& bp)
{

    // Only isotropic scattering factors are supported right now.  Only one of
    // delta1 or delta2 should be used. This is not enforced.
    float r, sigma, corr;
    float qbroad = GetPar("qbroad").GetValue();
    sigma = SrReal::BondWidthCalculator::calculate(bp);
    r = bp.getDistance();
    corr = 1.0 - delta1/r - delta2/(r*r) + pow(qbroad*r, 2);
    if(corr > 0)
    {
        sigma *= sqrt(corr);
    }
    return sigma;
}
