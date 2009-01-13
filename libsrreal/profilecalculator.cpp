/***********************************************************************
* $Id$
***********************************************************************/

#include "profilecalculator.h"

#include "ObjCryst/General.h"

SrReal::ProfileCalculator::
ProfileCalculator(
    SrReal::BondIterator& _bonditer,
    SrReal::BondWidthCalculator& _bwcalc) : 
        ObjCryst::RefinableObj(), bonditer(_bonditer), bwcalc(_bwcalc)
{
    rvals = NULL;
    numpoints = 0;
    qmin = qmax = 0.0;
    radtype = ObjCryst::RAD_XRAY;
}

SrReal::ProfileCalculator::
~ProfileCalculator()
{
    if( rvals != NULL )
    {
        delete [] rvals;
    }

}

SrReal::BondIterator& 
SrReal::ProfileCalculator::
getBondIterator()
{
    return bonditer;
}

SrReal::BondWidthCalculator& 
SrReal::ProfileCalculator::
getBondWidthCalculator()
{
    return bwcalc;
}

void
SrReal::ProfileCalculator::
setScatType(ObjCryst::RadiationType _radtype)
{
    radtype = _radtype;
}

ObjCryst::RadiationType 
SrReal::ProfileCalculator::
getScatType()
{
    return radtype;
}

void
SrReal::ProfileCalculator::
setCalculationPoints(
    const float* _rvals, const size_t _numpoints)
{
    // make space for the copies
    if( rvals != NULL )
    {
        delete [] rvals;
    }

    rvals = new float[numpoints];
    numpoints = _numpoints;
}

// This may return NULL!
const float*
SrReal::ProfileCalculator::
getCalculationPoints()
{
    return rvals;
}

size_t
SrReal::ProfileCalculator::
getNumPoints()
{
    return numpoints;
}


void
SrReal::ProfileCalculator::
setQmax(float val)
{
    qmax = (val > 0 ? val : 0);
}

float
SrReal::ProfileCalculator::
getQmax()
{
    return qmax;
}

void
SrReal::ProfileCalculator::
setQmin(float val)
{
    qmin = (val > 0 ? val : 0);
}

float
SrReal::ProfileCalculator::
getQmin()
{
    return qmin;
}


