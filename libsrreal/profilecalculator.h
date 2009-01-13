#ifndef PROFILECALCULATOR_H
#define PROFILECALCULATOR_H

#include "bonditerator.h"
#include "bondwidthcalculator.h"

#include "ObjCryst/Crystal.h"
#include "ObjCryst/Scatterer.h"
#include "ObjCryst/General.h"
#include "ObjCryst/Crystal.h"
#include "RefinableObj/RefinableObj.h" // From ObjCryst

namespace SrReal
{

/* Virtual base class for real space profile calculation. All PDF and RDF
 * calculators need this basic functionality
 */

class ProfileCalculator : public ObjCryst::RefinableObj
{
    public:

    ProfileCalculator(
        SrReal::BondIterator& _bonditer,
        SrReal::BondWidthCalculator& _bwcalc);
    virtual ~ProfileCalculator();

    virtual SrReal::BondIterator& getBondIterator();
    virtual SrReal::BondWidthCalculator& getBondWidthCalculator();

    // Calculation setup
    virtual void setScatType(const ObjCryst::RadiationType _radtype);
    virtual ObjCryst::RadiationType getScatType();
    virtual void setCalculationPoints(
            const float* _rvals, const size_t _numpoints); // copies _rvals
    virtual const float* getCalculationPoints();
    virtual size_t getNumPoints(); // The number of calculation points
    virtual void setQmax(float val);
    virtual float getQmax();
    virtual void setQmin(float val);
    virtual float getQmin();

    // Related to calculation
    virtual float* getPDF() = 0;
    virtual float* getRDF() = 0;

    protected:

    // Data necessary for the above functions
    SrReal::BondIterator& bonditer;
    SrReal::BondWidthCalculator& bwcalc;
    ObjCryst::RadiationType radtype;
    float *rvals;
    size_t numpoints;
    float qmin, qmax;

};

// Refinable parameter type for ProfileCalculator parameters
const ObjCryst::RefParType profilerefpartype(string("profilerefpartype"));

}
#endif
