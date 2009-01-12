/***********************************************************************
* $Id$
***********************************************************************/

#ifndef PDFCALCULATOR_H
#define PDFCALCULATOR_H

#include "profilecalculator.h"
#include "bonditerator.h"
#include "bondwidthcalculator.h"

#include "ObjCryst/Crystal.h"
#include "ObjCryst/Scatterer.h"
#include "ObjCryst/General.h"
#include "ObjCryst/Crystal.h"
#include "RefinableObj/RefinableObj.h" // From ObjCryst

namespace SrReal
{

/* Implementation of ProfileCalculator virtual class. This calculator is meant
 * for crystal systems where periodic boundary conditions are enforced on the
 * unit cell.
 *
 */

class PDFCalculator : public SrReal::ProfileCalculator
{
    public:

    PDFCalculator(
        SrReal::BondIterator& _bonditer,
        SrReal::BondWidthCalculator& _bwcalc);
    virtual ~PDFCalculator();

    /* Defined in ProfileCalculator
     * virtual void setScatType(const ObjCryst::RadiationType _rt);
     * virtual ObjCryst::RadiationType getScatType();
     * virtual const float* getCalculationPoints();
     * virtual size_t getNumPoints(); // The number of calculation points
     * virtual float getQmax();
     * virtual float getQmin();
     */

    /* Overloaded from ProfileCalculator */
     // Set the calculation points and determine the internal calculation points
     virtual void setCalculationPoints(
             const float* _rvals, const size_t _numpoints);
     // Set the maximum Q value
     virtual void setQmax(float val);
     // Set the minimum Q value
     virtual void setQmin(float val);

    // Get the RDF over the requested calculation points
    virtual float* getPDF(); 
    // Get the RDF over the requested calculation points
    virtual float* getRDF();

    private:

    /* Defined in ProfileCalculator
     * SrReal::BondIterator& bonditer;
     * SrReal::BondWidthCalculator& bwcalc;
     * ObjCryst::RadiationType radtype;
     * float *rvals;
     * size_t numpoints;
     * float qmin, qmax;
    */

    // Calculate the RDF over the internal calculation range. This is extended
    // beyond the requested calculation range so peaks centered outside of the
    // calculation range that overlap with the calculation range are included.
    void calculateRDF();
    // Calculate the PDF over the internal calculation range.
    void calculatePDF();

    // Calculate the average scattering power in the unit cell
    void calcAvgScatPow();
    // Get the scattering power of a pair of scatterers
    float getPairScatPow(SrReal::BondPair &bp);
    // Get the corrected profile
    float* getCorrectedProfile(float* df);
    // Apply termination ripples
    void applyTerminationRipples();
    // Setup the FFT for termination ripples
    void setupFFT();


    /* Refinable parameters */
    // These are accessible through the refinable parameter interface inherited
    // from RefinableObj
    float qbroad;
    float qdamp;
    float scale;

    /* Data */

    // RDF without resolution or qmax corrections
    float *rdf; 
    // PDF without resolution or qmax corrections
    float *pdf; 
    // array for computing fft for termination ripples
    double *ffta; 
    size_t fftlen;
    size_t qmaxidxlo;
    size_t qmaxidxhi;
    size_t qminidxhi;

    // For the internal calculation range
    size_t cnumpoints; 
    float crmin;
    float crmax;
    float cdr;

    // The average scattering power in the unit cell
    float bavg; 
    // The number of scatterers in the unit cell, calculated from occupancy
    float numscat; 

};

}
#endif
