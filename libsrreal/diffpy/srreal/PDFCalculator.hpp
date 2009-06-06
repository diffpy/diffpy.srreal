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
* class PDFCalculator -- brute force PDF calculator
*
* $Id$
*
*****************************************************************************/

#ifndef PDFCALCULATOR_HPP_INCLUDED
#define PDFCALCULATOR_HPP_INCLUDED

#include <memory>
#include <set>

#include <diffpy/srreal/PairQuantity.hpp>
#include <diffpy/srreal/PeakWidthModel.hpp>
#include <diffpy/srreal/ScatteringFactorTable.hpp>

namespace diffpy {
namespace srreal {

class PDFEnvelope;  // FIXME -- temporary, to be removed

class PDFCalculator : public PairQuantity
{
    public:

        // constructor
        PDFCalculator();

        // results
        QuantityType getPDF() const;
        QuantityType getRDF() const;
        QuantityType getRgrid() const;

        // Q-range configuration
        void setQmin(double);
        const double& getQmin() const;
        void setQmax(double);
        const double& getQmax() const;
        QuantityType applyBandPassFilter(const QuantityType& a) const;

        // R-range configuration
        virtual void setRmin(double);
        virtual void setRmax(double);
        void setRstep(double);
        const double& getRstep() const;

        // PDF peak width configuration
        void setPeakWidthModel(const PeakWidthModel&);
        const PeakWidthModel& getPeakWidthModel() const;

        // PDF envelope functions
        // application on an array
        QuantityType applyEnvelopes(const QuantityType&) const;
        // configuration of envelopes
        void addEnvelope(const PDFEnvelope&);
        void addEnvelope(const std::string& tp);
        const PDFEnvelope& getEnvelope(const std::string& tp) const;
        PDFEnvelope& getEnvelope(const std::string& tp);
        std::set<std::string> usedEnvelopeTypes() const;
        void clearEnvelopes();

        // scattering factors configuration
        void setScatteringFactorTable(const ScatteringFactorTable&);
        const ScatteringFactorTable& getScatteringFactorTable() const;
        void setRadiationType(const std::string&);
        const std::string& getRadiationType() const;
        // scattering factors lookup
        double sfAtomType(const std::string&) const;

    protected:

        // methods - PairQuantity overloads
        // FIXME
//      virtual void addPairContribution(const BaseBondGenerator&);

        // methods - calculation specific
        double rextlo() const;
        double rexthi() const;
        double extMagnitude() const;
        int extloPoints() const;
        int exthiPoints() const;
        int rgridPoints() const;
        int totalPoints() const;
        int totalIndex(double r) const;

        // structure factors - fast lookup by site index
        const double& sfSite(int) const;
        double sfAverage() const;
        void update_msfsite();

        // data
        // configuration
        double mqmin;
        double mqmax;
        double mrstep;
        std::auto_ptr<PeakWidthModel> mpwmodel;
        std::auto_ptr<ScatteringFactorTable> msftable;
        std::vector<double> msfsite;
};

// Public Template Methods ---------------------------------------------------

// FIXME
#if 0

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

    // Calculate the RDF over the internal calculation range from scratch. This
    // is extended beyond the requested calculation range so peaks centered
    // outside of the calculation range that overlap with the calculation range
    // are included.
    void calculateRDF();
    // Add contributions to the RDF from a ObjCryst::ScatteringComponentList
    void buildRDF(const ObjCryst::ScatteringComponentList &scl, float pref);
    // Update the RDF. This checks for changes in the parameters and then either
    // reshapes the RDF or recalculates it.
    void updateRDF();
    // Reshape the RDF by adding changes in scattering components.
    void reshapeRDF();
    // Calculate the PDF over the internal calculation range.
    void calculatePDF();

    // Add a gaussian to the rdf
    void addGaussian(float d, float sigma, float amp);
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
    // Calculate the diameter of the sphere that can encompass the crystal.
    float phaseDiameter() const;

    /* Clocks for tracking changes */
    // Compare this clock with the crystal lattice
    ObjCryst::RefinableObjClock latclock;
    // Compare this clock with the scattering list clock
    ObjCryst::RefinableObjClock slclock;
    // Compare this clock with the bond width calculator
    ObjCryst::RefinableObjClock bwclock;
    // These compare with the scatterers
    std::vector<ObjCryst::RefinableObjClock> scatclocks;

    // flag for recalculation
    bool recalc;

    // handle to the crystal
    const ObjCryst::Crystal &crystal;

    /* Refinable parameters */
    // These are accessible through the refinable parameter interface inherited
    // from RefinableObj
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

    // Ids for accessing saved crystal parameters
    size_t lastsave;
    size_t cursave;

};
#endif

}   // namespace srreal
}   // namespace diffpy

#endif  // PDFCALCULATOR_HPP_INCLUDED
