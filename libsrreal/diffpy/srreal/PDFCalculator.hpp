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

#include <set>
#include <boost/shared_ptr.hpp>

#include <diffpy/srreal/PairQuantity.hpp>
#include <diffpy/srreal/PeakProfile.hpp>
#include <diffpy/srreal/PeakWidthModel.hpp>
#include <diffpy/srreal/PDFBaseline.hpp>
#include <diffpy/srreal/PDFEnvelope.hpp>
#include <diffpy/srreal/ScatteringFactorTable.hpp>

namespace diffpy {
namespace srreal {

class PDFCalculator : public PairQuantity
{
    public:

        // constructor
        PDFCalculator();

        // results
        QuantityType getPDF() const;
        QuantityType getRDF() const;
        QuantityType getRgrid() const;

        /// PDF on extended range to allow propagation of termination ripples
        QuantityType getExtendedPDF() const;
        /// RDF on extended range to allow propagation of termination ripples
        QuantityType getExtendedRDF() const;
        /// extended r-grid that allows propagation of termination ripples
        QuantityType getExtendedRgrid() const;

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
        /// maximum total extension of the r-range accounting for both
        /// termination ripples and peak tails
        void setMaxExtension(double);
        /// maximum total extension of the r-range accounting for both
        /// termination ripples and peak tails
        const double& getMaxExtension() const;
        /// lower bound for the r-range extended for termination ripples
        const double& getExtendedRmin() const;
        /// upper bound of the r-range extended for termination ripples
        const double& getExtendedRmax() const;

        // PDF peak width configuration
        void setPeakWidthModel(const PeakWidthModel&);
        void setPeakWidthModel(const std::string& tp);
        const PeakWidthModel& getPeakWidthModel() const;

        // PDF profile configuration
        void setPeakProfile(const PeakProfile&);
        void setPeakProfile(const std::string& tp);
        const PeakProfile& getPeakProfile() const;

        // PDF baseline configuration
        void setBaseline(const PDFBaseline&);
        void setBaseline(const std::string& tp);
        const PDFBaseline& getBaseline() const;

        // PDF envelope functions
        // application on an array
        QuantityType applyEnvelopes(const QuantityType& x, const QuantityType& y) const;
        // configuration of envelopes
        void addEnvelope(const PDFEnvelope&);
        void addEnvelope(const std::string& tp);
        void popEnvelope(const std::string& tp);
        const PDFEnvelope& getEnvelope(const std::string& tp) const;
        PDFEnvelope& getEnvelope(const std::string& tp);
        std::set<std::string> usedEnvelopeTypes() const;
        void clearEnvelopes();

        // access and configuration of scattering factors
        void setScatteringFactorTable(const ScatteringFactorTable&);
        void setScatteringFactorTable(const std::string& tp);
        const ScatteringFactorTable& getScatteringFactorTable() const;
        const std::string& getRadiationType() const;
        // scattering factors lookup
        double sfAtomType(const std::string&) const;

    protected:

        // types
        typedef std::map<std::string, boost::shared_ptr<PDFEnvelope> > EnvelopeStorage;

        // Attributes overload to direct visitors around data structures
        virtual void accept(diffpy::BaseAttributesVisitor& v);
        virtual void accept(diffpy::BaseAttributesVisitor& v) const;

        // PairQuantity overloads
        virtual void resetValue();
        virtual void configureBondGenerator(BaseBondGenerator&);
        virtual void addPairContribution(const BaseBondGenerator&);

    private:

        // methods - calculation specific
        /// complete lower bound extension of the calculated grid
        const double& rcalclo() const;
        /// complete upper bound extension of the calculated grid
        const double& rcalchi() const;
        /// r-range extension to allow propagation of termination ripples
        double extFromTerminationRipples() const;
        /// r-range extension to account for tails from out-of-range peaks
        double extFromPeakTails() const;
        /// number of data points in the complete extension of the lower bound
        int calcloPoints() const;
        /// number of data points in the complete extension of the upper bound
        int calchiPoints() const;
        /// number of points in the lower extension due to termination ripples
        int ripplesloPoints() const;
        /// number of points in the upper extension due to termination ripples
        int rippleshiPoints() const;
        /// number of points within the user requested r-grid
        int rgridPoints() const;
        /// number of points in the r-grid extended with termination ripples
        int extendedPoints() const;
        /// number of points in the complete calculated r-grid
        int calcPoints() const;
        /// index of a nearby point in the complete calculated r-grid
        int calcIndex(double r) const;

        // structure factors - fast lookup by site index
        /// effective scattering factor at a given site scaled by occupancy
        const double& sfSite(int) const;
        /// average scattering factor
        double sfAverage() const;
        void cacheStructureData();
        void cacheRlimitsData();

        // data
        // configuration
        double mqmin;
        double mqmax;
        double mrstep;
        double mmaxextension;
        boost::shared_ptr<PeakWidthModel> mpwmodel;
        boost::shared_ptr<PeakProfile> mpeakprofile;
        boost::shared_ptr<PDFBaseline> mbaseline;
        EnvelopeStorage menvelope;
        boost::shared_ptr<ScatteringFactorTable> msftable;
        struct {
            std::vector<double> sfsite;
            double sfaverage;
            double totaloccupancy;
        } mstructure_cache;
        struct {
            double extendedrmin;
            double extendedrmax;
            double rcalclow;
            double rcalchigh;
        } mrlimits_cache;

};  // class PDFCalculator

}   // namespace srreal
}   // namespace diffpy

#endif  // PDFCALCULATOR_HPP_INCLUDED
