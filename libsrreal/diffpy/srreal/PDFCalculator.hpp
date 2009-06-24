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

        QuantityType getExtendedPDF() const;
        QuantityType getExtendedRDF() const;
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
        void setMaxExtension(double);
        const double& getMaxExtension() const;
        const double& getExtendedRmin() const;
        const double& getExtendedRmax() const;

        // PDF peak width configuration
        void setPeakWidthModel(const PeakWidthModel&);
        void setPeakWidthModel(const std::string& tp);
        const PeakWidthModel& getPeakWidthModel() const;

        // PDF profile configuration
        void setPeakProfile(const PeakProfile&);
        void setPeakProfile(const std::string& tp);
        const PeakProfile& getPeakProfile() const;
        void setPeakPrecision(double);
        double getPeakPrecision() const;

        // PDF baseline configuration
        void setBaseline(const PDFBaseline&);
        void setBaseline(const std::string& tp);
        const PDFBaseline& getBaseline() const;

        // PDF envelope functions
        // convenience functions for handling common envelopes
        void setScale(double);
        double getScale() const;
        void setQdamp(double);
        double getQdamp() const;
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

        // methods - PairQuantity overloads
        virtual void resetValue();
        virtual void configureBondGenerator(BaseBondGenerator&);
        virtual void addPairContribution(const BaseBondGenerator&);

    private:

        // methods - calculation specific
        const double& rextlo() const;
        const double& rexthi() const;
        double extTerminationRipples() const;
        double extPeakTails() const;
        int extloPoints() const;
        int exthiPoints() const;
        int ripplesloPoints() const;
        int rippleshiPoints() const;
        int rgridPoints() const;
        int extendedPoints() const;
        int totalPoints() const;
        int totalIndex(double r) const;

        // structure factors - fast lookup by site index
        const double& sfSite(int) const;
        double sfAverage() const;
        void cacheStructureData();
        void cacheRlimitsData();

        // data
        // configuration
        double mqmin;
        double mqmax;
        double mrstep;
        double mmaxextension;
        std::auto_ptr<PeakWidthModel> mpwmodel;
        std::auto_ptr<PeakProfile> mpeakprofile;
        std::auto_ptr<PDFBaseline> mbaseline;
        EnvelopeStorage menvelope;
        std::auto_ptr<ScatteringFactorTable> msftable;
        struct {
            std::vector<double> sfsite;
            double sfaverage;
            double totaloccupancy;
        } mstructure_cache;
        struct {
            double extendedrmin;
            double extendedrmax;
            double rextlow;
            double rexthigh;
        } mrlimits_cache;

};  // class PDFCalculator

}   // namespace srreal
}   // namespace diffpy

#endif  // PDFCALCULATOR_HPP_INCLUDED
