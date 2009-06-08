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
* class PDFCalculator -- concrete counter of pairs in a structure.
*
* $Id$
*
*****************************************************************************/

#include <stdexcept>
#include <sstream>
#include <cmath>
#include <cassert>

#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>
#include <diffpy/srreal/R3linalg.hpp>
#include <diffpy/srreal/PDFUtils.hpp>

using namespace std;
using namespace diffpy::srreal;

// Declaration of Local Helpers ----------------------------------------------

namespace {

void ensureNonNegative(const string& vname, double value);
double maxUii(const StructureAdapter*);


class PDFBaseLine
{
    private:

        // data
        double mslope;

    public:

        PDFBaseLine(double num_density)
        {
            mslope = (num_density > 0) ? (-4 * M_PI * num_density) : 0.0;
        }

        double operator()(const double& ri) const
        {
            return mslope * ri;
        }

};


}   // namespace

// Constructor ---------------------------------------------------------------

PDFCalculator::PDFCalculator() : PairQuantity()
{
    auto_ptr<PeakWidthModel> pwm(createPeakWidthModel("jeong"));
    this->setPeakWidthModel(*pwm);
}

// Public Methods ------------------------------------------------------------

// results

QuantityType PDFCalculator::getPDF() const
{
    QuantityType pdf(this->rgridPoints());
    QuantityType rdf = this->getRDF();
    QuantityType rgrid = this->getRgrid();
    assert(pdf.size() == rdf.size() && pdf.size() == rgrid.size());
    PDFBaseLine baseline(mstructure->numberDensity());
    QuantityType::iterator pdfi = pdf.begin();
    QuantityType::const_iterator rdfi = rdf.begin();
    QuantityType::const_iterator ri = rgrid.begin();
    for (; pdfi != pdf.end(); ++pdfi, ++rdfi, ++ri)
    {
        *pdfi = (*ri == 0.0) ? (0.0) :
            (*pdfi / *ri + baseline(*ri));
    }
    // fixme - factor out baseline application to a separate method
    QuantityType& pdf0 = pdf;
    QuantityType pdf1 = this->applyEnvelopes(pdf0);
    QuantityType pdf2 = this->applyBandPassFilter(pdf1);
    return pdf2;
}


QuantityType PDFCalculator::getRDF() const
{
    QuantityType rdf(this->rgridPoints());
    double totocc = mstructure->totalOccupancy();
    double sfavg = this->sfAverage();
    double rdf_scale = (totocc * sfavg == 0.0) ? 0.0 :
        1.0 / (totocc * sfavg * sfavg);
    QuantityType::iterator iirdf = rdf.begin();
    QuantityType::const_iterator iival =
        this->value().begin() + this->extloPoints();
    QuantityType::const_iterator iival_last = iival + rdf.size();
    assert(iival <= this->value().end());
    assert(iival_last <= this->value().end());
    for (; iival != iival_last; ++iival, ++iirdf)
    {
        *iirdf = *iival * rdf_scale;
    }
    return rdf;
}


QuantityType PDFCalculator::getRgrid() const
{
    QuantityType rv;
    int npts = this->rgridPoints();
    rv.resize(npts);
    for (int i = 0; i < npts; ++i)
    {
        rv[i] = this->getRmin() + i * this->getRstep();
    }
    return rv;
}

// Q-range configuration

void PDFCalculator::setQmin(double qmin)
{
    ensureNonNegative("Qmin", qmin);
    mqmin = qmin;
}


const double& PDFCalculator::getQmin() const
{
    return mqmin;
}


void PDFCalculator::setQmax(double qmax)
{
    ensureNonNegative("Qmax", qmax);
    mqmax = qmax;
}


const double& PDFCalculator::getQmax() const
{
    return mqmax;
}


QuantityType PDFCalculator::applyBandPassFilter(const QuantityType& a) const
{
    QuantityType rv(a);
    bandPassFilter(rv.begin(), rv.end(),
            this->getRstep(), this->getQmin(), this->getQmax());
    return rv;
}


// R-range configuration

void PDFCalculator::setRmin(double rmin)
{
    ensureNonNegative("Rmin", rmin);
    this->PairQuantity::setRmin(rmin);
}


void PDFCalculator::setRmax(double rmax)
{
    ensureNonNegative("Rmax", rmax);
    this->PairQuantity::setRmax(rmax);
}


void PDFCalculator::setRstep(double rstep)
{
    if (rstep <= 0)
    {
        const char* emsg = "Rstep must be positive.";
        throw invalid_argument(emsg);
    }
    mrstep = rstep;
}


const double& PDFCalculator::getRstep() const
{
    return mrstep;
}

// PDF peak width configuration

void PDFCalculator::setPeakWidthModel(const PeakWidthModel& pwm)
{
    mpwmodel.reset(pwm.copy());
}


const PeakWidthModel& PDFCalculator::getPeakWidthModel() const
{
    assert(mpwmodel.get());
    return *mpwmodel;
}

// PDF envelope methods

QuantityType PDFCalculator::applyEnvelopes(const QuantityType& a) const
{
    QuantityType rv = a;
    QuantityType rgrid = this->getRgrid();
    EnvelopeStorage::const_iterator evit;
    for (evit = menvelope.begin(); evit != menvelope.end(); ++evit)
    {
        PDFEnvelope& fenvelope = *(evit->second);
        QuantityType::iterator ri = rgrid.begin();
        QuantityType::iterator fi = rv.begin();
        for (; ri != rgrid.end(); ++ri, ++fi)
        {
            *fi *= fenvelope(*ri);
        }
    }
    return rv;
}


void PDFCalculator::addEnvelope(const PDFEnvelope& envlp)
{
    menvelope[envlp.type()].reset(envlp.copy());
}


void PDFCalculator::addEnvelope(const string& tp)
{
    // this throws invalid_argument for invalid type
    PDFEnvelope* penvlp = createPDFEnvelope(tp);
    // we get here only when createPDFEnvelope was successful
    menvelope[penvlp->type()].reset(penvlp);
}


const PDFEnvelope& PDFCalculator::getEnvelope(const std::string& tp) const
{
    // call non-constant method
    PDFEnvelope& rv = const_cast<PDFCalculator*>(this)->getEnvelope(tp);
    return rv;
}


PDFEnvelope& PDFCalculator::getEnvelope(const std::string& tp)
{
    if (!menvelope.count(tp))  this->addEnvelope(tp);
    PDFEnvelope& rv = *(menvelope[tp]);
    return rv;
}


set<string> PDFCalculator::usedEnvelopeTypes() const
{
    set<string> rv;
    EnvelopeStorage::const_iterator evit;
    for (evit = menvelope.begin(); evit != menvelope.end(); ++evit)
    {
        rv.insert(rv.end(), evit->first);
    }
    return rv;
}


void PDFCalculator::clearEnvelopes()
{
    menvelope.clear();
}

// access and configuration of scattering factors

void PDFCalculator::setScatteringFactorTable(const ScatteringFactorTable& sft)
{
    msftable.reset(sft.copy());
    this->update_msfsite();
}


const ScatteringFactorTable& PDFCalculator::getScatteringFactorTable() const
{
    return *msftable;
}


void PDFCalculator::setRadiationType(const string& tp)
{
    ScatteringFactorTable* p = createScatteringFactorTable(tp);
    msftable.reset(p);
    this->update_msfsite();
}


const string& PDFCalculator::getRadiationType() const
{
    assert(msftable.get());
    const string& tp = msftable->type();
    return tp;
}


double PDFCalculator::sfAtomType(const string& smbl) const
{
    assert(msftable.get());
    double rv = msftable->lookup(smbl);
    return rv;
}



// Protected Methods ---------------------------------------------------------

// PairQuantity overloads

/*
void PDFCalculator::addPairContribution(const BaseBondGenerator& bnds)
{
}
*/

// calculation specific

double PDFCalculator::rextlo() const
{
    double rxlo = this->getRmin() - this->extMagnitude();
    if (rxlo < 0.0)     rxlo = 0.0;
    return rxlo;
}


double PDFCalculator::rexthi() const
{
    double rxhi = this->getRmin() + this->extMagnitude();
    return rxhi;
}


double PDFCalculator::extMagnitude() const
{
    // number of ripples for extending the r-range
    const int nripples = 6;
    // extension due to termination ripples
    double ext_ripples = (this->getQmax() > 0.0) ?
        (nripples*2*M_PI / this->getQmax()) : 0.0;
    // extension due to peak width
    const int n_gaussian_sigma = 5;
    double ext_pkwidth = n_gaussian_sigma * sqrt(maxUii(mstructure));
    // combine extensions to get the total magnitude
    double ext_total = sqrt(pow(ext_ripples, 2) + pow(ext_pkwidth, 2));
    return ext_total;
}


int PDFCalculator::extloPoints() const
{
    int npts;
    npts = int(floor((this->getRmin() - this->rextlo()) / this->getRstep()));
    return npts;
}


int PDFCalculator::exthiPoints() const
{
    // evaluate all with respect to rmin
    int npts;
    npts = int(ceil((this->rexthi() - this->getRmin()) / this->getRstep()));
    npts -= this->rgridPoints();
    return npts;
}


int PDFCalculator::rgridPoints() const
{
    int npts;
    npts = int(ceil((this->getRmax() - this->getRmin()) / this->getRstep()));
    return npts;
}


int PDFCalculator::totalPoints() const
{
    int npts;
    npts = this->extloPoints() + this->rgridPoints() + this->exthiPoints();
    return npts;
}


int PDFCalculator::totalIndex(double r) const
{
    int npts;
    npts = int(ceil((r - this->rextlo()) / this->getRstep()));
    assert(0 <= npts && npts < this->totalPoints());
    return npts;
}


const double& PDFCalculator::sfSite(int siteidx) const
{
    assert(0 <= siteidx && siteidx < int(msfsite.size()));
    return msfsite[siteidx];
}


double PDFCalculator::sfAverage() const
{
    double totsf = 0.0;
    int cntsites = mstructure->countSites();
    for (int i = 0; i < cntsites; ++i) 
    {
        totsf += this->sfSite(i) *
            mstructure->siteOccupancy(i) * mstructure->siteMultiplicity(i);
    }
    double totocc = mstructure->totalOccupancy();
    double rv = (totocc == 0.0) ? 0.0 : (totsf / totocc);
    return rv;
}


void PDFCalculator::update_msfsite()
{
    int cntsites = mstructure->countSites();
    msfsite.resize(cntsites);
    for (int i = 0; i < cntsites; ++i)
    {
        const string& smbl = mstructure->siteAtomType(i);
        msfsite[i] = this->sfAtomType(smbl);
    }
}

// Local Helpers -------------------------------------------------------------

namespace {

void ensureNonNegative(const string& vname, double value)
{
    if (value < 0.0)
    {
        stringstream emsg;
        emsg << vname << " cannot be negative.";
        throw invalid_argument(emsg.str());
    }
}


double maxUii(const StructureAdapter* stru)
{
    if (!stru)  return 0.0;
    double rv = 0.0;
    for (int i = 0; i < stru->countSites(); ++i)
    {
        const R3::Matrix& U = stru->siteCartesianUij(i);
        for (int k = 0; k < R3::Ndim; k++)
        {
            if (U(k,k) > rv)   rv = U(k,k);
        }
    }
    return rv;
}

}   // namespace

// End of file
