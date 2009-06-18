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
#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/srreal/R3linalg.hpp>
#include <diffpy/srreal/PDFUtils.hpp>
#include <diffpy/srreal/ScaleEnvelope.hpp>
#include <diffpy/srreal/QResolutionEnvelope.hpp>
#include <diffpy/mathutils.hpp>

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
    using diffpy::mathutils::SQRT_DOUBLE_EPS;
    // default configuration
    this->setPeakWidthModel("jeong");
    this->setPeakProfile("gauss");
    this->setPeakPrecision(SQRT_DOUBLE_EPS);
    this->setScatteringFactorTable("SFTperiodictableXray");
    this->setRmax(10.0);
    this->setRstep(0.01);
    this->setQmax(0.0);
}

// Public Methods ------------------------------------------------------------

// results

QuantityType PDFCalculator::getPDF() const
{
    QuantityType pdf(this->rgridPoints());
    QuantityType rdf = this->getRDF();
    QuantityType rgrid = this->getRgrid();
    assert(pdf.size() == rdf.size() && pdf.size() == rgrid.size());
    PDFBaseLine baseline(mstructure_cache.numberdensity);
    QuantityType::iterator pdfi = pdf.begin();
    QuantityType::const_iterator rdfi = rdf.begin();
    QuantityType::const_iterator ri = rgrid.begin();
    for (; pdfi != pdf.end(); ++pdfi, ++rdfi, ++ri)
    {
        *pdfi = (*ri == 0.0) ? (0.0) :
            (*rdfi / *ri + baseline(*ri));
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
    const double& totocc = mstructure_cache.totaloccupancy;
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
    if (mpwmodel.get() == &pwm)   return;
    mpwmodel.reset(pwm.copy());
}


void PDFCalculator::setPeakWidthModel(const string& tp)
{
    auto_ptr<PeakWidthModel> ppwm(createPeakWidthModel(tp));
    this->setPeakWidthModel(*ppwm);
}


const PeakWidthModel& PDFCalculator::getPeakWidthModel() const
{
    assert(mpwmodel.get());
    return *mpwmodel;
}

// PDF peak profile configuration

void PDFCalculator::setPeakProfile(const PeakProfile& pkf)
{
    if (mpeakprofile.get() == &pkf)  return;
    mpeakprofile.reset(pkf.copy());
}


void PDFCalculator::setPeakProfile(const string& tp)
{
    auto_ptr<PeakProfile> pkf(createPeakProfile(tp));
    this->setPeakProfile(*pkf);
}


const PeakProfile& PDFCalculator::getPeakProfile() const
{
    assert(mpeakprofile.get());
    return *mpeakprofile;
}


void PDFCalculator::setPeakPrecision(double eps)
{
    if (this->getPeakProfile().getPrecision() == eps)  return;
    auto_ptr<PeakProfile> npkf(this->getPeakProfile().copy());
    npkf->setPrecision(eps);
    this->setPeakProfile(*npkf);
}


const double& PDFCalculator::getPeakPrecision() const
{
    const double& rv = this->getPeakProfile().getPrecision();
    return rv;
}

// PDF envelope methods

void PDFCalculator::setScale(double scale)
{
    ScaleEnvelope envelope;
    envelope.setScale(scale);
    this->addEnvelope(envelope);
}


const double& PDFCalculator::getScale() const
{
    const ScaleEnvelope& envelope =
        dynamic_cast<const ScaleEnvelope&>(this->getEnvelope("scale"));
    return envelope.getScale();
}


void PDFCalculator::setQdamp(double qdamp)
{
    QResolutionEnvelope envelope;
    envelope.setQdamp(qdamp);
    this->addEnvelope(envelope);
}


const double& PDFCalculator::getQdamp() const
{
    const QResolutionEnvelope& envelope =
        dynamic_cast<const QResolutionEnvelope&>(
                this->getEnvelope("qresolution"));
    return envelope.getQdamp();
}


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
    if (msftable.get() == &sft)   return;
    msftable.reset(sft.copy());
}


void PDFCalculator::setScatteringFactorTable(const string& tp)
{
    auto_ptr<ScatteringFactorTable> sft(createScatteringFactorTable(tp));
    this->setScatteringFactorTable(*sft);
}


const ScatteringFactorTable& PDFCalculator::getScatteringFactorTable() const
{
    assert(msftable.get());
    return *msftable;
}


const string& PDFCalculator::getRadiationType() const
{
    const string& tp = this->getScatteringFactorTable().radiationType();
    return tp;
}


double PDFCalculator::sfAtomType(const string& smbl) const
{
    const ScatteringFactorTable& sft = this->getScatteringFactorTable();
    double rv = sft.lookup(smbl);
    return rv;
}



// Protected Methods ---------------------------------------------------------

// PairQuantity overloads

void PDFCalculator::resetValue()
{
    this->resizeValue(this->totalPoints());
    this->PairQuantity::resetValue();
    this->cacheStructureData();
}


void PDFCalculator::configureBondGenerator(BaseBondGenerator& bnds)
{
    bnds.setRmin(this->rextlo());
    bnds.setRmax(this->rexthi());
}


void PDFCalculator::addPairContribution(const BaseBondGenerator& bnds)
{
    int summationscale = (bnds.site0() == bnds.site1()) ? 1 : 2;
    double sfprod = this->sfSite(bnds.site0()) * this->sfSite(bnds.site1());
    double fwhm = this->getPeakWidthModel().calculate(bnds);
    const PeakProfile& pkf = this->getPeakProfile();
    double dist = bnds.distance();
    double xlo = dist + pkf.xboundlo(fwhm);
    double xhi = dist + pkf.xboundhi(fwhm);
    int i = max(0, this->totalIndex(xlo));
    int ilast = min(this->totalPoints(), this->totalIndex(xhi) + 1);
    double x0 = this->rextlo();
    assert(ilast <= int(mvalue.size()));
    for (; i < ilast; ++i)
    {
        double x = x0 + i * this->getRstep() - dist;
        double y = pkf.y(x, fwhm);
        mvalue[i] += summationscale * sfprod * y;
    }
}


// calculation specific

double PDFCalculator::rextlo() const
{
    double rxlo = this->getRmin() - this->extMagnitude();
    if (rxlo < 0.0)     rxlo = 0.0;
    return rxlo;
}


double PDFCalculator::rexthi() const
{
    double rxhi = this->getRmax() + this->extMagnitude();
    return rxhi;
}


double PDFCalculator::extMagnitude() const
{
    // number of ripples for extending the r-range
    const int nripples = 6;
    // extension due to termination ripples
    double ext_ripples = (this->getQmax() > 0.0) ?
        (nripples*2*M_PI / this->getQmax()) : 0.0;
    // FIXME - use xboundlo, etc. for ext_pkwidth
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
    return npts;
}


const double& PDFCalculator::sfSite(int siteidx) const
{
    assert(0 <= siteidx && siteidx < int(mstructure_cache.sfsite.size()));
    return mstructure_cache.sfsite[siteidx];
}


double PDFCalculator::sfAverage() const
{
    return mstructure_cache.sfaverage;
}


void PDFCalculator::cacheStructureData()
{
    int cntsites = mstructure->countSites();
    // sfsite
    mstructure_cache.sfsite.resize(cntsites);
    for (int i = 0; i < cntsites; ++i)
    {
        const string& smbl = mstructure->siteAtomType(i);
        mstructure_cache.sfsite[i] = this->sfAtomType(smbl);
    }
    // sfaverage
    double totocc = mstructure->totalOccupancy();
    double totsf = 0.0;
    for (int i = 0; i < cntsites; ++i) 
    {
        totsf += this->sfSite(i) *
            mstructure->siteOccupancy(i) * mstructure->siteMultiplicity(i);
    }
    mstructure_cache.sfaverage = (totocc == 0.0) ? 0.0 : (totsf / totocc);
    // totaloccupancy
    mstructure_cache.totaloccupancy = totocc;
    // numberdensity
    mstructure_cache.numberdensity = mstructure->numberDensity();
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
