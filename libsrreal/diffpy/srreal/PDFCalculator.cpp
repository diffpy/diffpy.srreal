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
#include <diffpy/srreal/LinearBaseline.hpp>
#include <diffpy/srreal/GaussianProfile.hpp>
#include <diffpy/mathutils.hpp>

using namespace std;
using namespace diffpy::srreal;

// Declaration of Local Helpers ----------------------------------------------

namespace {

void ensureNonNegative(const string& vname, double value);
double maxUii(const StructureAdapter*);

/// Default peak precision was obtained from the tunePeakPrecision.py script
/// and it was tuned to give average zero slope in the difference curve
/// between pdffit2 and PDFCalculator results.
const double DEFAULT_PEAK_PRECISION = 3.33e-6;

}   // namespace

// Constructor ---------------------------------------------------------------

PDFCalculator::PDFCalculator()
{
    using diffpy::mathutils::SQRT_DOUBLE_EPS;
    // initialize mstructure_cache
    mstructure_cache.sfaverage = 0.0;
    mstructure_cache.totaloccupancy = 0.0;
    // initialize mrlimits_cache
    mrlimits_cache.extendedrmin = 0.0;
    mrlimits_cache.extendedrmax = 0.0;
    mrlimits_cache.rcalclow = 0.0;
    mrlimits_cache.rcalchigh = 0.0;
    // default configuration
    this->setPeakWidthModel("jeong");
    GaussianProfile pkf;
    pkf.setPrecision(DEFAULT_PEAK_PRECISION);
    this->setPeakProfile(pkf);
    this->setBaseline("linear");
    this->setScatteringFactorTable("SFTperiodictableXray");
    this->setRmax(10.0);
    this->setRstep(0.01);
    this->setQmin(0.0);
    this->setQmax(0.0);
    this->setMaxExtension(10.0);
    // envelopes
    this->addEnvelope(ScaleEnvelope());
    this->addEnvelope(QResolutionEnvelope());
    // attributes
    this->registerDoubleAttribute("qmin", this,
            &PDFCalculator::getQmin, &PDFCalculator::setQmin);
    this->registerDoubleAttribute("qmax", this,
            &PDFCalculator::getQmax, &PDFCalculator::setQmax);
    this->registerDoubleAttribute("rmin", this,
            &PDFCalculator::getRmin, &PDFCalculator::setRmin);
    this->registerDoubleAttribute("rmax", this,
            &PDFCalculator::getRmax, &PDFCalculator::setRmax);
    this->registerDoubleAttribute("rstep", this,
            &PDFCalculator::getRstep, &PDFCalculator::setRstep);
    this->registerDoubleAttribute("maxextension", this,
            &PDFCalculator::getMaxExtension, &PDFCalculator::setMaxExtension);
    this->registerDoubleAttribute("extendedrmin", this,
            &PDFCalculator::getExtendedRmin);
    this->registerDoubleAttribute("extendedrmax", this,
            &PDFCalculator::getExtendedRmax);
}

// Public Methods ------------------------------------------------------------

// results

QuantityType PDFCalculator::getPDF() const
{
    QuantityType pdf = this->getExtendedPDF();
    assert(this->ripplesloPoints() + this->rippleshiPoints() <= (int) pdf.size());
    pdf.erase(pdf.end() - this->rippleshiPoints(), pdf.end());
    pdf.erase(pdf.begin(), pdf.begin() + this->ripplesloPoints());
    return pdf;
}


QuantityType PDFCalculator::getRDF() const
{
    QuantityType rdf = this->getExtendedRDF();
    assert(this->rippleshiPoints() + this->ripplesloPoints() <= (int) rdf.size());
    rdf.erase(rdf.end() - this->rippleshiPoints(), rdf.end());
    rdf.erase(rdf.begin(), rdf.begin() + this->ripplesloPoints());
    return rdf;
}


QuantityType PDFCalculator::getRgrid() const
{
    QuantityType rv(this->rgridPoints());
    QuantityType::iterator ri = rv.begin();
    for (int i = 0; ri != rv.end(); ++i, ++ri)
    {
        *ri = this->getRmin() + i * this->getRstep();
    }
    return rv;
}


QuantityType PDFCalculator::getExtendedPDF() const
{
    // we need a full range PDF to apply termination ripples correctly
    QuantityType pdf_ext(this->extendedPoints());
    QuantityType rdf_ext = this->getExtendedRDF();
    QuantityType rgrid_ext = this->getExtendedRgrid();
    assert(pdf_ext.size() == rdf_ext.size());
    assert(pdf_ext.size() == rgrid_ext.size());
    const PDFBaseline& baseline = this->getBaseline();
    QuantityType::iterator pdfi = pdf_ext.begin();
    QuantityType::const_iterator rdfi = rdf_ext.begin();
    QuantityType::const_iterator ri = rgrid_ext.begin();
    for (; pdfi != pdf_ext.end(); ++pdfi, ++rdfi, ++ri)
    {
        *pdfi = (*ri == 0.0) ? (0.0) :
            (*rdfi / *ri + baseline(*ri));
    }
    // fixme - factor out baseline application to a separate method
    QuantityType pdf1 = this->applyEnvelopes(rgrid_ext, pdf_ext);
    QuantityType pdf2 = this->applyBandPassFilter(pdf1);
    return pdf2;
}


QuantityType PDFCalculator::getExtendedRDF() const
{
    QuantityType rdf(this->extendedPoints());
    const double& totocc = mstructure_cache.totaloccupancy;
    double sfavg = this->sfAverage();
    double rdf_scale = (totocc * sfavg == 0.0) ? 0.0 :
        1.0 / (totocc * sfavg * sfavg);
    QuantityType::iterator iirdf = rdf.begin();
    QuantityType::const_iterator iival, iival_last;
    iival = this->value().begin() +
        this->calcloPoints() - this->ripplesloPoints();
    iival_last = this->value().end() - this->calchiPoints()
        + this->rippleshiPoints();
    assert(iival >= this->value().begin());
    assert(iival_last <= this->value().end());
    assert(rdf.size() == size_t(iival_last - iival));
    for (; iirdf != rdf.end(); ++iival, ++iirdf)
    {
        *iirdf = *iival * rdf_scale;
    }
    return rdf;
}


QuantityType PDFCalculator::getExtendedRgrid() const
{
    QuantityType rv(this->extendedPoints());
    QuantityType::iterator ri = rv.begin();
    // make sure exact value of rmin will be in the extended grid
    for (int i = -1 * this->ripplesloPoints(); ri != rv.end(); ++i, ++ri)
    {
        *ri = this->getRmin() + i * this->getRstep();
    }
    assert(rv.empty() || rv.front() >= this->getExtendedRmin());
    assert(rv.empty() || rv.back() <= this->getExtendedRmax());
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
    const double infinite_Qmax = 1e6;
    if (0 < this->getQmax() && this->getQmax() < infinite_Qmax)
    {
        bandPassFilter(rv.begin(), rv.end(),
                this->getRstep(), this->getQmin(), this->getQmax());
    }
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


void PDFCalculator::setMaxExtension(double maxext)
{
    mmaxextension = max(0.0, maxext);
}


const double& PDFCalculator::getMaxExtension() const
{
    return mmaxextension;
}


const double& PDFCalculator::getExtendedRmin() const
{
    return mrlimits_cache.extendedrmin;
}


const double& PDFCalculator::getExtendedRmax() const
{
    return mrlimits_cache.extendedrmax;
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
    // If peak profile already exists, copy its precision data
    if (mpeakprofile.get())  *pkf = *mpeakprofile;
    this->setPeakProfile(*pkf);
}


const PeakProfile& PDFCalculator::getPeakProfile() const
{
    assert(mpeakprofile.get());
    return *mpeakprofile;
}

// PDF baseline configuration

void PDFCalculator::setBaseline(const PDFBaseline& baseline)
{
    if (mbaseline.get() == &baseline)  return;
    mbaseline.reset(baseline.copy());
}


void PDFCalculator::setBaseline(const std::string& tp)
{
    auto_ptr<PDFBaseline> pbl(createPDFBaseline(tp));
    this->setBaseline(*pbl);
}


const PDFBaseline& PDFCalculator::getBaseline() const
{
    assert(mbaseline.get());
    return *mbaseline;
}

// PDF envelope methods

QuantityType PDFCalculator::applyEnvelopes(
        const QuantityType& x, const QuantityType& y) const
{
    assert(x.size() == y.size());
    QuantityType z = y;
    EnvelopeStorage::const_iterator evit;
    for (evit = menvelope.begin(); evit != menvelope.end(); ++evit)
    {
        PDFEnvelope& fenvelope = *(evit->second);
        QuantityType::const_iterator xi = x.begin();
        QuantityType::iterator zi = z.begin();
        for (; xi != x.end(); ++xi, ++zi)
        {
            *zi *= fenvelope(*xi);
        }
    }
    return z;
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


void PDFCalculator::popEnvelope(const string& tp)
{
    menvelope.erase(tp);
}


const PDFEnvelope& PDFCalculator::getEnvelope(const string& tp) const
{
    // call non-constant method
    PDFEnvelope& rv = const_cast<PDFCalculator*>(this)->getEnvelope(tp);
    return rv;
}


PDFEnvelope& PDFCalculator::getEnvelope(const string& tp)
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

// Attributes overloads

void PDFCalculator::accept(diffpy::BaseAttributesVisitor& v)
{
    mpwmodel->accept(v);
    mpeakprofile->accept(v);
    mbaseline->accept(v);
    // PDF envelopes
    EnvelopeStorage::iterator evit;
    for (evit = menvelope.begin(); evit != menvelope.end(); ++evit)
    {
        evit->second->accept(v);
    }
    // finally call standard accept
    this->diffpy::Attributes::accept(v);
}

void PDFCalculator::accept(diffpy::BaseAttributesVisitor& v) const
{
    mpwmodel->accept(v);
    mpeakprofile->accept(v);
    mbaseline->accept(v);
    // PDF envelopes
    EnvelopeStorage::const_iterator evit;
    for (evit = menvelope.begin(); evit != menvelope.end(); ++evit)
    {
        evit->second->accept(v);
    }
    // finally call standard accept
    this->diffpy::Attributes::accept(v);
}

// PairQuantity overloads

void PDFCalculator::resetValue()
{
    // calcPoints requires that structure and rlimits data are cached.
    this->cacheStructureData();
    this->cacheRlimitsData();
    // when applicable, configure linear baseline
    double numdensity = mstructure->numberDensity();
    if (numdensity > 0 && this->getBaseline().type() == "linear")
    {
        LinearBaseline bl =
            dynamic_cast<const LinearBaseline&>(this->getBaseline());
        bl.setSlope(-4 * M_PI * numdensity);
        this->setBaseline(bl);
    }
    this->resizeValue(this->calcPoints());
    this->PairQuantity::resetValue();
}


void PDFCalculator::configureBondGenerator(BaseBondGenerator& bnds)
{
    bnds.setRmin(this->rcalclo());
    bnds.setRmax(this->rcalchi());
}


void PDFCalculator::addPairContribution(const BaseBondGenerator& bnds)
{
    int summationscale = (bnds.site0() == bnds.site1()) ? 1 : 2;
    double sfprod = this->sfSite(bnds.site0()) * this->sfSite(bnds.site1());
    double peakscale = sfprod * mstructure->siteMultiplicity(bnds.site0());
    double fwhm = this->getPeakWidthModel().calculate(bnds);
    const PeakProfile& pkf = this->getPeakProfile();
    double dist = bnds.distance();
    double xlo = dist + pkf.xboundlo(fwhm);
    double xhi = dist + pkf.xboundhi(fwhm);
    int i = max(0, this->calcIndex(xlo));
    int ilast = min(this->calcPoints(), this->calcIndex(xhi) + 1);
    double x0 = this->rcalclo();
    assert(ilast <= int(mvalue.size()));
    for (; i < ilast; ++i)
    {
        double x = x0 + i * this->getRstep() - dist;
        double y = pkf.yvalue(x, fwhm);
        mvalue[i] += summationscale * peakscale * y;
    }
}

// calculation specific

const double& PDFCalculator::rcalclo() const
{
    return mrlimits_cache.rcalclow;
}


const double& PDFCalculator::rcalchi() const
{
    return mrlimits_cache.rcalchigh;
}


double PDFCalculator::extFromTerminationRipples() const
{
    // number of termination ripples for extending the r-range
    const int nripples = 6;
    // extension due to termination ripples
    double rv = (this->getQmax() > 0.0) ?
        (nripples*2*M_PI / this->getQmax()) : 0.0;
    return rv;
}


double PDFCalculator::extFromPeakTails() const
{
    double maxmsd = 2 * maxUii(mstructure);
    double maxfwhm = this->getPeakWidthModel().calculateFromMSD(maxmsd);
    const PeakProfile& pkf = this->getPeakProfile();
    double xleft = fabs(pkf.xboundlo(maxfwhm));
    double xright = fabs(pkf.xboundhi(maxfwhm));
    double rv = max(xleft, xright);
    return rv;
}


int PDFCalculator::calcloPoints() const
{
    int npts;
    npts = int(floor((this->getRmin() - this->rcalclo()) / this->getRstep()));
    return npts;
}


int PDFCalculator::calchiPoints() const
{
    // evaluate all with respect to rmin
    int npts;
    npts = int(ceil((this->rcalchi() - this->getRmin()) / this->getRstep()));
    npts -= this->rgridPoints();
    return npts;
}


int PDFCalculator::ripplesloPoints() const
{
    int npts = int(floor((this->getRmin() - this->getExtendedRmin()) /
                this->getRstep()));
    return npts;
}


int PDFCalculator::rippleshiPoints() const
{
    // evaluate all with respect to rmin
    int npts = int(ceil((this->getExtendedRmax() - this->getRmin()) /
                this->getRstep()));
    npts -= this->rgridPoints();
    return npts;
}


int PDFCalculator::rgridPoints() const
{
    int npts;
    npts = int(ceil((this->getRmax() - this->getRmin()) / this->getRstep()));
    return npts;
}


int PDFCalculator::extendedPoints() const
{
    int npts = this->ripplesloPoints() + this->rgridPoints() +
        this->rippleshiPoints();
    return npts;
}


int PDFCalculator::calcPoints() const
{
    int npts;
    npts = this->calcloPoints() + this->rgridPoints() + this->calchiPoints();
    return npts;
}


int PDFCalculator::calcIndex(double r) const
{
    int npts;
    npts = int(ceil((r - this->rcalclo()) / this->getRstep()));
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
        mstructure_cache.sfsite[i] = this->sfAtomType(smbl) *
            mstructure->siteOccupancy(i);
    }
    // sfaverage
    double totocc = mstructure->totalOccupancy();
    double totsf = 0.0;
    for (int i = 0; i < cntsites; ++i)
    {
        totsf += this->sfSite(i) * mstructure->siteMultiplicity(i);
    }
    mstructure_cache.sfaverage = (totocc == 0.0) ? 0.0 : (totsf / totocc);
    // totaloccupancy
    mstructure_cache.totaloccupancy = totocc;
}


void PDFCalculator::cacheRlimitsData()
{
    // obtain extension magnitudes and rescale to fit maximum extension
    double ext_ripples = this->extFromTerminationRipples();
    double ext_pktails = this->extFromPeakTails();
    double ext_total = ext_ripples + ext_pktails;
    if (ext_total > this->getMaxExtension())
    {
        double sc = this->getMaxExtension() / ext_total;
        ext_ripples *= sc;
        ext_pktails *= sc;
        ext_total = this->getMaxExtension();
    }
    // r-range extended by termination ripples:
    mrlimits_cache.extendedrmin = this->getRmin() - ext_ripples;
    mrlimits_cache.extendedrmin = max(0.0, mrlimits_cache.extendedrmin);
    mrlimits_cache.extendedrmax = this->getRmax() + ext_ripples;
    // complete calculation range, extended for both ripples and peak tails
    mrlimits_cache.rcalclow = this->getRmin() - ext_total;
    mrlimits_cache.rcalclow = max(0.0, mrlimits_cache.rcalclow);
    mrlimits_cache.rcalchigh = this->getRmax() + ext_total;
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
