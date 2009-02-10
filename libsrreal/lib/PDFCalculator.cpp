/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    
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

using namespace std;
using namespace diffpy;
using namespace diffpy::srreal;

// Declaration of Local Helpers ----------------------------------------------

namespace {

void ensureNonNegative(const string& vname, double value);
double maxUii(const StructureAdapter*);

}   // namespace

// Public Methods ------------------------------------------------------------

// results

const QuantityType& PDFCalculator::getPDF() const
{
    return mpdf;
}


const QuantityType& PDFCalculator::getRDF() const
{
    return mrdf;
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

// R-range configuration

void PDFCalculator::setRmin(double rmin)
{
    ensureNonNegative("Rmin", rmin);
    mrmin = rmin;
}


const double& PDFCalculator::getRmin() const
{
    return mrmin;
}


void PDFCalculator::setRmax(double rmax)
{
    ensureNonNegative("Rmax", rmax);
    mrmax = rmax;
}


const double& PDFCalculator::getRmax() const
{
    return mrmax;
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

/*
void PDFCalculator::setPeakWidthModel(const PeakWidthModel& pwm)
{
    mpwmodel.reset(pwm.copy());
}


const PeakWidthModel& PDFCalculator::getPeakWidthModel() const;
{
    assert(mpwmodel.get());
    return *mpwmodel;
}
        void setScatteringFactorTable(const ScatteringFactorTable&);
        const ScatteringFactorTable& getScatteringFactorTable() const;
        void setRadiationType(const std::string&);
        const std::string& getRadiationType() const;
        // scattering factors lookup
        double sfAtomType(const std::string&) const;
*/

// Protected Methods ---------------------------------------------------------

// PairQuantity overloads

/*
void init();
void addPairContribution(const BaseBondIterator*);
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
        const R3::Matrix U = stru->siteCartesianUij(i);
        for (int k = 0; k < R3::Ndim; k++)
        {
            if (U(k,k) > rv)   rv = U(k,k);
        }
    }
    return rv;
}

}   // namespace

// End of file
