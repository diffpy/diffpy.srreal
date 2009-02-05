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

using namespace std;
using namespace diffpy;
using namespace diffpy::srreal;

// Declaration of Local Helpers ----------------------------------------------

namespace {

void ensureNonNegative(const string& vname, double value);

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
    return 0.0; // FIXME
}


double PDFCalculator::rexthi() const
{
    return 0.0; // FIXME
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

}   // namespace

// End of file
