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
* class BaseBondGenerator -- semi-abstract class for generation
*     of all atom pairs containing specified anchor atom.
*
* $Id$
*
*****************************************************************************/

#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>
#include <diffpy/mathutils.hpp>

using namespace std;
using namespace diffpy::srreal;
using diffpy::mathutils::DOUBLE_MAX;
using diffpy::mathutils::SQRT_DOUBLE_EPS;

// Constructor ---------------------------------------------------------------

BaseBondGenerator::BaseBondGenerator(const StructureAdapter* stru)
{
    mstructure = stru;
    this->setRmin(0.0);
    this->setRmax(DOUBLE_MAX);
    this->selectAnchorSite(0);
    this->selectSiteRange(0, mstructure->countSites());
}

// Public Methods ------------------------------------------------------------

// loop control

void BaseBondGenerator::rewind()
{
    msite_current = msite_first;
    this->advanceWhileInvalid();
}


void BaseBondGenerator::next()
{
    this->getNextBond();
    this->advanceWhileInvalid();
}


bool BaseBondGenerator::finished() const
{
    return msite_current >= msite_last;
}

// configuration

void BaseBondGenerator::selectAnchorSite(int anchor)
{
    msite_anchor = anchor;
    this->setFinishedFlag();
}


void BaseBondGenerator::selectSiteRange(int first, int last)
{
    msite_first = first;
    msite_last = last;
    this->setFinishedFlag();
}


void BaseBondGenerator::setRmin(double rmin)
{
    if (rmin != mrmin)  this->setFinishedFlag();
    mrmin = rmin;
    this->checkIfRangeSet();
}


void BaseBondGenerator::setRmax(double rmax)
{
    if (rmax != mrmax)  this->setFinishedFlag();
    mrmax = rmax;
    this->checkIfRangeSet();
}

// data query

const double& BaseBondGenerator::getRmin() const
{
    return mrmin;
}


const double& BaseBondGenerator::getRmax() const
{
    return mrmax;
}


const int& BaseBondGenerator::site0() const
{
    return msite_anchor;
}


const int& BaseBondGenerator::site1() const
{
    return msite_current;
}


const R3::Vector& BaseBondGenerator::r0() const
{
    const R3::Vector& rv = mstructure->siteCartesianPosition(this->site0());
    return rv;
}


const R3::Vector& BaseBondGenerator::r1() const
{
    const R3::Vector& rv = mstructure->siteCartesianPosition(this->site1());
    return rv;
}


double BaseBondGenerator::distance() const
{
    double d = R3::distance(this->r0(), this->r1());
    return d;
}


const R3::Vector& BaseBondGenerator::r01() const
{
    static R3::Vector rv;
    rv = this->r1() - this->r0();
    return rv;
}


double BaseBondGenerator::msd0() const
{
    return 0.0;
}


double BaseBondGenerator::msd1() const
{
    return 0.0;
}


double BaseBondGenerator::msd() const
{
    return (this->msd0() + this->msd1());
}

// Protected Methods ---------------------------------------------------------

bool BaseBondGenerator::iterateSymmetry()
{
    return false;
}

// Private Methods -----------------------------------------------------------

void BaseBondGenerator::getNextBond()
{
    if (this->iterateSymmetry())  return;
    msite_current += 1;
    this->rewindSymmetry();
}


void BaseBondGenerator::advanceWhileInvalid()
{
    while (!this->finished() &&
            (this->bondOutOfRange() || this->atSelfPair()))
    {
        this->getNextBond();
    }
}


bool BaseBondGenerator::bondOutOfRange() const
{
    bool rv = false;
    if (mrangeset)
    {
        double d = this->distance();
        rv = (d < this->getRmin()) || (d > this->getRmax());
    }
    return rv;
}


void BaseBondGenerator::checkIfRangeSet()
{
    mrangeset = (this->getRmin() > 0.0) || (this->getRmax() != DOUBLE_MAX);
}


bool BaseBondGenerator::atSelfPair() const
{
    bool rv = (this->site0() == this->site1()) &&
        (this->distance() < SQRT_DOUBLE_EPS);
    return rv;
}


void BaseBondGenerator::setFinishedFlag()
{
    msite_current = msite_last;
}

// End of file
