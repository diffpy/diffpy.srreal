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
* class BaseBondIterator -- semi-abstract class for iterator
*     over all atom pairs containing specified anchor atom.
*
* $Id$
*
*****************************************************************************/

#include <diffpy/srreal/BaseBondIterator.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>

using namespace std;
using namespace diffpy::srreal;

// Constructor ---------------------------------------------------------------

BaseBondIterator::BaseBondIterator(const StructureAdapter* stru)
{
    mstructure = stru;
    this->includeSelfPairs(false);
    this->selectAnchorSite(0);
    this->selectSiteRange(0, mstructure->countSites());
}

// Public Methods ------------------------------------------------------------

// loop control

void BaseBondIterator::rewind()
{
    msite_current = msite_first;
    this->skipSelfPair();
}


void BaseBondIterator::next()
{
    if (this->iterateSymmetry())  return;
    msite_current += 1;
    this->skipSelfPair();
}


bool BaseBondIterator::finished() const
{
    return msite_current >= msite_last;
}

// configuration

void BaseBondIterator::selectAnchorSite(int anchor)
{
    msite_anchor = anchor;
    this->setFinishedFlag();
}


void BaseBondIterator::selectSiteRange(int first, int last)
{
    msite_first = first;
    msite_last = last;
    this->setFinishedFlag();
}


void BaseBondIterator::includeSelfPairs(bool flag)
{
    minclude_self_pairs = flag;
}

// data query

const R3::Vector& BaseBondIterator::r0() const
{
    const R3::Vector& rv = mstructure->siteCartesianPosition(msite_anchor);
    return rv;
}


const R3::Vector& BaseBondIterator::r1() const
{
    const R3::Vector& rv = mstructure->siteCartesianPosition(msite_current);
    return rv;
}


double BaseBondIterator::distance() const
{
    double d = R3::distance(this->r0(), this->r1());
    return d;
}


const R3::Vector& BaseBondIterator::r01() const
{
    static R3::Vector rv;
    rv = this->r1() - this->r0();
    return rv;
}


double BaseBondIterator::msd0() const
{
    return 0.0;
}


double BaseBondIterator::msd1() const
{
    return 0.0;
}


double BaseBondIterator::msd() const
{
    return (this->msd0() + this->msd1());
}

// Protected Methods ---------------------------------------------------------

bool BaseBondIterator::iterateSymmetry()
{
    return false;
}

// Private Methods -----------------------------------------------------------

void BaseBondIterator::skipSelfPair()
{
    if (minclude_self_pairs)    return;
    if (msite_anchor == msite_current)    msite_current += 1;
}


void BaseBondIterator::setFinishedFlag()
{
    msite_current = msite_last;
}

// End of file
