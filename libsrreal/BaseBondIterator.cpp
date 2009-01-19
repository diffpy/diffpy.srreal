#include "BaseBondIterator.hpp"
#include "StructureAdapter.hpp"

using namespace std;
using namespace diffpy;

//////////////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////////////

BaseBondIterator::BaseBondIterator(const StructureAdapter* stru)
{
    mstructure = stru;
    this->includeSelfPairs(false);
    this->selectAnchorSite(0);
    this->selectSiteRange(0, mstructure->countSites());
}

//////////////////////////////////////////////////////////////////////////////
// Public Methods
//////////////////////////////////////////////////////////////////////////////

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

double BaseBondIterator::distance() const
{
    double d = R3::distance(this->r0(), this->r1());
    return d;
}

//////////////////////////////////////////////////////////////////////////////
// Protected Methods
//////////////////////////////////////////////////////////////////////////////

bool BaseBondIterator::iterateSymmetry()
{
    return false;
}

//////////////////////////////////////////////////////////////////////////////
// Private Methods
//////////////////////////////////////////////////////////////////////////////

void BaseBondIterator::skipSelfPair()
{
    if (minclude_self_pairs)    return;
    if (msite_anchor == msite_current)    msite_current += 1;
}


void BaseBondIterator::setFinishedFlag()
{
    msite_current = msite_last;
}
