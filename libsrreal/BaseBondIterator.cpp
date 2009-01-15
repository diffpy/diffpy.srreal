#include "BaseBondIterator.hpp"
#include "BaseBondPair.hpp"
#include "BaseStructure.hpp"

using namespace std;
using namespace diffpy;

//////////////////////////////////////////////////////////////////////////////
// Constructor
//////////////////////////////////////////////////////////////////////////////

BaseBondIterator::BaseBondIterator(const BaseStructure* stru)
{
    mstructure = stru;
    mbond_pair.reset(stru->createBondPair());
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
}


void BaseBondIterator::next()
{
    if (this->iterateSymmetry())  return;
    msite_current += 1;
}


bool BaseBondIterator::finished() const
{
    return msite_current < msite_last;
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

// data query

const BaseBondPair& BaseBondIterator::getBondPair() const
{
    return *mbond_pair;
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

void BaseBondIterator::setFinishedFlag()
{
    msite_current = msite_last;
}
