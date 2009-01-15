#include "BaseBondIterator.hpp"
#include "BaseBondPair.hpp"
#include "BaseStructure.hpp"

using namespace std;
using namespace diffpy;


//////////////////////////////////////////////////////////////////////////////
// Public Methods
//////////////////////////////////////////////////////////////////////////////

int BaseStructure::countSites() const
{
    return 0;
}


BaseBondIterator* BaseStructure::createBondIterator() const
{
    BaseBondIterator* bnds = new BaseBondIterator(this);
    return bnds;
}

BaseBondPair* BaseStructure::createBondPair() const
{
    BaseBondPair* bp = new BaseBondPair();
    return bp;
}

