#include "PairCounter.hpp"

using namespace diffpy;


// protected methods

void PairCounter::addPairContribution(const BaseBondIterator* bnds)
{
    mvalue.front() += 1;
}
