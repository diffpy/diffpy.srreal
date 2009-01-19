#ifndef PAIRCOUNTER_HPP_INCLUDED
#define PAIRCOUNTER_HPP_INCLUDED

#include <memory>
#include "PairQuantity.hpp"

namespace diffpy {

class BaseBondIterator;

class PairCounter : public PairQuantity
{
    public:

        template <class T> int operator()(const T&);

    protected:

        // methods
        virtual void addPairContribution(const BaseBondIterator*);

};

//////////////////////////////////////////////////////////////////////////////
// Definitions
//////////////////////////////////////////////////////////////////////////////

// template methods

template <class T>
int PairCounter::operator()(const T& stru)
{
    this->eval(stru);
    int cnt = this->value().front();
    return cnt;
}


}   // namespace diffpy

#endif  // PAIRCOUNTER_HPP_INCLUDED
