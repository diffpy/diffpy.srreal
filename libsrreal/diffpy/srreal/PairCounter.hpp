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
* class PairCounter -- concrete counter of pairs in a structure.
*
* $Id$
*
*****************************************************************************/

#ifndef PAIRCOUNTER_HPP_INCLUDED
#define PAIRCOUNTER_HPP_INCLUDED

#include <diffpy/srreal/PairQuantity.hpp>

namespace diffpy {
namespace srreal {

class BaseBondGenerator;

class PairCounter : public PairQuantity
{
    public:

        // methods
        template <class T> int operator()(const T&);

    protected:

        // methods
        virtual void addPairContribution(const BaseBondGenerator&);

};

// Public Template Methods ---------------------------------------------------

template <class T>
int PairCounter::operator()(const T& stru)
{
    this->eval(stru);
    int cnt = this->value().front();
    return cnt;
}


}   // namespace srreal
}   // namespace diffpy

#endif  // PAIRCOUNTER_HPP_INCLUDED
