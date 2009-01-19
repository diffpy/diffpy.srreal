/***********************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2008 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
************************************************************************
*
* class PairQuantity -- abstract base class for brute force
*     pair quantity calculator
*
* $Id$
*
***********************************************************************/

#ifndef PAIRQUANTITYSLOW_HPP_INCLUDED
#define PAIRQUANTITYSLOW_HPP_INCLUDED

#include "BasePairQuantity.hpp"

namespace diffpy {

class BaseBondIterator;

class PairQuantity : public BasePairQuantity
{
    public:

        // constructors
        PairQuantity();

        // methods
        virtual const QuantityType& eval(const BaseStructure&);
        template <class T> const QuantityType& eval(const T&);
        virtual const QuantityType& value() const;

    protected:

        // methods
        virtual void init();
        virtual void resizeValue(size_t);
        virtual void resetValue();
        virtual void updateValue();
        virtual void configureBondIterator(BaseBondIterator*) { }
        virtual void addPairContribution(const BaseBondIterator*) { }

        // data
        QuantityType mvalue;
        const BaseStructure* mstructure;

};

//////////////////////////////////////////////////////////////////////////////
// Definitions
//////////////////////////////////////////////////////////////////////////////

// template methods

template <class T>
const QuantityType& PairQuantity::eval(const T& stru)
{
    std::auto_ptr<BaseStructure> bstru(createPQAdaptor(stru));
    return this->eval(*bstru);
}


}   // namespace diffpy

#endif  // PAIRQUANTITYSLOW_HPP_INCLUDED
