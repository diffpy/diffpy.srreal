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
* class PairQuantity -- brute force pair quantity calculator
*
* $Id$
*
*****************************************************************************/

#ifndef PAIRQUANTITY_HPP_INCLUDED
#define PAIRQUANTITY_HPP_INCLUDED

#include "BasePairQuantity.hpp"

namespace diffpy {
namespace srreal {

class BaseBondGenerator;

class PairQuantity : public BasePairQuantity
{
    public:

        // constructors
        PairQuantity();

        // methods
        virtual const QuantityType& eval(const StructureAdapter&);
        template <class T> const QuantityType& eval(const T&);
        virtual const QuantityType& value() const;

    protected:

        // methods
        virtual void init();
        virtual void resizeValue(size_t);
        virtual void resetValue();
        virtual void updateValue();
        virtual void configureBondGenerator(BaseBondGenerator&) { }
        virtual void addPairContribution(const BaseBondGenerator&) { }

        // data
        QuantityType mvalue;
        const StructureAdapter* mstructure;

};

// Template Public Methods ---------------------------------------------------

template <class T>
const QuantityType& PairQuantity::eval(const T& stru)
{
    std::auto_ptr<StructureAdapter> bstru(createPQAdapter(stru));
    return this->eval(*bstru);
}


}   // namespace srreal
}   // namespace diffpy

#endif  // PAIRQUANTITY_HPP_INCLUDED
