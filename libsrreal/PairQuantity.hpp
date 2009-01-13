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


class PairQuantity : public BasePairQuantity
{
    public:

        // constructors
        PairQuantity();
        PairQuantity(const BaseStructure&);

        // methods
        virtual const std::vector<double>& getValue();
        virtual const std::vector<double>& getDerivative();
        virtual const BaseStructure& getStructure() const;
        virtual void setStructure(const BaseStructure&);

    protected:

        // types
        typedef boost::shared_ptr<const BaseStructure> StructurePtr;

        // methods
        virtual void init();
        virtual void updateValue(bool derivate=false);
        void uncache();

        // data
        StructurePtr mstructure;
        QuantityType mvalue;
        bool mvalue_cached;
        QuantityType mderivative;
        bool mderivative_cached;

    private:

        // methods
        const BaseStructure& getBlankStructure() const;
};


}   // namespace diffpy

#endif  // PAIRQUANTITYSLOW_HPP_INCLUDED
