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
* class BasePairQuantity -- abstract base class for general pair
*     quantity calculator
*
* $Id$
*
***********************************************************************/

#ifndef BASEPAIRQUANTITY_HPP_INCLUDED
#define BASEPAIRQUANTITY_HPP_INCLUDED

#include <vector>
#include <boost/shared_ptr.hpp>

namespace diffpy {


typedef std::vector<double> QuantityType;
class BaseStructure;

class BasePairQuantity
{
    public:

        // constructors
        BasePairQuantity() { }
        BasePairQuantity(const BaseStructure&) { }
        virtual ~BasePairQuantity()  { }

        // methods
        virtual const QuantityType& getValue() = 0;
        virtual const QuantityType& getDerivative() = 0;
        virtual const BaseStructure& getStructure() const = 0;
        virtual void setStructure(const BaseStructure&) = 0;

};


}   // namespace diffpy

#endif  // BASEPAIRQUANTITY_HPP_INCLUDED
