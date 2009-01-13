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


#include "PairQuantity.hpp"
#include "BaseStructure.hpp"

using namespace std;
using namespace diffpy;


//////////////////////////////////////////////////////////////////////////////
// Constructors
//////////////////////////////////////////////////////////////////////////////


PairQuantity::PairQuantity()
{
    this->init();
    const BaseStructure& blank = this->getBlankStructure();
    this->setStructure(blank);
}


PairQuantity::PairQuantity(const BaseStructure& stru)
{
    this->init();
    this->setStructure(stru);
}


//////////////////////////////////////////////////////////////////////////////
// Public Methods
//////////////////////////////////////////////////////////////////////////////


const QuantityType& PairQuantity::getValue()
{
    this->updateValue();
    return mvalue;
}


const QuantityType& PairQuantity::getDerivative()
{
    this->updateValue(true);
    return mderivative;
}


const BaseStructure& PairQuantity::getStructure() const
{
    return *mstructure;
}


void PairQuantity::setStructure(const BaseStructure& stru)
{
    this->uncache();
    mstructure.reset(&stru);
}


//////////////////////////////////////////////////////////////////////////////
// Protected Methods
//////////////////////////////////////////////////////////////////////////////


void PairQuantity::init()
{ 
    this->uncache();
}


void PairQuantity::updateValue(bool derivate)
{
    mvalue.resize(1);
    mvalue[0] = 0.0;
    mvalue_cached = true;
    if (derivate)
    {
        mderivative.resize(1);
        mderivative[0] = 0.0;
        mderivative_cached = true;
    }
}


void PairQuantity::uncache()
{
    mvalue_cached = false;
    mderivative_cached = false;
}


//////////////////////////////////////////////////////////////////////////////
// Private Methods
//////////////////////////////////////////////////////////////////////////////


const BaseStructure& PairQuantity::getBlankStructure() const
{
    static BaseStructure blank;
    return blank;
};
