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


#include <memory>

#include "PairQuantity.hpp"
#include "BaseStructure.hpp"
#include "BaseBondIterator.hpp"
#include "BaseBondPair.hpp"

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
    mderivative_needed = false;
    this->updateValue();
    return mvalue;
}


const QuantityType& PairQuantity::getDerivative()
{
    mderivative_needed = true;
    this->updateValue();
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


void PairQuantity::resizeValue(size_t sz)
{
    mvalue.resize(sz);
}


void PairQuantity::updateValue()
{
    bool iscached = mvalue_cached && !this->structureChanged() &&
        (mderivative_cached || !mderivative_needed);
    if (iscached)    return;
    this->resetValue();
    auto_ptr<BaseBondIterator> bnds;
    bnds.reset(mstructure->createBondIterator());
    this->configureBondIterator(bnds.get());
    int nsites = mstructure->countSites();
    for (int i0 = 0; i0 < nsites; ++i0)
    {
        bnds->selectAnchorSite(i0);
        bnds->selectSiteRange(0, i0 + 1);
        for (bnds->rewind(); !bnds->finished(); bnds->next())
        {
            const BaseBondPair& bp = bnds->getBondPair();
            this->addPairContribution(bp);
        }
    }
    mvalue_cached = true;
    mderivative_cached = mderivative_needed;
}


void PairQuantity::resetValue()
{
    fill(mvalue.begin(), mvalue.end(), 0.0);
    if (mderivative_needed)
    {
        mderivative.resize(mvalue.size());
        fill(mderivative.begin(), mderivative.end(), 0.0);
    }
}
        

void PairQuantity::uncache()
{
    mvalue_cached = false;
    mderivative_cached = false;
}


bool PairQuantity::structureChanged() const
{
    // base implementation recalculates everything from scratch every time.
    return true;
}

//////////////////////////////////////////////////////////////////////////////
// Private Methods
//////////////////////////////////////////////////////////////////////////////

const BaseStructure& PairQuantity::getBlankStructure() const
{
    static BaseStructure blank;
    return blank;
};
