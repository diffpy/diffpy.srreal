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
#include "StructureAdapter.hpp"
#include "BaseBondIterator.hpp"

using namespace std;
using namespace diffpy;

//////////////////////////////////////////////////////////////////////////////
// Constructors
//////////////////////////////////////////////////////////////////////////////

PairQuantity::PairQuantity()
{
    this->init();
}

//////////////////////////////////////////////////////////////////////////////
// Public Methods
//////////////////////////////////////////////////////////////////////////////

const QuantityType& PairQuantity::eval(const StructureAdapter& stru)
{
    mstructure = &stru;
    this->updateValue();
    mstructure = NULL;
    return this->value();
}


const QuantityType& PairQuantity::value() const
{
    return mvalue;
}

//////////////////////////////////////////////////////////////////////////////
// Protected Methods
//////////////////////////////////////////////////////////////////////////////

void PairQuantity::init()
{ 
    this->resizeValue(1);
}


void PairQuantity::resizeValue(size_t sz)
{
    mvalue.resize(sz);
}


void PairQuantity::resetValue()
{
    fill(mvalue.begin(), mvalue.end(), 0.0);
}

        
void PairQuantity::updateValue()
{
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
            this->addPairContribution(bnds.get());
        }
    }
}
