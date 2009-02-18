/*****************************************************************************
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
******************************************************************************
*
* class PairQuantity -- brute force pair quantity calculator
*
* $Id$
*
*****************************************************************************/

#include <memory>

#include <diffpy/srreal/PairQuantity.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>
#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/mathutils.hpp>

using namespace std;
using namespace diffpy::srreal;

// Constructor ---------------------------------------------------------------

PairQuantity::PairQuantity() : BasePairQuantity()
{
    using diffpy::mathutils::DOUBLE_MAX;
    this->setRmin(0.0);
    this->setRmax(DOUBLE_MAX);
    this->resizeValue(1);
}

// Public Methods ------------------------------------------------------------

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


void PairQuantity::setRmin(double rmin)
{
    mrmin = rmin;
}


const double& PairQuantity::getRmin() const
{
    return mrmin;
}


void PairQuantity::setRmax(double rmax)
{
    mrmax = rmax;
}


const double& PairQuantity::getRmax() const
{
    return mrmax;
}

// Protected Methods ---------------------------------------------------------

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
    auto_ptr<BaseBondGenerator> bnds;
    bnds.reset(mstructure->createBondGenerator());
    this->configureBondGenerator(*bnds);
    int nsites = mstructure->countSites();
    for (int i0 = 0; i0 < nsites; ++i0)
    {
        bnds->selectAnchorSite(i0);
        bnds->selectSiteRange(0, i0 + 1);
        for (bnds->rewind(); !bnds->finished(); bnds->next())
        {
            this->addPairContribution(*bnds);
        }
    }
}


void PairQuantity::configureBondGenerator(BaseBondGenerator& bnds)
{
    bnds.setRmin(this->getRmin());
    bnds.setRmax(this->getRmax());
}

// End of file
