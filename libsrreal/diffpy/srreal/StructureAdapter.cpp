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
* class StructureAdapter -- abstract base class for interfacing general
*     structure objects with srreal classes such as PairQuantity
*
* $Id$
*
*****************************************************************************/

#include <memory>
#include <diffpy/srreal/StructureAdapter.hpp>
#include <diffpy/srreal/BaseBondGenerator.hpp>

using namespace std;
using namespace diffpy::srreal;

// Public Methods ------------------------------------------------------------

double StructureAdapter::totalOccupancy() const
{
    auto_ptr<BaseBondGenerator> bnds(this->createBondGenerator());
    // This needs to sum over all atoms in expanded unit cell. 
    // This is ensured by looping over all bonds of atom 0 with
    // self-pairs included.
    bnds->includeSelfPairs(true);
    double total_occupancy = 0.0;
    for (bnds->rewind(); !bnds->finished(); bnds->next())
    {
        total_occupancy += this->siteOccupancy(bnds->site1());
    }
    return total_occupancy;
}


double StructureAdapter::siteOccupancy(int idx) const
{
    return 1.0;
}

// End of file
