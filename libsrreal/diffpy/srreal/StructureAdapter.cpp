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
    double total_occupancy = 0.0;
    int cnt_sites = this->countSites();
    for (int i = 0; i < cnt_sites; ++i)
    {
        total_occupancy += this->siteOccupancy(i);
    }
    return total_occupancy;
}


double StructureAdapter::numberDensity() const
{
    return 0.0;
}


double StructureAdapter::siteOccupancy(int idx) const
{
    return 1.0;
}

// End of file
