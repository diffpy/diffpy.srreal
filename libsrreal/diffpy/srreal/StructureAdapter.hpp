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

#ifndef STRUCTUREADAPTER_HPP_INCLUDED
#define STRUCTUREADAPTER_HPP_INCLUDED

#include <diffpy/srreal/R3linalg.hpp>

namespace diffpy {
namespace srreal {

class BaseBondGenerator;

class StructureAdapter
{
    public:

        virtual ~StructureAdapter()  { }
        // methods
        virtual BaseBondGenerator* createBondGenerator() const = 0;
        virtual int countSites() const = 0;
        virtual const R3::Vector& siteCartesianPosition(int idx) const = 0;
        virtual bool siteAnisotropy(int idx) const = 0;
        virtual const R3::Matrix& siteCartesianUij(int idx) const = 0;
};


}   // namespace srreal
}   // namespace diffpy

#endif  // STRUCTUREADAPTER_HPP_INCLUDED
