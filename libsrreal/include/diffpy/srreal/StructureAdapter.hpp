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

namespace diffpy {
namespace srreal {

class BaseBondIterator;


class StructureAdapter
{
    public:

        // methods
        virtual int countSites() const = 0;
        virtual BaseBondIterator* createBondIterator() const = 0;

};


}   // namespace srreal
}   // namespace diffpy

#endif  // STRUCTUREADAPTER_HPP_INCLUDED
