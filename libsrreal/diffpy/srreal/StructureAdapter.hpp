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
class PairQuantity;

/// @class StructureAdapter
/// @brief abstract adaptor to structure data needed by
/// PairQuantity calculator

class StructureAdapter
{
    public:

        virtual ~StructureAdapter()  { }

        // methods

        /// factory for creating compatible BondGenerator instance.
        virtual BaseBondGenerator* createBondGenerator() const = 0;

        /// number of independent sites in the structure, before
        /// any symmetry expansion.
        virtual int countSites() const = 0;

        /// total number of atoms in the structure unit accounting
        /// for possibly fractional occupancies.
        virtual double totalOccupancy() const;

        /// number density in the structure model or 0 when not defined.
        virtual double numberDensity() const;

        /// symbol for element or ion at the independent site @param idx
        virtual const std::string& siteAtomType(int idx) const;

        /// Cartesian coordinates of the independent site @param idx
        virtual const R3::Vector& siteCartesianPosition(int idx) const = 0;

        /// multiplicity of the independent site @param idx in the structure
        virtual double siteMultiplicity(int idx) const;

        /// site occupancy at the independent site @param idx
        virtual double siteOccupancy(int idx) const;

        /// flag for anisotropic atom displacements at independent site
        /// @param idx
        virtual bool siteAnisotropy(int idx) const = 0;

        /// tensor of atom displacement parameters converted in
        /// Cartesian coordinate system at independent site @param idx
        virtual const R3::Matrix& siteCartesianUij(int idx) const = 0;

        /// this method allows custom special configuration for a concrete 
        /// pair of StructureAdapter and PairQuantity objects.
        virtual void customPQConfig(PairQuantity& pq) const  { }
};


}   // namespace srreal
}   // namespace diffpy

#endif  // STRUCTUREADAPTER_HPP_INCLUDED
