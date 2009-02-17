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
* class VR3Structure -- trivial structure representation.
* class VR3Adapter -- concrete StructureAdapter for VR3Structure
* class VR3BondGenerator -- concrete BaseBondGenerator for VR3Structure
*
* $Id$
*
*****************************************************************************/

#ifndef VR3STRUCTURE_HPP_INCLUDED
#define VR3STRUCTURE_HPP_INCLUDED

#include <vector>

#include <diffpy/srreal/R3linalg.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>
#include <diffpy/srreal/BaseBondGenerator.hpp>

namespace diffpy {
namespace srreal {

typedef std::vector<R3::Vector> VR3Structure;


class VR3Adapter : public StructureAdapter
{
    friend class VR3BondGenerator;
    public:

        // constructors
        VR3Adapter(const VR3Structure& vr3s);

        // methods
        virtual int countSites() const;
        virtual const R3::Vector& siteCartesianPosition(int idx) const;
        virtual bool siteAnisotropy(int idx) const;
        virtual const R3::Matrix& siteCartesianUij(int idx) const;

        virtual BaseBondGenerator* createBondGenerator() const;

    private:

        // data
        const VR3Structure* mvr3structure;
};


class VR3BondGenerator : public BaseBondGenerator
{
    public:

        // constructors
        VR3BondGenerator(const VR3Adapter*);

        // methods
        virtual const R3::Vector& r0() const;
        virtual const R3::Vector& r1() const;

    private:
        // data
        const VR3Structure* mvr3structure;
};


//////////////////////////////////////////////////////////////////////////////
// Definitions
//////////////////////////////////////////////////////////////////////////////

// VR3Adapter - Constructor --------------------------------------------------

inline
VR3Adapter::VR3Adapter(const VR3Structure& vr3s)
{
    mvr3structure = &vr3s;
}

// VR3Adapter - Public Methods -----------------------------------------------

inline
int VR3Adapter::countSites() const
{
    return mvr3structure->size();
}


inline
const R3::Vector& VR3Adapter::siteCartesianPosition(int idx) const
{
    return mvr3structure->at(idx);
}


inline
bool VR3Adapter::siteAnisotropy(int idx) const
{
    return false;
}


inline
const R3::Matrix& VR3Adapter::siteCartesianUij(int idx) const
{
    static R3::Matrix Uzero;
    Uzero = 0.0;
    return Uzero;
}


inline
BaseBondGenerator* VR3Adapter::createBondGenerator() const
{
    BaseBondGenerator* bnds = new VR3BondGenerator(this);
    return bnds;
}


// VR3BondGenerator - Constructor --------------------------------------------

inline
VR3BondGenerator::VR3BondGenerator(const VR3Adapter* adpt) :
    BaseBondGenerator(adpt)
{
    mvr3structure = adpt->mvr3structure;
}

// VR3BondGenerator - Public Methods -----------------------------------------

inline
const R3::Vector& VR3BondGenerator::r0() const
{
    return mvr3structure->at(msite_anchor);
}


inline
const R3::Vector& VR3BondGenerator::r1() const
{
    return mvr3structure->at(msite_current);
}

// Inline Routines -----------------------------------------------------------

inline
StructureAdapter* createPQAdapter(const VR3Structure& vr3stru)
{
    StructureAdapter* adapter = new VR3Adapter(vr3stru);
    return adapter;
}


}   // namespace srreal
}   // namespace diffpy

#endif  // VR3STRUCTURE_HPP_INCLUDED
