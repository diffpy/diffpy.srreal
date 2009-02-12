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
* class DiffPyStructureAdapter -- adapter to the Structure class from the
*     Python diffpy.Structure package.
* class DiffPyStructureBondGenerator -- bond generator
*     
*
* $Id$
*
*****************************************************************************/

#ifndef VR3STRUCTURE_HPP_INCLUDED
#define VR3STRUCTURE_HPP_INCLUDED

#include <memory>
#include <vector>
#include <boost/python.hpp>

#include "R3linalg.hpp"
#include "StructureAdapter.hpp"
#include "BaseBondGenerator.hpp"

namespace diffpy {
namespace srreal {

class Lattice;
class PointsInSphere;

class DiffPyStructureAdapter : public StructureAdapter
{
    friend class DiffPyStructureBondGenerator;
    public:

        // constructors
        DiffPyStructureAdapter(const boost::python::object&);

        // methods
        virtual int countSites() const;
        virtual BaseBondGenerator* createBondGenerator() const;

    protected:

        // methods
        const Lattice& getLattice() const;
        const R3::Vector& siteCartesianPosition(int idx) const;
        bool siteAnisotropy(int idx) const;
        const R3::Matrix& siteCartesianUij(int idx) const;
        const std::string& siteAtomType(int idx) const;
        void fetchPythonData();

    private:

        // data
        const boost::python::object* mdpstructure;
        // copied properties
        std::auto_ptr<Lattice> mlattice;
        std::vector<R3::Vector> mcartesian_positions;
        std::vector<bool> manisotropies;
        std::vector<R3::Matrix> mcartesian_uijs;
        std::vector<std::string> matomtypes;

};


class DiffPyStructureBondGenerator : public BaseBondGenerator
{
    public:

        // constructors
        DiffPyStructureBondGenerator(const DiffPyStructureAdapter*);

        // methods
        virtual const R3::Vector& r0() const;
        virtual const R3::Vector& r1() const;
        virtual double msd0() const;
        virtual double msd1() const;

    private:
        // data
        const DiffPyStructureAdapter* mdpstructure;
        std::auto_ptr<PointsInSphere> msphere;
        R3::Vector mr0ucv;
        R3::Vector mr1ucv;
};

double meanSquareDisplacement(const R3::Matrix& Uijcartn, const R3::Vector& s,
        bool anisotropy=false);


//////////////////////////////////////////////////////////////////////////////
// Definitions
//////////////////////////////////////////////////////////////////////////////

// Inline Routines -----------------------------------------------------------

inline
StructureAdapter* createPQAdapter(const boost::python::object& dpstru)
{
    StructureAdapter* adapter = new DiffPyStructureAdapter(dpstru);
    return adapter;
}


}   // namespace srreal
}   // namespace diffpy

#endif  // VR3STRUCTURE_HPP_INCLUDED
