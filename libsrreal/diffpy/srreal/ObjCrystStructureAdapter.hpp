/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* class ObjCrystStructureAdapter -- adapter to the Crystal class from
* ObjCryst++.
*     
* class ObjCrystBondGenerator -- bond generator
*
*
* $Id:$
*
*****************************************************************************/

#ifndef OBJCRYSTSTRUCTUREADAPTER_HPP_INCLUDED
#define OBJCRYSTSTRUCTUREADAPTER_HPP_INCLUDED

#include <memory>
#include <vector>

#include <diffpy/srreal/R3linalg.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>
#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/srreal/Lattice.hpp>

#include <ObjCryst/Crystal.h>
#include <ObjCryst/ScatteringPower.h>

namespace diffpy {
namespace srreal {

class PointsInSphere;
class PDFCalculator;

class ObjCrystStructureAdapter : public StructureAdapter
{
    friend class ObjCrystBondGenerator;
    public:

        // constructors
        ObjCrystStructureAdapter(const ObjCryst::Crystal*);

        // methods - overloaded
        virtual BaseBondGenerator* createBondGenerator() const;
        virtual int countSites() const;
        virtual double numberDensity() const;
        virtual const R3::Vector& siteCartesianPosition(int idx) const;
        virtual double siteOccupancy(int idx) const;
        virtual bool siteAnisotropy(int idx) const;
        virtual const R3::Matrix& siteCartesianUij(int idx) const;
        virtual const std::string& siteAtomType(int idx) const;

        const Lattice& getLattice() const;

    private:

        /* ShiftedSC
        *
        * Class for holding a shifted scattering component in an
        * ObjCryst::Crystal.  It holds the cartesian position of a scatterer
        * within the conventional unit cell, along with a pointer to the
        * unshifted ScatteringComponent.
        * 
        */
        class ShiftedSC {

            public:
            ShiftedSC(const ObjCryst::ScatteringComponent* _sc,
                const double& x, const double& y, const double& z);

            ShiftedSC(const ShiftedSC& _ssc);
            ShiftedSC();

            /* Data members */

            // Pointer to a ScatteringComponent
            const ObjCryst::ScatteringComponent* sc;

            // Orthonormal coordinates
            R3::Vector xyz;
            // Orthonormal displacement parameters
            R3::Matrix uij;

            /* Operators */

            bool operator<(const ShiftedSC& rhs) const;

            // Compares equality.
            bool operator==(const ShiftedSC& rhs) const;

        };

        // methods - own
        void getUnitCell();

        // data
        const ObjCryst::Crystal* pcryst;
        std::vector<ShiftedSC> vssc;
        Lattice mlattice;


};


class ObjCrystBondGenerator : public BaseBondGenerator
{
    public:

        // constructors
        ObjCrystBondGenerator(const ObjCrystStructureAdapter*);

        // methods
        // loop control
        virtual void rewind();

        // configuration
        virtual void setRmin(double);
        virtual void setRmax(double);

        // data access
        virtual const R3::Vector& r1() const;
        virtual double msd0() const;
        virtual double msd1() const;

    protected:

        // methods
        virtual bool iterateSymmetry();
        virtual void rewindSymmetry();

    private:

        // data
        const ObjCrystStructureAdapter* pstructure;
        std::auto_ptr<PointsInSphere> msphere;

        // methods
        double msdSiteDir(int siteidx, const R3::Vector& s) const;

};

inline
StructureAdapter* 
createPQAdapter(const ObjCryst::Crystal* cryst)
{
    StructureAdapter* adapter = new ObjCrystStructureAdapter(cryst);
    return adapter;
}

inline
StructureAdapter* 
createPQAdapter(const ObjCryst::Crystal& cryst)
{
    StructureAdapter* adapter = new ObjCrystStructureAdapter(&cryst);
    return adapter;
}




}   // namespace srreal
}   // namespace diffpy

#endif  // OBJCRYSTSTRUCTUREADAPTER_HPP_INCLUDED
