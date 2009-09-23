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
#include <set>
#include <vector>

#include <diffpy/srreal/R3linalg.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>
#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/srreal/Lattice.hpp>

#include <ObjCryst/Crystal.h>
#include <ObjCryst/ScatteringPower.h>

struct _cmpVector;

namespace diffpy {
namespace srreal {


class PointsInSphere;
class PDFCalculator;

class ObjCrystStructureAdapter : public StructureAdapter
{
    friend class ObjCrystBondGenerator;
    public:

        // constructors
        ObjCrystStructureAdapter(const ObjCryst::Crystal&);

        // methods - overloaded
        virtual BaseBondGenerator* createBondGenerator() const;
        virtual int countSites() const;
        virtual double numberDensity() const;
        virtual const R3::Vector& siteCartesianPosition(int idx) const;
        virtual double siteOccupancy(int idx) const;
        virtual bool siteAnisotropy(int idx) const;
        virtual double siteMultiplicity(int idx) const;
        virtual const R3::Matrix& siteCartesianUij(int idx) const;
        virtual const std::string& siteAtomType(int idx) const;

        const Lattice& getLattice() const;


    private:

        // methods - own
        void getUnitCell();


        // For comparing vectors - this is needed by the R3::Vector set that holds
        // symmetry operations.
        struct cmp
        {
            bool operator()(const R3::Vector& v1, const R3::Vector& v2) const
            {
                for(size_t l = 0; l < 3; ++l)
                {
                    if( fabs(v1[l] - v2[l]) > toler )
                    {
                        return v1[l] < v2[l];
                    }
                }
                return false;
            }
        };

        // Tolerance on distance measurments
        static const double toler;
        // The ObjCryst::Crystal
        const ObjCryst::Crystal* pcryst;
        // The asymmetric unit cell of ScatteringComponent instances
        std::vector< ObjCryst::ScatteringComponent > vsc;
        // The symmetry-related operations on the asymmetric unit cell
        std::vector< std::set< R3::Vector, cmp > > vsym;
        // The Uij for the scatterers
        std::vector< R3::Matrix > vuij;
        // The Lattice instance needed by the ObjCrystBondGenerator
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
        std::set< R3::Vector >::const_iterator symiter;
        std::auto_ptr<PointsInSphere> msphere;

        double msdSiteDir(int siteidx, const R3::Vector& s) const;

};


inline
StructureAdapter* 
createPQAdapter(const ObjCryst::Crystal& cryst)
{
    StructureAdapter* adapter = new ObjCrystStructureAdapter(cryst);
    return adapter;
}

}   // namespace srreal
}   // namespace diffpy

#endif  // OBJCRYSTSTRUCTUREADAPTER_HPP_INCLUDED
