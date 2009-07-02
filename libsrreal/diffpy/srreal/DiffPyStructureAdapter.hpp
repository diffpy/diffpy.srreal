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

#ifndef DIFFPYSTRUCTUREADAPTER_HPP_INCLUDED
#define DIFFPYSTRUCTUREADAPTER_HPP_INCLUDED

#include <memory>
#include <vector>
#include <boost/python.hpp>

#include <diffpy/srreal/R3linalg.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>
#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/srreal/Lattice.hpp>

namespace diffpy {
namespace srreal {

class PointsInSphere;
class PDFCalculator;

class DiffPyStructureAdapter : public StructureAdapter
{
    friend class DiffPyStructureBondGenerator;
    public:

        // constructors
        DiffPyStructureAdapter(const boost::python::object&);

        // methods - overloaded
        virtual BaseBondGenerator* createBondGenerator() const;
        virtual int countSites() const;
        virtual double numberDensity() const;
        virtual const R3::Vector& siteCartesianPosition(int idx) const;
        virtual double siteOccupancy(int idx) const;
        virtual bool siteAnisotropy(int idx) const;
        virtual const R3::Matrix& siteCartesianUij(int idx) const;
        virtual const std::string& siteAtomType(int idx) const;
        virtual void customPQConfig(PairQuantity& pq) const;

        // methods - own
        const Lattice& getLattice() const;

    protected:

        // methods
        void fetchPythonData();
        void configurePDFCalculator(PDFCalculator& pdfc) const;

    private:

        // data
        const boost::python::object* mdpstructure;
        // copied properties
        Lattice mlattice;
        std::vector<R3::Vector> mcartesian_positions;
        std::vector<double> moccupancies;
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
        const DiffPyStructureAdapter* mdpstructure;
        std::auto_ptr<PointsInSphere> msphere;
        R3::Vector mr0ucv;
        R3::Vector mr1ucv;

        // methods
        double msdSiteDir(int siteidx, const R3::Vector& s) const;
};


}   // namespace srreal
}   // namespace diffpy

#endif  // DIFFPYSTRUCTUREADAPTER_HPP_INCLUDED
