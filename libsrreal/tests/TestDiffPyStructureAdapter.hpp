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
* class TestDiffPyStructureAdapter -- unit tests for an adapter
*     to Structure class from diffpy.Structure
*
* $Id$
*
*****************************************************************************/

#include <cxxtest/TestSuite.h>
#include <boost/filesystem/path.hpp>
#include <boost/python.hpp>

#include <diffpy/srreal/DiffPyStructureAdapter.hpp>
#include <diffpy/PythonInterface.hpp>
#include "globals.hpp"

using namespace std;
using namespace boost;
using namespace diffpy::srreal;

// Local Helpers -------------------------------------------------------------

namespace {

python::object newDiffPyStructure()
{
    python::object mod = python::import("diffpy.Structure");
    python::object stru = mod.attr("Structure")();
    return stru;
}


python::object loadTestStructure(const string& tailname)
{
    using boost::filesystem::path;
    path fp = path(testdata_dir()) /= tailname;
    python::object stru = newDiffPyStructure();
    stru.attr("read")(fp.string());
    return stru;
}


int countBonds(BaseBondGenerator& bnds)
{
    int rv = 0;
    for (bnds.rewind(); !bnds.finished(); bnds.next())  ++rv;
    return rv;
}

}   // namespace

//////////////////////////////////////////////////////////////////////////////
// class TestDiffPyStructureAdapter
//////////////////////////////////////////////////////////////////////////////

class TestDiffPyStructureAdapter : public CxxTest::TestSuite
{
private:

    auto_ptr<DiffPyStructureAdapter> m_ni;
    auto_ptr<DiffPyStructureAdapter> m_kbise;
    auto_ptr<DiffPyStructureAdapter> m_catio3;
    auto_ptr<DiffPyStructureAdapter> m_pswt;

public:

    void setUp()
    {
        diffpy::initializePython();
        if (!m_ni.get())
        {
            python::object stru;
            stru = loadTestStructure("Ni.cif");
            m_ni.reset(new DiffPyStructureAdapter(stru));
        }
        if (!m_kbise.get())
        {
            python::object stru;
            stru = loadTestStructure("alpha_K2Bi8Se13.cif");
            m_kbise.reset(new DiffPyStructureAdapter(stru));
        }
        if (!m_catio3.get())
        {
            python::object stru;
            stru = loadTestStructure("icsd_62149.cif");
            m_catio3.reset(new DiffPyStructureAdapter(stru));
        }
        if (!m_pswt.get())
        {
            python::object stru;
            stru = loadTestStructure("PbScW25TiO3.stru");
            m_pswt.reset(new DiffPyStructureAdapter(stru));
        }
    }


    void test_countSites()
    {
        TS_ASSERT_EQUALS(4, m_ni->countSites());
        TS_ASSERT_EQUALS(23, m_kbise->countSites());
        TS_ASSERT_EQUALS(20, m_catio3->countSites());
        TS_ASSERT_EQUALS(56, m_pswt->countSites());
    }


    void test_totalOccupancy()
    {
        TS_ASSERT_EQUALS(4.0, m_ni->totalOccupancy());
        TS_ASSERT_EQUALS(40.0, m_pswt->totalOccupancy());
    }


    void test_numberDensity()
    {
        const double eps = 1.0e-7;
        TS_ASSERT_DELTA(0.0914114, m_ni->numberDensity(), eps);
        TS_ASSERT_DELTA(0.0760332, m_pswt->numberDensity(), eps);
        TS_ASSERT_DELTA(0.0335565, m_kbise->numberDensity(), eps);
    }


    void test_siteCartesianPosition()
    {
        const double eps = 1.0e-5;
        R3::Vector rCa = m_catio3->siteCartesianPosition(0);
        TS_ASSERT_DELTA(2.72617, rCa[0], eps);
        TS_ASSERT_DELTA(2.91718, rCa[1], eps);
        TS_ASSERT_DELTA(1.91003, rCa[2], eps);
    }


    void test_siteAnisotropy()
    {
        for (int i = 0; i < m_ni->countSites(); ++i)
        {
            TS_ASSERT_EQUALS(false, m_ni->siteAnisotropy(i));
        }
        for (int i = 0; i < m_catio3->countSites(); ++i)
        {
            TS_ASSERT_EQUALS(true, m_catio3->siteAnisotropy(i));
        }
    }


    void test_siteCartesianUij()
    {
        // nickel should have all Uij equal zero.
        const double* puij = m_ni->siteCartesianUij(0).data();
        for (int i = 0; i < 9; ++i)
        {
            TS_ASSERT_EQUALS(0.0, puij[i]);
        }
        // check CaTiO3 values
        const R3::Matrix& UTi = m_catio3->siteCartesianUij(7);
        const double eps = 1e-4;
        TS_ASSERT_DELTA(0.0052, UTi(0,0), eps);
        TS_ASSERT_DELTA(0.0049, UTi(1,1), eps);
        TS_ASSERT_DELTA(0.0049, UTi(2,2), eps);
        TS_ASSERT_DELTA(0.00016, UTi(0,1), eps);
        TS_ASSERT_DELTA(0.00001, UTi(0,2), eps);
        TS_ASSERT_DELTA(0.00021, UTi(1,2), eps);
    }


    void test_siteAtomType()
    {
        TS_ASSERT_EQUALS(string("Ni"), m_ni->siteAtomType(0));
        TS_ASSERT_EQUALS(string("Ni"), m_ni->siteAtomType(3));
        TS_ASSERT_EQUALS(string("K1+"), m_kbise->siteAtomType(0));
        TS_ASSERT_EQUALS(string("Bi3+"), m_kbise->siteAtomType(2));
        TS_ASSERT_EQUALS(string("Se"), m_kbise->siteAtomType(10));
        TS_ASSERT_EQUALS(string("Se"), m_kbise->siteAtomType(22));
    }


    void test_getLattice()
    {
        const Lattice& L = m_kbise->getLattice();
        TS_ASSERT_EQUALS(13.768, L.a());
        TS_ASSERT_EQUALS(12.096, L.b());
        TS_ASSERT_EQUALS(4.1656, L.c());
        TS_ASSERT_EQUALS(89.98, L.alpha());
        TS_ASSERT_EQUALS(98.64, L.beta());
        TS_ASSERT_EQUALS(87.96, L.gamma() );
    }

};  // class TestDiffPyStructureAdapter

//////////////////////////////////////////////////////////////////////////////
// class TestDiffPyStructureBondGenerator
//////////////////////////////////////////////////////////////////////////////

class TestDiffPyStructureBondGenerator : public CxxTest::TestSuite
{
private:

    auto_ptr<DiffPyStructureAdapter> m_ni;
    auto_ptr<BaseBondGenerator> m_nibnds;

public:

    void setUp()
    {
        diffpy::initializePython();
        if (!m_ni.get())
        {
            python::object stru;
            stru = loadTestStructure("Ni.cif");
            m_ni.reset(new DiffPyStructureAdapter(stru));
        }
        m_nibnds.reset(m_ni->createBondGenerator());
    }


    void test_bondCountNickel()
    {
        m_nibnds->selectAnchorSite(0);
        m_nibnds->setRmin(0);
        m_nibnds->setRmax(1.0);
        TS_ASSERT_EQUALS(0, countBonds(*m_nibnds));
        m_nibnds->selectAnchorSite(3);
        TS_ASSERT_EQUALS(0, countBonds(*m_nibnds));
        m_nibnds->setRmin(-10);
        TS_ASSERT_EQUALS(0, countBonds(*m_nibnds));
        // there are 12 nearest neighbors at 2.49
        m_nibnds->setRmax(3);
        TS_ASSERT_EQUALS(12, countBonds(*m_nibnds));
        m_nibnds->selectAnchorSite(0);
        TS_ASSERT_EQUALS(12, countBonds(*m_nibnds));
        // there are no self neighbors below the cell length of 3.52
        m_nibnds->selectAnchorSite(0);
        m_nibnds->selectSiteRange(0, 1);
        TS_ASSERT_EQUALS(0, countBonds(*m_nibnds));
        // and any other unit cell atom would give 4 neighbors
        m_nibnds->selectAnchorSite(0);
        m_nibnds->selectSiteRange(3, 4);
        TS_ASSERT_EQUALS(4, countBonds(*m_nibnds));
        // there are no bonds between 2.6 and 3.4
        m_nibnds->setRmin(2.6);
        m_nibnds->setRmax(3.4);
        m_nibnds->selectSiteRange(0, 4);
        TS_ASSERT_EQUALS(0, countBonds(*m_nibnds));
        // there are 6 second nearest neighbors at 3.52
        m_nibnds->setRmax(3.6);
        TS_ASSERT_EQUALS(6, countBonds(*m_nibnds));
        // which sums to 18 neigbhors within radius 3.6
        m_nibnds->setRmin(0);
        TS_ASSERT_EQUALS(18, countBonds(*m_nibnds));
    }

};  // class TestDiffPyStructureBondGenerator

// End of file
