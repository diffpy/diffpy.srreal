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

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include <boost/filesystem/path.hpp>
#include <boost/python.hpp>

#include <diffpy/srreal/DiffPyStructureAdapter.hpp>
#include <diffpy/PythonInterface.hpp>
#include "globals.hpp"

using namespace std;
using namespace boost;
using namespace diffpy::srreal;


class TestDiffPyStructureAdapter : public CppUnit::TestFixture
{

    CPPUNIT_TEST_SUITE(TestDiffPyStructureAdapter);
    CPPUNIT_TEST(test_countSites);
    CPPUNIT_TEST(test_totalOccupancy);
    CPPUNIT_TEST(test_numberDensity);
    CPPUNIT_TEST(test_siteCartesianPosition);
    CPPUNIT_TEST(test_siteAnisotropy);
    CPPUNIT_TEST(test_siteCartesianUij);
    CPPUNIT_TEST(test_siteAtomType);
    CPPUNIT_TEST(test_getLattice);
    CPPUNIT_TEST_SUITE_END();

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
            stru = this->loadTestStructure("Ni.cif");
            m_ni.reset(new DiffPyStructureAdapter(stru));
        }
        if (!m_kbise.get())
        {
            python::object stru;
            stru = this->loadTestStructure("alpha_K2Bi8Se13.cif");
            m_kbise.reset(new DiffPyStructureAdapter(stru));
        }
        if (!m_catio3.get())
        {
            python::object stru;
            stru = this->loadTestStructure("icsd_62149.cif");
            m_catio3.reset(new DiffPyStructureAdapter(stru));
        }
        if (!m_pswt.get())
        {
            python::object stru;
            stru = this->loadTestStructure("PbScW25TiO3.stru");
            m_pswt.reset(new DiffPyStructureAdapter(stru));
        }
    }


    void test_countSites()
    {
        CPPUNIT_ASSERT_EQUAL(4, m_ni->countSites());
        CPPUNIT_ASSERT_EQUAL(23, m_kbise->countSites());
        CPPUNIT_ASSERT_EQUAL(20, m_catio3->countSites());
        CPPUNIT_ASSERT_EQUAL(56, m_pswt->countSites());
    }


    void test_totalOccupancy()
    {
        CPPUNIT_ASSERT_EQUAL(4.0, m_ni->totalOccupancy());
        CPPUNIT_ASSERT_EQUAL(40.0, m_pswt->totalOccupancy());
    }


    void test_numberDensity()
    {
        const double eps = 1.0e-7;
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0914114, m_ni->numberDensity(), eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0760332, m_pswt->numberDensity(), eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0335565, m_kbise->numberDensity(), eps);
    }


    void test_siteCartesianPosition()
    {
        const double eps = 1.0e-5;
        R3::Vector rCa = m_catio3->siteCartesianPosition(0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.72617, rCa[0], eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.91718, rCa[1], eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.91003, rCa[2], eps);
    }


    void test_siteAnisotropy()
    {
        for (int i = 0; i < m_ni->countSites(); ++i)
        {
            CPPUNIT_ASSERT_EQUAL(false, m_ni->siteAnisotropy(i));
        }
        for (int i = 0; i < m_catio3->countSites(); ++i)
        {
            CPPUNIT_ASSERT_EQUAL(true, m_catio3->siteAnisotropy(i));
        }
    }


    void test_siteCartesianUij()
    {
        // nickel should have all Uij equal zero.
        const double* puij = m_ni->siteCartesianUij(0).data();
        for (int i = 0; i < 9; ++i)
        {
            CPPUNIT_ASSERT_EQUAL(0.0, puij[i]);
        }
        // check CaTiO3 values
        const R3::Matrix& UTi = m_catio3->siteCartesianUij(7);
        const double eps = 1e-4;
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0052, UTi(0,0), eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0049, UTi(1,1), eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0049, UTi(2,2), eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00016, UTi(0,1), eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00001, UTi(0,2), eps);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.00021, UTi(1,2), eps);
    }


    void test_siteAtomType()
    {
        CPPUNIT_ASSERT_EQUAL(string("Ni"), m_ni->siteAtomType(0));
        CPPUNIT_ASSERT_EQUAL(string("Ni"), m_ni->siteAtomType(3));
        CPPUNIT_ASSERT_EQUAL(string("K1+"), m_kbise->siteAtomType(0));
        CPPUNIT_ASSERT_EQUAL(string("Bi3+"), m_kbise->siteAtomType(2));
        CPPUNIT_ASSERT_EQUAL(string("Se"), m_kbise->siteAtomType(10));
        CPPUNIT_ASSERT_EQUAL(string("Se"), m_kbise->siteAtomType(22));
    }


    void test_getLattice()
    {
        const Lattice& L = m_kbise->getLattice();
        CPPUNIT_ASSERT_EQUAL(13.768, L.a());
        CPPUNIT_ASSERT_EQUAL(12.096, L.b());
        CPPUNIT_ASSERT_EQUAL(4.1656, L.c());
        CPPUNIT_ASSERT_EQUAL(89.98, L.alpha());
        CPPUNIT_ASSERT_EQUAL(98.64, L.beta());
        CPPUNIT_ASSERT_EQUAL(87.96, L.gamma() );
    }


private:

    python::object loadTestStructure(const string& tailname) const
    {
        using boost::filesystem::path;
        path fp = path(testdata_dir()) /= tailname;
        python::object stru = this->newDiffPyStructure();
        stru.attr("read")(fp.string());
        return stru;
    }


    python::object newDiffPyStructure() const
    {
        python::object mod = python::import("diffpy.Structure");
        python::object stru = mod.attr("Structure")();
        return stru;
    }

};

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION(TestDiffPyStructureAdapter);

// End of file
