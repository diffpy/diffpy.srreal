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
* class TestScatteringFactorTable -- unit tests for implementations
*     of the ScatteringFactorTable class
*
* $Id$
*
*****************************************************************************/

#include <stdexcept>
#include <memory>
#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

#include <diffpy/srreal/ScatteringFactorTable.hpp>
#include <diffpy/PythonInterface.hpp>
#include "globals.hpp"

using namespace std;
using namespace boost;
using namespace diffpy::srreal;


class TestScatteringFactorTable : public CppUnit::TestFixture
{

    CPPUNIT_TEST_SUITE(TestScatteringFactorTable);
    CPPUNIT_TEST(test_factory);
    CPPUNIT_TEST(test_setCustom);
    CPPUNIT_TEST(test_periodictableXray);
    CPPUNIT_TEST(test_periodictableNeutron);
    CPPUNIT_TEST_SUITE_END();

private:

    static const double mtol = 1.0e-4;
    auto_ptr<ScatteringFactorTable> sftb;

public:

    void test_factory()
    {
        typedef auto_ptr<ScatteringFactorTable> APSFT;
        CPPUNIT_ASSERT_THROW(createScatteringFactorTable("invalid"),
                invalid_argument);
        APSFT sfx0(createScatteringFactorTable("SFTperiodictableXray"));
        APSFT sfx1(createScatteringFactorTable("X"));
        CPPUNIT_ASSERT(sfx0.get());
        CPPUNIT_ASSERT_EQUAL(sfx0->type(), sfx1->type());
        APSFT sfn0(createScatteringFactorTable("SFTperiodictableNeutron"));
        APSFT sfn1(createScatteringFactorTable("N"));
        CPPUNIT_ASSERT(sfn0.get());
        CPPUNIT_ASSERT_EQUAL(sfn0->type(), sfn1->type());
    }


    void test_setCustom()
    {
        sftb.reset(createScatteringFactorTable("X"));
        CPPUNIT_ASSERT_EQUAL(6.0, sftb->lookup("C"));
        sftb->setCustom("C", 6.3);
        CPPUNIT_ASSERT_THROW(sftb->lookup("Ccustom"), invalid_argument);
        sftb->setCustom("Ccustom", 6.5);
        CPPUNIT_ASSERT_EQUAL(6.5, sftb->lookup("Ccustom"));
        sftb->resetCustom("C");
        CPPUNIT_ASSERT_EQUAL(6.5, sftb->lookup("Ccustom"));
        CPPUNIT_ASSERT_EQUAL(6.0, sftb->lookup("C"));
        sftb->resetAll();
        CPPUNIT_ASSERT_THROW(sftb->lookup("Ccustom"), invalid_argument);
        CPPUNIT_ASSERT_EQUAL(6.0, sftb->lookup("C"));
    }


    void test_periodictableXray()
    {
        sftb.reset(createScatteringFactorTable("X"));
        CPPUNIT_ASSERT_EQUAL(1.0, sftb->lookup("H"));
        CPPUNIT_ASSERT_EQUAL(8.0, sftb->lookup("O"));
        CPPUNIT_ASSERT_EQUAL(74.0, sftb->lookup("W"));
        CPPUNIT_ASSERT_EQUAL(88.0, sftb->lookup("Ra"));
    }


    void test_periodictableNeutron()
    {
        sftb.reset(createScatteringFactorTable("N"));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.63, sftb->lookup("Na"), mtol);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-3.37, sftb->lookup("Ti"), mtol);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(5.805, sftb->lookup("O"), mtol);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(6.6484, sftb->lookup("C"), mtol);
    }

};

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION(TestScatteringFactorTable);

// End of file
