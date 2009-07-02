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
#include <cxxtest/TestSuite.h>

#include <diffpy/srreal/ScatteringFactorTable.hpp>
#include <diffpy/PythonInterface.hpp>

using namespace std;
using namespace diffpy::srreal;


class TestScatteringFactorTable : public CxxTest::TestSuite
{

private:

    static const double mtol = 1.0e-4;
    auto_ptr<ScatteringFactorTable> sftb;

public:

    void test_factory()
    {
        typedef auto_ptr<ScatteringFactorTable> APSFT;
        TS_ASSERT_THROWS(createScatteringFactorTable("invalid"),
                invalid_argument);
        APSFT sfx0(createScatteringFactorTable("SFTperiodictableXray"));
        APSFT sfx1(createScatteringFactorTable("X"));
        TS_ASSERT(sfx0.get());
        TS_ASSERT_EQUALS(sfx0->type(), sfx1->type());
        APSFT sfn0(createScatteringFactorTable("SFTperiodictableNeutron"));
        APSFT sfn1(createScatteringFactorTable("N"));
        TS_ASSERT(sfn0.get());
        TS_ASSERT_EQUALS(sfn0->type(), sfn1->type());
    }


    void test_setCustom()
    {
        sftb.reset(createScatteringFactorTable("X"));
        TS_ASSERT_EQUALS(6.0, sftb->lookup("C"));
        sftb->setCustom("C", 6.3);
        TS_ASSERT_THROWS(sftb->lookup("Ccustom"), invalid_argument);
        sftb->setCustom("Ccustom", 6.5);
        TS_ASSERT_EQUALS(6.5, sftb->lookup("Ccustom"));
        sftb->resetCustom("C");
        TS_ASSERT_EQUALS(6.5, sftb->lookup("Ccustom"));
        TS_ASSERT_EQUALS(6.0, sftb->lookup("C"));
        sftb->resetAll();
        TS_ASSERT_THROWS(sftb->lookup("Ccustom"), invalid_argument);
        TS_ASSERT_EQUALS(6.0, sftb->lookup("C"));
    }


    void test_periodictableXray()
    {
        sftb.reset(createScatteringFactorTable("X"));
        TS_ASSERT_EQUALS(1.0, sftb->lookup("H"));
        TS_ASSERT_EQUALS(8.0, sftb->lookup("O"));
        TS_ASSERT_EQUALS(74.0, sftb->lookup("W"));
        TS_ASSERT_EQUALS(88.0, sftb->lookup("Ra"));
    }


    void test_periodictableNeutron()
    {
        sftb.reset(createScatteringFactorTable("N"));
        TS_ASSERT_DELTA(3.63, sftb->lookup("Na"), mtol);
        TS_ASSERT_DELTA(-3.37, sftb->lookup("Ti"), mtol);
        TS_ASSERT_DELTA(5.805, sftb->lookup("O"), mtol);
        TS_ASSERT_DELTA(6.6484, sftb->lookup("C"), mtol);
    }

};

// End of file
