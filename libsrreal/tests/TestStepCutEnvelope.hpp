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
* class TestStepCutEnvelope -- unit tests for StepCutEnvelope class
*
* $Id$
*
*****************************************************************************/

#include <memory>
#include <cxxtest/TestSuite.h>

#include <diffpy/srreal/StepCutEnvelope.hpp>

using namespace std;
using namespace diffpy::srreal;

class TestStepCutEnvelope : public CxxTest::TestSuite
{
private:

    auto_ptr<PDFEnvelope> menvelope;

public:

    void setUp()
    {
        menvelope.reset(createPDFEnvelope("stepcut"));
    }


    void test_create()
    {
        TS_ASSERT_EQUALS(0.0, menvelope->getDoubleAttr("stepcut"));
        menvelope->setDoubleAttr("stepcut", 13.0);
        TS_ASSERT_EQUALS(13.0, menvelope->getDoubleAttr("stepcut"));
        auto_ptr<PDFEnvelope> e1(menvelope->create());
        TS_ASSERT_EQUALS(0.0, e1->getDoubleAttr("stepcut"));
    }


    void test_copy()
    {
        menvelope->setDoubleAttr("stepcut", 13.0);
        TS_ASSERT_EQUALS(13.0, menvelope->getDoubleAttr("stepcut"));
        auto_ptr<PDFEnvelope> e1(menvelope->copy());
        TS_ASSERT_EQUALS(13.0, e1->getDoubleAttr("stepcut"));
    }


    void test_type()
    {
        TS_ASSERT_EQUALS("stepcut", menvelope->type());
    }


    void test_parentheses_operator()
    {
        const PDFEnvelope& fne = *menvelope;
        TS_ASSERT_EQUALS(1.0, fne(-1.0));
        TS_ASSERT_EQUALS(1.0, fne(0.0));
        TS_ASSERT_EQUALS(1.0, fne(+1.0));
        menvelope->setDoubleAttr("stepcut", 1.0);
        TS_ASSERT_EQUALS(1.0, fne(-1.0));
        TS_ASSERT_EQUALS(1.0, fne(0.0));
        TS_ASSERT_EQUALS(1.0, fne(+1.0));
        TS_ASSERT_EQUALS(0.0, fne(+1.0001));
        TS_ASSERT_EQUALS(0.0, fne(+2.0));
    }

};  // class TestDiffPyStructureBondGenerator

// End of file
