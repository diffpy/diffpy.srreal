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
* class TestPDFCalculator -- unit tests for PDFCalculator class
*
* $Id$
*
*****************************************************************************/

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/JeongPeakWidth.hpp>
#include <diffpy/srreal/ConstantPeakWidth.hpp>

using namespace std;
using namespace diffpy::srreal;

class TestPDFCalculator : public CppUnit::TestFixture
{

    CPPUNIT_TEST_SUITE(TestPDFCalculator);
    CPPUNIT_TEST(test_setPeakWidthModel);
    CPPUNIT_TEST(test_getPeakWidthModel);
    CPPUNIT_TEST_SUITE_END();

private:

    auto_ptr<PDFCalculator> mpdfc;

public:

    void setUp()
    {
        mpdfc.reset(new PDFCalculator);
    }


    void test_setPeakWidthModel()
    {
        const JeongPeakWidth& jpw0 =
            dynamic_cast<const JeongPeakWidth&>(mpdfc->getPeakWidthModel());
        CPPUNIT_ASSERT_EQUAL(0.0, jpw0.getDelta1());
        CPPUNIT_ASSERT_EQUAL(0.0, jpw0.getDelta2());
        CPPUNIT_ASSERT_EQUAL(0.0, jpw0.getQbroad());
        JeongPeakWidth jpw;
        jpw.setDelta1(1.0);
        jpw.setDelta2(2.0);
        jpw.setQbroad(3.0);
        mpdfc->setPeakWidthModel(jpw);
        const JeongPeakWidth& jpw1 =
            dynamic_cast<const JeongPeakWidth&>(mpdfc->getPeakWidthModel());
        CPPUNIT_ASSERT_EQUAL(1.0, jpw1.getDelta1());
        CPPUNIT_ASSERT_EQUAL(2.0, jpw1.getDelta2());
        CPPUNIT_ASSERT_EQUAL(3.0, jpw1.getQbroad());
    }


    void test_getPeakWidthModel()
    {
        string tp = "jeong";
        CPPUNIT_ASSERT_EQUAL(tp, mpdfc->getPeakWidthModel().type());
        auto_ptr<PeakWidthModel> pwm(createPeakWidthModel("debye-waller"));
        mpdfc->setPeakWidthModel(*pwm);
        tp = "debye-waller";
        CPPUNIT_ASSERT_EQUAL(tp, mpdfc->getPeakWidthModel().type());
        mpdfc->setPeakWidthModel(ConstantPeakWidth());
        tp = "constant";
        CPPUNIT_ASSERT_EQUAL(tp, mpdfc->getPeakWidthModel().type());
    }


};

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION(TestPDFCalculator);

// End of file
