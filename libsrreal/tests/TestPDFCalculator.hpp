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

#include <cxxtest/TestSuite.h>

#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/JeongPeakWidth.hpp>
#include <diffpy/srreal/ConstantPeakWidth.hpp>

using namespace std;
using namespace diffpy::srreal;

class TestPDFCalculator : public CxxTest::TestSuite
{
private:

    auto_ptr<PDFCalculator> mpdfc;

public:

    void setUp()
    {
        mpdfc.reset(new PDFCalculator);
    }


    void testAddition( void )
    {
        TS_ASSERT( 1 + 1 > 1 );
        TS_ASSERT_EQUALS( 1 + 1, 2 );
    }

    void testSetPeakWidthModel()
    {
        const JeongPeakWidth& jpw0 =
            dynamic_cast<const JeongPeakWidth&>(mpdfc->getPeakWidthModel());
        TS_ASSERT_EQUALS(0.0, jpw0.getDelta1());
        TS_ASSERT_EQUALS(0.0, jpw0.getDelta2());
        TS_ASSERT_EQUALS(0.0, jpw0.getQbroad());
        JeongPeakWidth jpw;
        jpw.setDelta1(1.0);
        jpw.setDelta2(2.0);
        jpw.setQbroad(3.0);
        mpdfc->setPeakWidthModel(jpw);
        const JeongPeakWidth& jpw1 =
            dynamic_cast<const JeongPeakWidth&>(mpdfc->getPeakWidthModel());
        TS_ASSERT_EQUALS(1.0, jpw1.getDelta1());
        TS_ASSERT_EQUALS(2.0, jpw1.getDelta2());
        TS_ASSERT_EQUALS(3.0, jpw1.getQbroad());
    }


    void testGetPeakWidthModel()
    {
        string tp = "jeong";
        TS_ASSERT_EQUALS(tp, mpdfc->getPeakWidthModel().type());
        auto_ptr<PeakWidthModel> pwm(createPeakWidthModel("debye-waller"));
        mpdfc->setPeakWidthModel(*pwm);
        tp = "debye-waller";
        TS_ASSERT_EQUALS(tp, mpdfc->getPeakWidthModel().type());
        mpdfc->setPeakWidthModel(ConstantPeakWidth());
        tp = "constant";
        TS_ASSERT_EQUALS(tp, mpdfc->getPeakWidthModel().type());
    }

};

// End of file
