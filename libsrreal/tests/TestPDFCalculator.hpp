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

#include <memory>
#include <cxxtest/TestSuite.h>

#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/JeongPeakWidth.hpp>
#include <diffpy/srreal/ConstantPeakWidth.hpp>
#include <diffpy/srreal/QResolutionEnvelope.hpp>

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


    void test_setPeakWidthModel()
    {
        const PeakWidthModel& jpw0 = mpdfc->getPeakWidthModel();
        TS_ASSERT_EQUALS(0.0, jpw0.getDoubleAttr("delta1"));
        TS_ASSERT_EQUALS(0.0, jpw0.getDoubleAttr("delta2"));
        TS_ASSERT_EQUALS(0.0, jpw0.getDoubleAttr("qbroad"));
        JeongPeakWidth jpw;
        jpw.setDelta1(1.0);
        jpw.setDelta2(2.0);
        jpw.setQbroad(3.0);
        mpdfc->setPeakWidthModel(jpw);
        const PeakWidthModel& jpw1 = mpdfc->getPeakWidthModel();
        TS_ASSERT_EQUALS(1.0, jpw1.getDoubleAttr("delta1"));
        TS_ASSERT_EQUALS(2.0, jpw1.getDoubleAttr("delta2"));
        TS_ASSERT_EQUALS(3.0, jpw1.getDoubleAttr("qbroad"));
    }


    void test_getPeakWidthModel()
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


    void test_access_PeakProfile()
    {
        auto_ptr<PeakProfile> pkf(mpdfc->getPeakProfile().copy());
        pkf->setPrecision(1.1);
        mpdfc->setPeakProfile(*pkf);
        TS_ASSERT_EQUALS(1.1, mpdfc->getPeakProfile().getPrecision());
        TS_ASSERT_EQUALS(1.1, mpdfc->getDoubleAttr("peakprecision"));
        mpdfc->setDoubleAttr("peakprecision", 0.2);
        TS_ASSERT_EQUALS(0.2, mpdfc->getDoubleAttr("peakprecision"));
        TS_ASSERT_EQUALS(pkf->type(), mpdfc->getPeakProfile().type());
        mpdfc->setPeakProfile("gaussian");
        TS_ASSERT_EQUALS(0.2, mpdfc->getDoubleAttr("peakprecision"));
        TS_ASSERT_THROWS(mpdfc->setPeakProfile("invalid"), logic_error);
    }


    void test_access_Envelopes()
    {
        TS_ASSERT_EQUALS(2u, mpdfc->usedEnvelopeTypes().size());
        TS_ASSERT_EQUALS(1.0, mpdfc->getDoubleAttr("scale"));
        TS_ASSERT_EQUALS(0.0, mpdfc->getDoubleAttr("qdamp"));
        mpdfc->setDoubleAttr("scale", 3.0);
        TS_ASSERT_EQUALS(3.0, mpdfc->getDoubleAttr("scale"));
        mpdfc->addEnvelope("scale");
        TS_ASSERT_EQUALS(1.0, mpdfc->getDoubleAttr("scale"));
        QResolutionEnvelope qdamp4;
        qdamp4.setQdamp(4);
        mpdfc->addEnvelope(qdamp4);
        TS_ASSERT_EQUALS(4.0, mpdfc->getDoubleAttr("qdamp"));
        TS_ASSERT_THROWS(mpdfc->addEnvelope("invalid"), logic_error);
    }

};

// End of file
