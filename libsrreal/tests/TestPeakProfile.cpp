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
* class TestPeakProfile -- unit tests for class GaussPeakProfile
*
* $Id$
*
*****************************************************************************/

#include <cmath>
#include <stdexcept>
#include <memory>
#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

#include <diffpy/srreal/PeakProfile.hpp>

using namespace std;
using namespace diffpy::srreal;


class TestPeakProfile : public CppUnit::TestFixture
{

    CPPUNIT_TEST_SUITE(TestPeakProfile);
    CPPUNIT_TEST(test_factory);
    CPPUNIT_TEST(test_y);
    CPPUNIT_TEST(test_xboundlo);
    CPPUNIT_TEST(test_xboundhi);
    CPPUNIT_TEST(test_setPrecision);
    CPPUNIT_TEST_SUITE_END();

private:

    auto_ptr<PeakProfile> mpkgauss;
    static const int mdigits = 12;

public:

    void setUp()
    {
        mpkgauss.reset(createPeakProfile("gauss"));
    }


    void test_factory()
    {
        CPPUNIT_ASSERT_THROW(createPeakProfile("invalid"),
                invalid_argument);
        CPPUNIT_ASSERT_EQUAL(string("gauss"), mpkgauss->type());
    }


    void test_y()
    {
        double Afwhm1 = 2 * sqrt(M_LN2 / M_PI);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(Afwhm1, mpkgauss->yvalue(0, 1), mdigits);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(Afwhm1 / 2,
                mpkgauss->yvalue(1, 1), mdigits);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(Afwhm1 / 2,
                mpkgauss->yvalue(-1, 1), mdigits);
    }


    void test_xboundlo()
    {
        double epsy = 1e-8;
        mpkgauss->setPrecision(epsy);
        double xblo1 = mpkgauss->xboundlo(1);
        double xblo3 = mpkgauss->xboundlo(3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(epsy, mpkgauss->yvalue(xblo1, 1), mdigits);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(epsy, mpkgauss->yvalue(xblo3, 3), mdigits);
        CPPUNIT_ASSERT(xblo1 < 0);
        CPPUNIT_ASSERT(xblo3 < 0);
        mpkgauss->setPrecision(10);
        CPPUNIT_ASSERT_EQUAL(0.0, mpkgauss->xboundlo(1));
        mpkgauss->setPrecision(0.1);
        CPPUNIT_ASSERT_EQUAL(0.0, mpkgauss->xboundlo(0));
        CPPUNIT_ASSERT_EQUAL(0.0, mpkgauss->xboundlo(-1));
    }


    void test_xboundhi()
    {
        double epsy = 1e-8;
        mpkgauss->setPrecision(epsy);
        double xbhi1 = mpkgauss->xboundhi(1);
        double xbhi3 = mpkgauss->xboundhi(3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(epsy,
                mpkgauss->yvalue(xbhi1, 1), mdigits);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(epsy,
                mpkgauss->yvalue(xbhi3, 3), mdigits);
        CPPUNIT_ASSERT(xbhi1 > 0);
        CPPUNIT_ASSERT(xbhi3 > 0);
    }


    void test_setPrecision()
    {
        double epsy = 1e-7;
        CPPUNIT_ASSERT_EQUAL(0.0, mpkgauss->getPrecision());
        mpkgauss->setPrecision(epsy);
        CPPUNIT_ASSERT_EQUAL(epsy, mpkgauss->getPrecision());
        double xbhi1 = mpkgauss->xboundhi(1);
        mpkgauss->setPrecision(1e-4);
        CPPUNIT_ASSERT(xbhi1 != mpkgauss->xboundhi(1));
    }


};

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION(TestPeakProfile);

// End of file
