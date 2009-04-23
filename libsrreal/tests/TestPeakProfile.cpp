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
    CPPUNIT_TEST_SUITE_END();

private:

    const PeakProfile* mpkgauss;
    static const int mdigits = 12;

public:

    void setUp()
    {
        mpkgauss = borrowPeakProfile("gauss");
    }


    void test_factory()
    {
        CPPUNIT_ASSERT_THROW(borrowPeakProfile("invalid"),
                invalid_argument);
        CPPUNIT_ASSERT_EQUAL(mpkgauss, borrowPeakProfile("gauss"));
    }


    void test_y()
    {
        double Afwhm1 = 2 * sqrt(M_LN2 / M_PI);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(Afwhm1, mpkgauss->y(0, 1), mdigits);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(Afwhm1 / 2, mpkgauss->y(1, 1), mdigits);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(Afwhm1 / 2, mpkgauss->y(-1, 1), mdigits);
    }


    void test_xboundlo()
    {
        double epsy = 1e-8;
        double xblo1 = mpkgauss->xboundlo(epsy, 1);
        double xblo3 = mpkgauss->xboundlo(epsy, 3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(epsy, mpkgauss->y(xblo1, 1), mdigits);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(epsy, mpkgauss->y(xblo3, 3), mdigits);
        CPPUNIT_ASSERT(xblo1 < 0);
        CPPUNIT_ASSERT(xblo3 < 0);
        CPPUNIT_ASSERT_EQUAL(0.0, mpkgauss->xboundlo(10, 1));
        CPPUNIT_ASSERT_EQUAL(0.0, mpkgauss->xboundlo(0.1, 0));
        CPPUNIT_ASSERT_EQUAL(0.0, mpkgauss->xboundlo(0.1, -1));
    }


    void test_xboundhi()
    {
        double epsy = 1e-8;
        double xbhi1 = mpkgauss->xboundhi(epsy, 1);
        double xbhi3 = mpkgauss->xboundhi(epsy, 3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(epsy, mpkgauss->y(xbhi1, 1), mdigits);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(epsy, mpkgauss->y(xbhi3, 3), mdigits);
        CPPUNIT_ASSERT(xbhi1 > 0);
        CPPUNIT_ASSERT(xbhi3 > 0);
    }

};

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION(TestPeakProfile);

// End of file
