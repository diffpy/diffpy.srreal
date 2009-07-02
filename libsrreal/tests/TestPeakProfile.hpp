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
* class TestPeakProfile -- unit tests for various PeakProfile classes
*
* $Id$
*
*****************************************************************************/

#include <cmath>
#include <stdexcept>
#include <memory>

#include <cxxtest/TestSuite.h>

#include <diffpy/srreal/PeakProfile.hpp>

using namespace std;
using namespace diffpy::srreal;


class TestPeakProfile : public CxxTest::TestSuite
{

private:

    auto_ptr<PeakProfile> mpkgauss;
    static const int mdigits = 12;

public:

    void setUp()
    {
        mpkgauss.reset(createPeakProfile("gaussian"));
    }


    void test_factory()
    {
        TS_ASSERT_THROWS(createPeakProfile("invalid"),
                invalid_argument);
        TS_ASSERT_EQUALS(string("gaussian"), mpkgauss->type());
    }


    void test_y()
    {
        double Afwhm1 = 2 * sqrt(M_LN2 / M_PI);
        TS_ASSERT_DELTA(Afwhm1, mpkgauss->yvalue(0, 1), mdigits);
        TS_ASSERT_DELTA(Afwhm1 / 2,
                mpkgauss->yvalue(1, 1), mdigits);
        TS_ASSERT_DELTA(Afwhm1 / 2,
                mpkgauss->yvalue(-1, 1), mdigits);
    }


    void test_xboundlo()
    {
        double epsy = 1e-8;
        mpkgauss->setPrecision(epsy);
        double xblo1 = mpkgauss->xboundlo(1);
        double xblo3 = mpkgauss->xboundlo(3);
        TS_ASSERT_DELTA(epsy, mpkgauss->yvalue(xblo1, 1), mdigits);
        TS_ASSERT_DELTA(epsy, mpkgauss->yvalue(xblo3, 3), mdigits);
        TS_ASSERT(xblo1 < 0);
        TS_ASSERT(xblo3 < 0);
        mpkgauss->setPrecision(10);
        TS_ASSERT_EQUALS(0.0, mpkgauss->xboundlo(1));
        mpkgauss->setPrecision(0.1);
        TS_ASSERT_EQUALS(0.0, mpkgauss->xboundlo(0));
        TS_ASSERT_EQUALS(0.0, mpkgauss->xboundlo(-1));
    }


    void test_xboundhi()
    {
        double epsy = 1e-8;
        mpkgauss->setPrecision(epsy);
        double xbhi1 = mpkgauss->xboundhi(1);
        double xbhi3 = mpkgauss->xboundhi(3);
        TS_ASSERT_DELTA(epsy,
                mpkgauss->yvalue(xbhi1, 1), mdigits);
        TS_ASSERT_DELTA(epsy,
                mpkgauss->yvalue(xbhi3, 3), mdigits);
        TS_ASSERT(xbhi1 > 0);
        TS_ASSERT(xbhi3 > 0);
    }


    void test_setPrecision()
    {
        double epsy = 1e-7;
        TS_ASSERT_EQUALS(0.0, mpkgauss->getPrecision());
        mpkgauss->setPrecision(epsy);
        TS_ASSERT_EQUALS(epsy, mpkgauss->getPrecision());
        double xbhi1 = mpkgauss->xboundhi(1);
        mpkgauss->setPrecision(1e-4);
        TS_ASSERT(xbhi1 != mpkgauss->xboundhi(1));
    }

};  // class TestPeakProfile

// End of file
