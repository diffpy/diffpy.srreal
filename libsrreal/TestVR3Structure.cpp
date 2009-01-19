/***********************************************************************
* Short Title: unit tests for R3linalg
*
* Comments:
*
* $Id$
*
* <license text>
***********************************************************************/

#include <stdexcept>
#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

#include "VR3Structure.hpp"
#include "PairCounter.hpp"

using namespace std;
using namespace diffpy;

class TestVR3Structure : public CppUnit::TestFixture
{

    CPPUNIT_TEST_SUITE(TestVR3Structure);
    CPPUNIT_TEST(test_PairCounter);
    CPPUNIT_TEST_SUITE_END();

private:

    VR3Structure mstru;

public:

    void setUp()
    {
        mstru.clear();
    }


    void test_PairCounter()
    {
        PairCounter pcount;
        for (int i = 0; i < 100; ++i)
        {
            int npairs = i * (i - 1) / 2;
            CPPUNIT_ASSERT_EQUAL(npairs, pcount(mstru));
            R3::Vector P(1.0*i, 0.0, 0.0);
            mstru.push_back(P);
        }
        mstru.clear();
        CPPUNIT_ASSERT_EQUAL(0, pcount(mstru));
    }

};

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION(TestVR3Structure);

// End of file
