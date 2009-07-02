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
* class TestPairCounter -- unit tests for PairCounter class
*
* $Id$
*
*****************************************************************************/

#include <cxxtest/TestSuite.h>

#include <diffpy/srreal/PairCounter.hpp>
#include <diffpy/srreal/VR3Structure.hpp>

using namespace std;
using namespace diffpy::srreal;

class TestPairCounter : public CxxTest::TestSuite
{

private:

    VR3Structure mstru;
    VR3Structure mline100;

public:

    void setUp()
    {
        mstru.clear();
        if (mline100.empty())
        {
            for (int i = 0; i < 100; ++i)
            {
                R3::Vector P(1.0*i, 0.0, 0.0);
                mline100.push_back(P);
            }
        }
    }


    void test_call()
    {
        PairCounter pcount;
        for (int i = 0; i < 100; ++i)
        {
            int npairs = i * (i - 1) / 2;
            TS_ASSERT_EQUALS(npairs, pcount(mstru));
            R3::Vector P(1.0*i, 0.0, 0.0);
            mstru.push_back(P);
        }
        mstru.clear();
        TS_ASSERT_EQUALS(0, pcount(mstru));
    }


    void test_setRmin()
    {
        PairCounter pcount;
        TS_ASSERT_EQUALS(100*99/2, pcount(mline100));
        pcount.setRmin(100);
        TS_ASSERT_EQUALS(0, pcount(mline100));
        pcount.setRmin(99.0);
        TS_ASSERT_EQUALS(1, pcount(mline100));
        pcount.setRmin(1.1);
        TS_ASSERT_EQUALS(100*99/2 - 99, pcount(mline100));
    }


    void test_setRmax()
    {
        PairCounter pcount;
        pcount.setRmax(0.9);
        TS_ASSERT_EQUALS(0, pcount(mline100));
        pcount.setRmax(1.1);
        TS_ASSERT_EQUALS(99, pcount(mline100));
        pcount.setRmax(98.5);
        TS_ASSERT_EQUALS(100*99/2 - 1, pcount(mline100));
    }

};  // class TestPairCounter

// End of file
