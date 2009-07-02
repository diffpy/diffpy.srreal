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
* class TestR3linalg -- unit test suite for the R3linalg module
*
* $Id$
*
*****************************************************************************/

#include <cxxtest/TestSuite.h>

#include <diffpy/srreal/R3linalg.hpp>

using namespace std;
using namespace diffpy::srreal;
using diffpy::srreal::R3::MatricesAlmostEqual;
using diffpy::srreal::R3::VectorsAlmostEqual;


class TestR3linalg : public CxxTest::TestSuite
{

private:

    static const double precision = 1.0e-12;

public:

    void test_determinant()
    {
	// default lattice should be cartesian
	R3::Matrix A1, A2;
	A1 = 9, 1, 9,
	     6, 7, 4,
	     0, 2, 9;
	A2 = 9, 1, 9,
	     0, 2, 9,
	     6, 7, 4;
	double detA = 549;
	TS_ASSERT_EQUALS(detA, R3::determinant(A1));
	TS_ASSERT_EQUALS(-detA, R3::determinant(A2));
    }


    void test_inverse()
    {
	R3::Matrix A, invA;
	A =
	    0.5359, -0.5904, 0.8670,
	   -0.0053,  0.7559, 0.2692,
	   -0.8926,  0.9424, 0.9692;
	invA =
	    0.49063197005867, 1.42323870111089, -0.83420736316541,
	   -0.24089965988852, 1.32489393619466, -0.15249839300481,
	    0.68609361943181, 0.02249568627913,  0.41178393851247;
	TS_ASSERT(MatricesAlmostEqual(invA, R3::inverse(A), precision));
    }


    void test_transpose()
    {
	R3::Matrix A, Atrans;
	A =
	    0.5359, -0.5904, 0.8670,
	   -0.0053,  0.7559, 0.2692,
	   -0.8926,  0.9424, 0.9692;
	Atrans =
	    0.5359, -0.0053, -0.8926,
           -0.5904, 0.7559, 0.9424,
            0.8670, 0.2692, 0.9692;
	TS_ASSERT(MatricesAlmostEqual(Atrans, R3::transpose(A), 0.0));
    }


    void test_norm()
    {
        R3::Vector v1, v2;
        v1 = 3.0, 4.0, 0.0;
        v2 = 0.67538115798129, 0.72108424545413, 0.15458914063315;
        TS_ASSERT_DELTA(5.0, R3::norm(v1), precision);
        TS_ASSERT_DELTA(1.0, R3::norm(v2), precision);
    }


    void test_dot()
    {
        R3::Vector v1, v2;
        v1 = -0.97157650177843, 0.43206192654604, 0.56318686427062;
        v2 = -0.04787719419083, 0.55895824010234, -0.34472910285751;
        double dot_v1v2 = 0.09387402846316;
        TS_ASSERT_DELTA(dot_v1v2, R3::dot(v1, v2), precision);
    }


    void test_cross()
    {
        R3::Vector v1, v2, v1xv2;
        v1 = -0.55160549932839, -0.58291452407504,  0.12378162306543;
        v2 =  0.60842511285200, -0.97946444006248, -0.02828214306095;
        v1xv2 = 0.13772577008800,  0.05971126233738,  0.89493780662849;
        TS_ASSERT(VectorsAlmostEqual(v1xv2,
                    R3::cross(v1, v2), precision));
    }


    void test_product()
    {
        R3::Matrix M;
        R3::Vector v, prod_Mv, prod_vM;
        M = 0.459631856585519, 0.726448904209060, 0.085844209317482,
            0.806838095807669, 0.240116998848762, 0.305032463662873,
            0.019487235483683, 0.580605953831255, 0.726077578738676 ;
        v = 0.608652521912322, 0.519716469261062, 0.842577887601566;
        prod_Mv = 0.729633980805669, 0.872890409522479, 0.925388343907868;
        prod_vM = 0.715502648789536, 1.056153454546559, 0.822556602046019;
        TS_ASSERT(VectorsAlmostEqual(prod_Mv,
                    R3::mxvecproduct(M, v), precision));
        TS_ASSERT(VectorsAlmostEqual(prod_vM,
                    R3::mxvecproduct(v, M), precision));
    }


};  // class TestR3linalg

// End of file
