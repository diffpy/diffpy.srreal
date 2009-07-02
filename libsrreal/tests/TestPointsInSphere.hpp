/***********************************************************************
*
* pdffit2           by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2006 trustees of the Michigan State University
*                   All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
************************************************************************
*
* Unit tests for PointsInSphere module
*
* Comments:
*
* $Id$
*
***********************************************************************/

#include <algorithm>

#include <cxxtest/TestSuite.h>

#include <diffpy/srreal/PointsInSphere.hpp>

using namespace std;
using namespace diffpy::srreal;

namespace {

const double eps = 1.0e-12;

struct vidxgroup
{
    double vijk[4];
    vidxgroup(double v, const int* ijk)
    {
	vijk[0] = v;
	for (size_t i = 0; i != 3; ++i) { vijk[i+1] = ijk[i]; }
    }
    vidxgroup(double v, int i, int j, int k)
    {
	vijk[0] = v; vijk[1] = i; vijk[2] = j; vijk[3] = k;
    }
};

bool operator<(const vidxgroup &x, const vidxgroup &y)
{
    return (x.vijk[0] < y.vijk[0] - eps) ||
	    lexicographical_compare(x.vijk+1, x.vijk+4, y.vijk+1, y.vijk+4);
}

bool operator==(const vidxgroup &x, const vidxgroup &y)
{
    bool eq = (fabs(x.vijk[0] - y.vijk[0]) < eps) &&
	    equal(x.vijk+1, x.vijk+4, y.vijk+1);
    return eq;
}

ostream& operator<<(ostream &s, const vidxgroup &x)
{
    return s << "<" << x.vijk[0] << ";" << int(x.vijk[1])
	<< ',' << int(x.vijk[2]) << ',' << int(x.vijk[3]) << '>';
}

}   // namespace

////////////////////////////////////////////////////////////////////////
// TestPointsInSphere
////////////////////////////////////////////////////////////////////////

using diffpy::srreal::pointsinsphere::LatticeParameters;

class TestPointsInSphere : public CxxTest::TestSuite
{

private:

    LatticeParameters* latpar;

public:

    void setUp()
    {
	latpar = new LatticeParameters(1, 1, 1, 90, 90, 90);
    }

    void tearDown()
    {
	delete latpar;
    }

private:

    int count(double Rmin, double Rmax)
    {
	int c = 0;
	PointsInSphere sph(Rmin, Rmax, *latpar);
	for (; !sph.finished(); sph.next())     ++c;
	return c;
    }

    vector<vidxgroup> sortedPoints(double Rmin, double Rmax)
    {
	vector<vidxgroup> ridx;
	for (   PointsInSphere sph(Rmin, Rmax, *latpar);
		not sph.finished(); sph.next()  )
	{
	    ridx.push_back(vidxgroup(sph.r(), sph.mno()));
	}
	sort(ridx.begin(), ridx.end());
	return ridx;
    }

public:

    void test_Cubic()
    {
	latpar->a = latpar->b = latpar->c = 1.0;
	latpar->alpha = latpar->beta = latpar->gamma = 90.0;
	latpar->update();
	TS_ASSERT_EQUALS(0, count(0.0, 0.0));
	TS_ASSERT_EQUALS(0, count(eps, 0.5));
	TS_ASSERT_EQUALS(0, count(1.0 + eps, 1.1));
	TS_ASSERT_EQUALS(1, count(0.0, eps));
	TS_ASSERT_EQUALS(7, count(0.0, 1 + eps));
	TS_ASSERT_EQUALS(19, count(0.0, sqrt(2.0) + eps));
	TS_ASSERT_EQUALS(12, count(1.0 + eps, sqrt(2.0) + eps));
    }

    void test_Orthorombic()
    {
	latpar->a = 1.0; latpar->b = 2.0; latpar->c = 3.0;
	latpar->alpha = latpar->beta = latpar->gamma = 90.0;
	latpar->update();
	TS_ASSERT_EQUALS(3, count(0.0, 1.1));
	TS_ASSERT_EQUALS(4, count(1.9, 2.1));
	vidxgroup ep[] = {
	    vidxgroup(0, 0, 0, 0),
	    vidxgroup(1, -1, 0, 0),
	    vidxgroup(1, 1, 0, 0),
	    vidxgroup(2, -2, 0, 0),
	    vidxgroup(2, 0, -1, 0),
	    vidxgroup(2, 0, 1, 0),
	    vidxgroup(2, 2, 0, 0),
	    vidxgroup(sqrt(5.0), -1, -1, 0),
	    vidxgroup(sqrt(5.0), -1, 1, 0),
	    vidxgroup(sqrt(5.0), 1, -1, 0),
	    vidxgroup(sqrt(5.0), 1, 1, 0),
	    vidxgroup(sqrt(8.0), -2, -1, 0),
	    vidxgroup(sqrt(8.0), -2, 1, 0),
	    vidxgroup(sqrt(8.0), 2, -1, 0),
	    vidxgroup(sqrt(8.0), 2, 1, 0),
	    vidxgroup(3, -3, 0, 0),
	    vidxgroup(3, 0, 0, -1),
	    vidxgroup(3, 0, 0, 1),
	    vidxgroup(3, 3, 0, 0),
	};
	vector<vidxgroup> exp_pts(ep, ep + sizeof(ep)/sizeof(vidxgroup));
	vector<vidxgroup> act_pts = sortedPoints(0.0, 3.0+eps);
	TS_ASSERT_EQUALS(exp_pts.size(), act_pts.size());
	for (size_t i = 0; i != exp_pts.size(); ++i)
	{
	    TS_ASSERT_EQUALS(exp_pts[i], act_pts[i]);
	}
    }

    void test_Hexagonal()
    {
	latpar->a = 1.0; latpar->b = 1.0; latpar->c = 2.0;
	latpar->alpha = latpar->beta = 90.0; latpar->gamma = 120.0;
	latpar->update();
	TS_ASSERT_EQUALS(7, count(0.0, 1+eps));
	vidxgroup ep[] = {
	    vidxgroup(0, 0, 0, 0),
	    vidxgroup(1, -1, -1, 0),
	    vidxgroup(1, -1, 0, 0),
	    vidxgroup(1, 0, -1, 0),
	    vidxgroup(1, 0, 1, 0),
	    vidxgroup(1, 1, 0, 0),
	    vidxgroup(1, 1, 1, 0),
	    vidxgroup(sqrt(3.0), -2, -1, 0),
	    vidxgroup(sqrt(3.0), -1, -2, 0),
	    vidxgroup(sqrt(3.0), -1, 1, 0),
	    vidxgroup(sqrt(3.0), 1, -1, 0),
	    vidxgroup(sqrt(3.0), 1, 2, 0),
	    vidxgroup(sqrt(3.0), 2, 1, 0),
	    vidxgroup(2, -2, -2, 0),
	    vidxgroup(2, -2, 0, 0),
	    vidxgroup(2, 0, -2, 0),
	    vidxgroup(2, 0, 0, -1),
	    vidxgroup(2, 0, 0, 1),
	    vidxgroup(2, 0, 2, 0),
	    vidxgroup(2, 2, 0, 0),
	    vidxgroup(2, 2, 2, 0),
	};
	vector<vidxgroup> exp_pts(ep, ep + sizeof(ep)/sizeof(vidxgroup));
	vector<vidxgroup> act_pts = sortedPoints(0.0, 2.0+eps);
	TS_ASSERT_EQUALS(exp_pts.size(), act_pts.size());
	for (size_t i = 0; i != exp_pts.size(); ++i)
	{
	    TS_ASSERT_EQUALS(exp_pts[i], act_pts[i]);
	}
    }

    void test_FCC()
    {
	latpar->a = latpar->b = latpar->c = sqrt(0.5);
	latpar->alpha = latpar->beta = latpar->gamma = 60.0;
	latpar->update();
	TS_ASSERT_EQUALS(13, count(0.0, sqrt(0.5)+eps));
	TS_ASSERT_EQUALS(19, count(0.0, 1.0+eps));
    }

};  // class TestPointsInSphere

// End of file
