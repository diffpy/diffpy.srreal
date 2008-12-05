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
* classes PointsInSphere, ReflectionsInQminQmax, ReflectionsInDmaxDmin
*
* Comments: sequencers for lattice points insided 3D sphere
*
* $Id$
*
***********************************************************************/

#include <algorithm>
#include "PointsInSphere.h"

using namespace NS_POINTSINSPHERE;

////////////////////////////////////////////////////////////////////////
// LatticeParameters
////////////////////////////////////////////////////////////////////////

LatticeParameters::LatticeParameters( double _a, double _b, double _c,
	double _alpha, double _beta, double _gamma ) :
	    a(_a), b(_b), c(_c),
	    alpha(_alpha), beta(_beta), gamma(_gamma)
{
    update();
}

LatticeParameters& LatticeParameters::update()
{
    ca = cosd(alpha); cb = cosd(beta); cg = cosd(gamma);
    sa = sind(alpha); sb = sind(beta); sg = sind(gamma);
    // Vunit is a volume of unit cell with a=b=c=1
    const double Vunit = sqrt(1.0 + 2.0*ca*cb*cg - ca*ca - cb*cb - cg*cg);
    ar = sa/(a*Vunit);
    br = sb/(b*Vunit);
    cr = sg/(c*Vunit);
    car = (cb*cg - ca)/(sb*sg); sar = sqrt(1.0 - car*car);
    cbr = (ca*cg - cb)/(sa*sg); sbr = sqrt(1.0 - cbr*cbr);
    cgr = (ca*cb - cg)/(sa*sb); sgr = sqrt(1.0 - cgr*cgr);
    alphar = 180.0/M_PI*acos(car);
    betar = 180.0/M_PI*acos(cbr);
    gammar = 180.0/M_PI*acos(cgr);
    return *this;
}

LatticeParameters LatticeParameters::reciprocal() const
{
    using namespace std;
    LatticeParameters rec(*this);
    swap(rec.a, rec.ar);
    swap(rec.b, rec.br);
    swap(rec.c, rec.cr);
    swap(rec.alpha, rec.alphar);
    swap(rec.beta, rec.betar);
    swap(rec.gamma, rec.gammar);
    swap(rec.ca, rec.car);
    swap(rec.cb, rec.cbr);
    swap(rec.cg, rec.cgr);
    swap(rec.sa, rec.sar);
    swap(rec.sb, rec.sbr);
    swap(rec.sg, rec.sgr);
    return rec;
}


////////////////////////////////////////////////////////////////////////
// PointsInSphere
////////////////////////////////////////////////////////////////////////

PointsInSphere::PointsInSphere( double _Rmin, double _Rmax,
	const LatticeParameters& _latpar ) :
	    m(mno[0]), n(mno[1]), o(mno[2]),
	    Rmin(_Rmin), Rmax(_Rmax), latpar(_latpar)
{
    init();
    rewind();
}

PointsInSphere::PointsInSphere( double _Rmin, double _Rmax,
	double _a, double _b, double _c,
	double _alpha, double _beta, double _gamma ) :
	    m(mno[0]), n(mno[1]), o(mno[2]),
	    Rmin(_Rmin), Rmax(_Rmax),
	    latpar(_a, _b, _c, _alpha, _beta, _gamma)
{
    init();
    rewind();
}

void PointsInSphere::init()
{
    RminSquare = (Rmin < 0.0) ? -(Rmin*Rmin) : Rmin*Rmin;
    RmaxSquare = (Rmax < 0.0) ? -(Rmax*Rmax) : Rmax*Rmax;
    dn0dm = latpar.cgr*latpar.br/latpar.ar;
    do0dm = latpar.cbr*latpar.cr/latpar.ar;
    // 2D reciprocal parameters in bc plane
    b2r = 1.0/(latpar.b*latpar.sa);
    c2r = 1.0/(latpar.c*latpar.sa);
    ca2r = -latpar.ca;
    do0dn = ca2r*c2r/b2r;
    // 1D reciprocal along c axis
    c1r = 1.0/latpar.c;
}

void PointsInSphere::rewind()
{
    mHalfSpan = Rmax*latpar.ar;
    hi_m = int(ceil(mHalfSpan));
    m = -hi_m;
    // make indices n, o invalid, reset the neares point
    n = hi_n = 0;
    o = hi_o = outside_o = 0;
    n0plane = o0plane = o0line = 0.0;
    // unset excluded zone
    oExclHalfSpan = 0.0;
    // get the first inside point
    next_o();
}

void PointsInSphere::next_o()
{
    do
    {
	o++;
	if (o < hi_o)
	{
	    return;
	}
	if (hi_o != outside_o)
	{
	    hi_o = outside_o;
	    o = int( ceil(o0line+oExclHalfSpan) ) - 1;
	    continue;
	}
	next_n();
    }
    while (!finished());
}

void PointsInSphere::next_n()
{
    do
    {
	n++;
	if (n < hi_n)
	{
	    o0line = o0plane + (n-n0plane)*do0dn;
	    double RlineSquare = RplaneSquare - pow((n-n0plane)/b2r,2);
	    oHalfSpan = RlineSquare > 0.0 ? sqrt(RlineSquare)*c1r : 0.0;
	    // parentheses improve round-off errors around [0,0,0]
	    double RExclSquare = RminSquare + (RlineSquare - RmaxSquare);
	    oExclHalfSpan = RExclSquare > 0.0 ? sqrt(RExclSquare)*c1r : 0.0;
	    o = int(floor(o0line - oHalfSpan));
	    outside_o = int(ceil(o0line + oHalfSpan));
	    hi_o = outside_o;
	    if (oExclHalfSpan)
	    {
		int hole_o = int(ceil(o0line - oExclHalfSpan));
		if (fabs(hole_o-o0line) < oExclHalfSpan)    hi_o = hole_o;
	    }
	    return;
	}
	next_m();
    }
    while (!finished());
}

void PointsInSphere::next_m()
{
    m++;
    if (finished())
    {
	return;
    }
    // not finished here
    n0plane = m*dn0dm;
    o0plane = m*do0dm;
    RplaneSquare = RmaxSquare - pow(m/latpar.ar,2);
    nHalfSpan = RplaneSquare > 0.0 ? sqrt(RplaneSquare)*b2r : 0.0;
    n = int(floor(n0plane - nHalfSpan));
    hi_n = int(ceil(n0plane + nHalfSpan));
}

double PointsInSphere::r() const
{
    const double &a = latpar.a, &b = latpar.b, &c = latpar.c;
    const double &ca = latpar.ca, &cb = latpar.cb, &cg = latpar.cg;
    return sqrt( m*m*a*a + n*n*b*b + o*o*c*c
	    + 2*m*n*a*b*cg + 2*m*o*a*c*cb + 2*n*o*b*c*ca );
}


////////////////////////////////////////////////////////////////////////
// ReflectionsInQminQmax
////////////////////////////////////////////////////////////////////////

ReflectionsInQminQmax::ReflectionsInQminQmax( double _Qmin, double _Qmax,
	const LatticeParameters& _latpar ) :
	    Qmin(_Qmin), Qmax(_Qmax),
	    latpar(_latpar),
	    sph(Qmin*M_1_PI/2.0, Qmax*M_1_PI/2.0, latpar.reciprocal()),
	    hkl(sph.mno), h(hkl[0]), k(hkl[1]), l(hkl[2])
{ }

ReflectionsInQminQmax::ReflectionsInQminQmax( double _Qmin, double _Qmax,
	double _a, double _b, double _c,
	double _alpha, double _beta, double _gamma ) :
	    Qmin(_Qmin), Qmax(_Qmax),
	    latpar(_a, _b, _c, _alpha, _beta, _gamma),
	    sph(Qmin*M_1_PI/2.0, Qmax*M_1_PI/2.0, latpar.reciprocal()),
	    hkl(sph.mno), h(hkl[0]), k(hkl[1]), l(hkl[2])
{ }


////////////////////////////////////////////////////////////////////////
// ReflectionsInDmaxDmin
////////////////////////////////////////////////////////////////////////

ReflectionsInDmaxDmin::ReflectionsInDmaxDmin( double _Dmax, double _Dmin,
	const LatticeParameters& _latpar ) :
	    ReflectionsInQminQmax(2.0*M_PI/_Dmax, 2.0*M_PI/_Dmin, _latpar),
	    Dmax(_Dmax), Dmin(_Dmin)
{ }

ReflectionsInDmaxDmin::ReflectionsInDmaxDmin( double _Dmax, double _Dmin,
	double _a, double _b, double _c,
	double _alpha, double _beta, double _gamma ) :
	    ReflectionsInQminQmax( 2.0*M_PI/_Dmax, 2.0*M_PI/_Dmin,
		    _a, _b, _c, _alpha, _beta, _gamma ),
	    Dmax(_Dmax), Dmin(_Dmin)
{ }

// End of file
