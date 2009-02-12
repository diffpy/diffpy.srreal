/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 trustees of the Michigan State University
*                   All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* classes PointsInSphere, ReflectionsInQminQmax, ReflectionsInDmaxDmin
*
* Comments: sequencers for lattice points insided 3D sphere
*
* $Id$
*
*****************************************************************************/

#include <algorithm>
#include <diffpy/srreal/PointsInSphere.hpp>
#include <diffpy/mathutils.hpp>

using namespace diffpy::srreal;
using pointsinsphere::LatticeParameters;

//////////////////////////////////////////////////////////////////////////////
// class LatticeParameters
//////////////////////////////////////////////////////////////////////////////

// Constructor ---------------------------------------------------------------

LatticeParameters::LatticeParameters( double _a, double _b, double _c,
	double _alpha, double _beta, double _gamma ) :
	    a(_a), b(_b), c(_c),
	    alpha(_alpha), beta(_beta), gamma(_gamma)
{
    update();
}

// Public Methods ------------------------------------------------------------

void LatticeParameters::update()
{
    using namespace diffpy::mathutils;
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

//////////////////////////////////////////////////////////////////////////////
// class PointsInSphere
//////////////////////////////////////////////////////////////////////////////

// Constructors --------------------------------------------------------------

PointsInSphere::PointsInSphere(double rmin, double rmax,
	const LatticeParameters& _latpar ) :
            _Rmin(rmin), _Rmax(rmax),
	    latpar(_latpar),
            _m(_mno[0]), _n(_mno[1]), _o(_mno[2])
{
    init();
    rewind();
}


PointsInSphere::PointsInSphere(double rmin, double rmax,
	double _a, double _b, double _c,
	double _alpha, double _beta, double _gamma) :
	    _Rmin(rmin), _Rmax(rmax),
	    latpar(_a, _b, _c, _alpha, _beta, _gamma),
            _m(_mno[0]), _n(_mno[1]), _o(_mno[2])
{
    init();
    rewind();
}

// Public Methods ------------------------------------------------------------

// loop control

void PointsInSphere::rewind()
{
    mHalfSpan = Rmax()*latpar.ar;
    hi_m = int(ceil(mHalfSpan));
    this->_m = -hi_m;
    // make indices n, o invalid, reset the neares point
    this->_n = hi_n = 0;
    this->_o = hi_o = outside_o = 0;
    n0plane = o0plane = o0line = 0.0;
    // unset excluded zone
    oExclHalfSpan = 0.0;
    // get the first inside point
    next_o();
}


void PointsInSphere::next()
{
    next_o();
}


bool PointsInSphere::finished() const
{
    return !(m() < this->hi_m);
}

// data access

const double& PointsInSphere::Rmin() const
{
    return this->_Rmin;
}


const double& PointsInSphere::Rmax() const
{
    return this->_Rmax;
}


const int* PointsInSphere::mno() const
{
    return this->_mno;
}


const int& PointsInSphere::m() const
{
    return this->_mno[0];
}


const int& PointsInSphere::n() const
{
    return this->_mno[1];
}


const int& PointsInSphere::o() const
{
    return this->_mno[2];
}


double PointsInSphere::r() const
{
    const double &a = latpar.a, &b = latpar.b, &c = latpar.c;
    const double &ca = latpar.ca, &cb = latpar.cb, &cg = latpar.cg;
    return sqrt( m()*m()*a*a + n()*n()*b*b + o()*o()*c*c
	    + 2*m()*n()*a*b*cg + 2*m()*o()*a*c*cb + 2*n()*o()*b*c*ca );
}

// Private Methods -----------------------------------------------------------

void PointsInSphere::next_m()
{
    this->_m += 1;
    if (finished())
    {
	return;
    }
    // not finished here
    n0plane = m()*dn0dm;
    o0plane = m()*do0dm;
    RplaneSquare = RmaxSquare - pow(m()/latpar.ar,2);
    nHalfSpan = RplaneSquare > 0.0 ? sqrt(RplaneSquare)*b2r : 0.0;
    this->_n = int(floor(n0plane - nHalfSpan));
    hi_n = int(ceil(n0plane + nHalfSpan));
}


void PointsInSphere::next_n()
{
    do
    {
	this->_n += 1;
	if (n() < hi_n)
	{
	    o0line = o0plane + (n()-n0plane)*do0dn;
	    double RlineSquare = RplaneSquare - pow((n()-n0plane)/b2r,2);
	    oHalfSpan = RlineSquare > 0.0 ? sqrt(RlineSquare)*c1r : 0.0;
	    // parentheses improve round-off errors around [0,0,0]
	    double RExclSquare = (RlineSquare - RmaxSquare) + RminSquare;
	    oExclHalfSpan = RExclSquare > 0.0 ? sqrt(RExclSquare)*c1r : 0.0;
	    this->_o = int(floor(o0line - oHalfSpan));
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


void PointsInSphere::next_o()
{
    do
    {
	this->_o += 1;
	if (o() < hi_o)
	{
	    return;
	}
	if (hi_o != outside_o)
	{
	    hi_o = outside_o;
	    this->_o = int( ceil(o0line+oExclHalfSpan) ) - 1;
	    continue;
	}
	next_n();
    }
    while (!finished());
}


void PointsInSphere::init()
{
    RminSquare = (_Rmin < 0.0) ? -(_Rmin*_Rmin) : _Rmin*_Rmin;
    RmaxSquare = (_Rmax < 0.0) ? -(_Rmax*_Rmax) : _Rmax*_Rmax;
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

//////////////////////////////////////////////////////////////////////////////
// class ReflectionsInQminQmax
//////////////////////////////////////////////////////////////////////////////

// Constructors --------------------------------------------------------------

ReflectionsInQminQmax::ReflectionsInQminQmax(double qmin, double qmax,
	const LatticeParameters& _latpar) :
	    _Qmin(qmin), _Qmax(qmax),
	    latpar(_latpar),
	    sph(qmin*M_1_PI/2.0, qmax*M_1_PI/2.0, latpar.reciprocal())
{ }


ReflectionsInQminQmax::ReflectionsInQminQmax(double qmin, double qmax,
	double _a, double _b, double _c,
	double _alpha, double _beta, double _gamma ) :
	    _Qmin(qmin), _Qmax(qmax),
	    latpar(_a, _b, _c, _alpha, _beta, _gamma),
	    sph(qmin*M_1_PI/2.0, qmax*M_1_PI/2.0, latpar.reciprocal())
{ }

// Public Methods ------------------------------------------------------------

// loop control

void ReflectionsInQminQmax::rewind()
{
    this->sph.rewind();
}


void ReflectionsInQminQmax::next()
{
    this->sph.next();
}


bool ReflectionsInQminQmax::finished() const
{
    return this->sph.finished();
}

// data access

const double& ReflectionsInQminQmax::Qmin() const
{
    return this->_Qmin;
}


const double& ReflectionsInQminQmax::Qmax() const
{
    return this->_Qmax;
}


const int* ReflectionsInQminQmax::hkl() const
{
    return this->sph.mno();
}


const int& ReflectionsInQminQmax::h() const
{
    return this->sph.m();
}


const int& ReflectionsInQminQmax::k() const
{
    return this->sph.n();
}


const int& ReflectionsInQminQmax::l() const
{
    return this->sph.o();
}


double ReflectionsInQminQmax::Q() const
{
    return 2.0*M_PI*sph.r();
}


double ReflectionsInQminQmax::d() const
{
    return 1.0/sph.r();
}

//////////////////////////////////////////////////////////////////////////////
// class ReflectionsInDmaxDmin
//////////////////////////////////////////////////////////////////////////////

// Constructors --------------------------------------------------------------

ReflectionsInDmaxDmin::ReflectionsInDmaxDmin(double dmax, double dmin,
	const LatticeParameters& _latpar) :
	    ReflectionsInQminQmax(2.0*M_PI/dmax, 2.0*M_PI/dmin, _latpar),
            _Dmin(dmin), _Dmax(dmax)
{ }


ReflectionsInDmaxDmin::ReflectionsInDmaxDmin(double dmax, double dmin,
	double _a, double _b, double _c,
	double _alpha, double _beta, double _gamma) :
	    ReflectionsInQminQmax(2.0*M_PI/dmax, 2.0*M_PI/dmin,
		    _a, _b, _c, _alpha, _beta, _gamma),
	    _Dmin(dmin), _Dmax(dmax)
{ }

// Public Methods ------------------------------------------------------------

// data access

const double& ReflectionsInDmaxDmin::Dmin() const
{
    return this->_Dmin;
}


const double& ReflectionsInDmaxDmin::Dmax() const
{
    return this->_Dmax;
}

// End of file
