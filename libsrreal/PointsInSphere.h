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
* Constructors:
*
*     PointsInSphere(Rmin, Rmax, a, b, c, alpha, beta, gamma)
*     ReflectionsInQminQmax(Qmin, Qmax, a, b, c, alpha, beta, gamma)
*     ReflectionsInDmaxDmin(Dmax, Dmin, a, b, c, alpha, beta, gamma)
*
* Examples:
*
*     PointsInSphere sph(Rmin, Rmax, a, b, c, alpha, beta, gamma)
*     for (sph.rewind(); !sph.finished(); sph.next())
*     { 
*         // lattice indices are in sph.m, sph.n, sph.o or sph.mno[3]
*         // sph.r() is distance from origin,
*         // where Rmin < sph.r() < Rmax
*     }
*
*     ReflectionsInQminQmax refl(Qmin, Qmax, a, b, c, alpha, beta, gamma)
*     for (ReflectionsInQminQmax ref(Qmin, Qmax, a, b, c, alpha, beta, gamma);
*	   !ref.finished(); ref.next() )
*     { 
*         // Miller indices are in ref.h, ref.k, ref.l or ref.hkl[3]
*         // ref.Q() is magnitude of Q vector
*         // ref.d() is lattice plane spacing
*     }
*
* Tip: add epsilon to Rmax to avoid roundoff issues
*
* $Id$
*
***********************************************************************/

#ifndef POINTSINSPHERE_H_INCLUDED
#define POINTSINSPHERE_H_INCLUDED

// ensure math constants get defined for MSVC 
#define _USE_MATH_DEFINES
#include <cmath>


namespace NS_POINTSINSPHERE {

class LatticeParameters
{
public:
    LatticeParameters( double _a, double _b, double _c,
	    double _alpha, double _beta, double _gamma );
    // calculate all properties from current lattice parameters
    LatticeParameters& update();
    // return a reciprocal of this lattice
    LatticeParameters reciprocal() const;
    // input arguments
    double a, b, c, alpha, beta, gamma;
    // cosines and sines of direct lattice angles
    double ca, cb, cg, sa, sb, sg;
    // reciprocal lattice and its cosines and sines
    double ar, br, cr, alphar, betar, gammar;
    double car, cbr, cgr, sar, sbr, sgr;
private:
    // helper functions
    inline double cosd(double x) { return cos(M_PI/180.0*x); }
    inline double sind(double x) { return sin(M_PI/180.0*x); }
};

class PointsInSphere
{
public:
    PointsInSphere( double _Rmin, double _Rmax,
	    const LatticeParameters& _latpar );
    PointsInSphere( double _Rmin, double _Rmax,
	    double _a, double _b, double _c,
	    double _alpha, double _beta, double _gamma );
    void rewind();
    inline void next()
    {
	next_o();
    }
    inline bool finished()
    {
	return !(m < hi_m);
    }
    // mno array and m, n, o aliases are supposed to be read only
    int mno[3];
    int &m, &n, &o;
    double r() const;
    // input arguments
    const double Rmin, Rmax;
    const LatticeParameters latpar;
private:
    // loop advance functions
    void next_m();
    void next_n();
    void next_o();
    void init();
    // calculated constants set by init()
    double RminSquare, RmaxSquare;
    // 2D reciprocal parameters and cosine in bc plane
    double b2r, c2r, ca2r;
    // reciprocal c
    double c1r;
    // offset of the nearest point to [0,0,0]
    double dn0dm, do0dm, do0dn;
    // loop variables
    double n0plane, o0plane, o0line;
    double mHalfSpan, nHalfSpan, oHalfSpan;
    // o indices excluded due to Rmin
    double oExclHalfSpan;
    int hi_m, hi_n, hi_o, outside_o;
    double RplaneSquare;
};

class ReflectionsInQminQmax
{
public:
    ReflectionsInQminQmax( double _Qmin, double _Qmax,
	    const LatticeParameters& _latpar );
    ReflectionsInQminQmax( double _Qmin, double _Qmax,
	    double _a, double _b, double _c,
	    double _alpha, double _beta, double _gamma );
    inline void rewind()
    {
	sph.rewind();
    }
    inline void next()
    {
	sph.next();
    }
    inline bool finished()
    {
	return sph.finished();
    }
    // input arguments
    const double Qmin, Qmax;
    const LatticeParameters latpar;
private:
    // sph must be initialized before hkl and h, k, l
    PointsInSphere sph;
public:
    // hkl array and h, k, l aliases are supposed to be read only
    int *hkl;
    int &h, &k, &l;
    inline double Q() const
    {
	return 2.0*M_PI*sph.r();
    }
    inline double d() const
    {
	return 1.0/sph.r();
    }
};

class ReflectionsInDmaxDmin : ReflectionsInQminQmax
{
public:
    ReflectionsInDmaxDmin( double _Dmax, double _Dmin,
	    const LatticeParameters& _latpar );
    ReflectionsInDmaxDmin( double _Dmax, double _Dmin,
	    double _a, double _b, double _c,
	    double _alpha, double _beta, double _gamma );
    // input arguments
    const double Dmax, Dmin;
};


}	// namespace NS_POINTSINSPHERE

using NS_POINTSINSPHERE::PointsInSphere;
using NS_POINTSINSPHERE::ReflectionsInQminQmax;
using NS_POINTSINSPHERE::ReflectionsInDmaxDmin;

#endif	// POINTSINSPHERE_H_INCLUDED
