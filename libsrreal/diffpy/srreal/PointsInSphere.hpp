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
* Constructors:
*
*     PointsInSphere(Rmin, Rmax, a, b, c, alpha, beta, gamma)
*     ReflectionsInQminQmax(Qmin, Qmax, a, b, c, alpha, beta, gamma)
*     ReflectionsInDmaxDmin(Dmax, Dmin, a, b, c, alpha, beta, gamma)
*
*     template<class Lattice> PointsInSphere(Rmin, Rmax, const Lattice&)
*
*     where class Lattice must provide methods a(), b(), c(),
*     alpha(), beta(), gamma()
*
* Examples:
*
*     PointsInSphere sph(Rmin, Rmax, a, b, c, alpha, beta, gamma)
*     for (sph.rewind(); !sph.finished(); sph.next())
*     {
*         // lattice indices are in sph.m(), sph.n(), sph.o() or sph.mno()
*         // sph.r() is distance from origin,
*         // where sph.Rmin() < sph.r() < sph.Rmax()
*     }
*
*     ReflectionsInQminQmax ref(Qmin, Qmax, a, b, c, alpha, beta, gamma)
*     for (ReflectionsInQminQmax ref(Qmin, Qmax, a, b, c, alpha, beta, gamma);
*	   !ref.finished(); ref.next() )
*     {
*         // Miller indices are in ref.h(), ref.k(), ref.l() or ref.hkl()
*         // ref.Q() is magnitude of Q vector
*         // ref.d() is lattice plane spacing
*     }
*
* Tip: add epsilon to Rmax to avoid roundoff issues
*
* $Id$
*
*****************************************************************************/

#ifndef POINTSINSPHERE_HPP_INCLUDED
#define POINTSINSPHERE_HPP_INCLUDED

// ensure math constants get defined for MSVC
#define _USE_MATH_DEFINES
#include <cmath>

namespace diffpy {
namespace srreal {

namespace pointsinsphere {

class LatticeParameters
{
    public:

        // data
        // input arguments
        double a, b, c, alpha, beta, gamma;
        // cosines and sines of direct lattice angles
        double ca, cb, cg, sa, sb, sg;
        // reciprocal lattice and its cosines and sines
        double ar, br, cr, alphar, betar, gammar;
        double car, cbr, cgr, sar, sbr, sgr;

        // constructor
        LatticeParameters(double _a, double _b, double _c,
                double _alpha, double _beta, double _gamma);

        // methods
        // calculate all properties from current lattice parameters
        void update();
        // return a reciprocal of this lattice
        LatticeParameters reciprocal() const;

};

}   // namespace pointsinsphere


class PointsInSphere
{
    public:

        // constructors
        PointsInSphere(double rmin, double rmax,
                const pointsinsphere::LatticeParameters& _latpar);
        PointsInSphere(double rmin, double rmax,
                double _a, double _b, double _c,
                double _alpha, double _beta, double _gamma);
        template <class L>
            PointsInSphere(double rmin, double rmax, const L&);

        // methods
        // loop control
        void rewind();
        void next();
        bool finished() const;
        // data access
        const double& Rmin() const;
        const double& Rmax() const;
        const int* mno() const;
        const int& m() const;
        const int& n() const;
        const int& o() const;
        double r() const;

    private:

        // data
        // inputs
        const double _Rmin;
        const double _Rmax;
        const pointsinsphere::LatticeParameters latpar;
        // output
        int _mno[3];
        int& _m;
        int& _n;
        int& _o;
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

        // methods
        // loop advance
        void next_m();
        void next_n();
        void next_o();
        void init();
};


class ReflectionsInQminQmax
{
    public:

        // constructors
        ReflectionsInQminQmax(double _Qmin, double _Qmax,
                const pointsinsphere::LatticeParameters& _latpar);
        ReflectionsInQminQmax(double _Qmin, double _Qmax,
                double _a, double _b, double _c,
                double _alpha, double _beta, double _gamma);
        template <class L>
            ReflectionsInQminQmax(double _Qmin, double _Qmax, const L&);

        // methods
        // loop control
        void rewind();
        void next();
        bool finished() const;
        // data access
        const double& Qmin() const;
        const double& Qmax() const;
        const int* hkl() const;
        const int& h() const;
        const int& k() const;
        const int& l() const;
        double Q() const;
        double d() const;

    private:

        // data
        // inputs
        const double _Qmin;
        const double _Qmax;
        const pointsinsphere::LatticeParameters latpar;
        // composite
        PointsInSphere sph;
};


class ReflectionsInDmaxDmin : public ReflectionsInQminQmax
{
    public:

        // constructors
        ReflectionsInDmaxDmin(double dmax, double dmin,
                const pointsinsphere::LatticeParameters& _latpar);
        ReflectionsInDmaxDmin(double dmax, double dmin,
                double _a, double _b, double _c,
                double _alpha, double _beta, double _gamma);
        template <class L>
            ReflectionsInDmaxDmin(double dmax, double dmin, const L&);

        // methods
        // data access
        const double& Dmin() const;
        const double& Dmax() const;

    private:

        // data
        // inputs
        const double _Dmin;
        const double _Dmax;
};

// Template Constructor for PointsInSphere -----------------------------------

template <class L>
PointsInSphere::PointsInSphere(double rmin, double rmax, const L& lat) :
    _Rmin(rmin), _Rmax(rmax),
    latpar(lat.a(), lat.b(), lat.c(), lat.alpha(), lat.beta(), lat.gamma()),
    _m(_mno[0]), _n(_mno[1]), _o(_mno[2])
{
    init();
    rewind();
}

// Template Constructor for ReflectionsInQminQmax ----------------------------

template <class L>
ReflectionsInQminQmax::ReflectionsInQminQmax(
        double _qmin, double _qmax, const L& lat) :
	    latpar(lat.a(), lat.b(), lat.c(),
                    lat.alpha(), lat.beta(), lat.gamma()),
	    sph(_qmin*M_1_PI/2.0, _qmax*M_1_PI/2.0, latpar.reciprocal()),
	    _Qmin(_qmin), _Qmax(_qmax)
{ }

// Template Constructor for ReflectionsInDmaxDmin ----------------------------

template <class L>
ReflectionsInDmaxDmin::ReflectionsInDmaxDmin(
        double dmax, double dmin, const L& lat) :
	    ReflectionsInQminQmax(2.0*M_PI/dmax, 2.0*M_PI/dmin, lat),
	    _Dmax(dmax), _Dmin(dmin)
{ }


}   // namespace srreal
}   // namespace diffpy

#endif	// POINTSINSPHERE_HPP_INCLUDED
