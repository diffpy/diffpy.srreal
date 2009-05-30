/***********************************************************************
* Short Title: declaration of Lattice class
*
* Comments: class for general coordinate system
*
* $Id$
*
* <license text>
***********************************************************************/

#ifndef LATTICE_HPP_INCLUDED
#define LATTICE_HPP_INCLUDED

#include <diffpy/mathutils.hpp>
#include <diffpy/srreal/R3linalg.hpp>

namespace diffpy {
namespace srreal {

class Lattice
{
    public:

        // constructors

        Lattice();
        Lattice(double a, double b, double c,
                double alpha, double beta, double gamma);
        // create from base vectors
        template <class V>
            Lattice(const V& va, const V& vb, const V& vc);

        // methods

        // set lattice parameters
        void setLatPar(
                double a, double b, double c,
                double alpha, double beta, double gamma );
        // set lattice base vectors
        void setLatBase(
                const R3::Vector& va,
                const R3::Vector& vb,
                const R3::Vector& vc);
        template <class V>
            void setLatBase(const V& va, const V& vb, const V& vc);
        // direct lattice
        double a() const;
        double b() const;
        double c() const;
        double alpha() const;
        double beta() const;
        double gamma() const;
        double cosalpha() const;
        double cosbeta() const;
        double cosgamma() const;
        double sinalpha() const;
        double sinbeta() const;
        double singamma() const;
        const R3::Vector& va() const;
        const R3::Vector& vb() const;
        const R3::Vector& vc() const;
        double volumeNormal() const;
        double volume() const;
        // reciprocal lattice
        double ar() const;
        double br() const;
        double cr() const;
        double alphar() const;
        double betar() const;
        double gammar() const;
        double cosalphar() const;
        double cosbetar() const;
        double cosgammar() const;
        double sinalphar() const;
        double sinbetar() const;
        double singammar() const;
        const R3::Vector& var() const;
        const R3::Vector& vbr() const;
        const R3::Vector& vcr() const;
        // lattice related tensors
        // metrics tensor
        const R3::Matrix& metrics() const;
        // matrix of base vectors, base() = stdbase() * baserot()
        const R3::Matrix& base() const;
        // standard base vectors
        const R3::Matrix& stdbase() const;
        // base rotation matrix
        const R3::Matrix& baserot() const;
        // inverse of base matrix
        const R3::Matrix& recbase() const;
        // vector operations using lattice coordinates
        template <class V>
            double dot(const V& u, const V& v) const;
        template <class V>
            double norm(const V& u) const;
        template <class V>
            double distance(const V& u, const V& v) const;
        // angle in degrees
        template <class V>
            double angledeg(const V& u, const V& v) const;
        // angle in radians
        template <class V>
            double anglerad(const V& u, const V& v) const;
        // conversion of coordinates and tensors
        const R3::Vector& cartesian(const R3::Vector& lv) const;
        template <class V>
            const R3::Vector& cartesian(const V& lv) const;
        const R3::Vector& fractional(const R3::Vector& cv) const;
        template <class V>
            const R3::Vector& fractional(const V& cv) const;
        const R3::Vector& ucvCartesian(const R3::Vector& cv) const;
        template <class V>
            const R3::Vector& ucvCartesian(const V& cv) const;
        const R3::Vector& ucvFractional(const R3::Vector& lv) const;
        template <class V>
            const R3::Vector& ucvFractional(const V& lv) const;
        const R3::Matrix& cartesianMatrix(const R3::Matrix& Ml) const;
        const R3::Matrix& fractionalMatrix(const R3::Matrix& Mc) const;
        // largest cell diagonal in fractional coordinates
        const R3::Vector& ucMaxDiagonal() const;
        double ucMaxDiagonalLength() const;

    private:

        // methods
        void updateMetrics();
        void updateStandardBase();

        // data - direct lattice parameters
        double ma, mb, mc;
        double malpha, mbeta, mgamma;
        double mcosa, mcosb, mcosg;
        double msina, msinb, msing;
        R3::Vector mva, mvb, mvc;

        // data - reciprocal lattice parameters
        double mar, mbr, mcr;
        double malphar, mbetar, mgammar;
        double mcosar, mcosbr, mcosgr;
        double msinar, msinbr, msingr;
        R3::Vector mvar, mvbr, mvcr;

        // data - tensors
        // mbase = mstdbase * mbaserot
        R3::Matrix mmetrics;        // metrics tensor
        R3::Matrix mbase;           // lattice base
        R3::Matrix mstdbase;        // standard unit cell base
        R3::Matrix mbaserot;        // base rotation matrix
        R3::Matrix mrecbase;        // inverse of base matrix
        // base multiplied by magnitudes of reciprocal vectors
        R3::Matrix mnormbase;
        R3::Matrix mrecnormbase;    // inverse of mnormbase

};


//////////////////////////////////////////////////////////////////////////////
// Definitions
//////////////////////////////////////////////////////////////////////////////

// Template Constructor ------------------------------------------------------

template <class V>
Lattice::Lattice(const V& va0, const V& vb0, const V& vc0)
{
    this->setLatBase(va0, vb0, vc0);
}

// Template and Inline Methods -----------------------------------------------

template <class V>
void Lattice::setLatBase(const V& va0, const V& vb0, const V& vc0)
{
    R3::Vector va1(va0[0], va0[1], va0[2]);
    R3::Vector vb1(vb0[0], vb0[1], vb0[2]);
    R3::Vector vc1(vc0[0], vc0[1], vc0[2]);
    this->setLatBase(va1, vb1, vc1);
}

// direct lattice

inline
double Lattice::a() const
{
    return ma;
}


inline
double Lattice::b() const
{
    return mb;
}


inline
double Lattice::c() const
{
    return mc;
}


inline
double Lattice::alpha() const
{
    return malpha;
}


inline
double Lattice::beta() const
{
    return mbeta;
}


inline
double Lattice::gamma() const
{
    return mgamma;
}


inline
double Lattice::cosalpha() const
{
    return mcosa;
}


inline
double Lattice::cosbeta() const
{
    return mcosb;
}


inline
double Lattice::cosgamma() const
{
    return mcosg;
}


inline
double Lattice::sinalpha() const
{
    return msina;
}


inline
double Lattice::sinbeta() const
{
    return msinb;
}


inline
double Lattice::singamma() const
{
    return msing;
}


inline
const R3::Vector& Lattice::va() const
{
    return mva;
}


inline
const R3::Vector& Lattice::vb() const
{
    return mvb;
}


inline
const R3::Vector& Lattice::vc() const
{
    return mvc;
}

// reciprocal lattice

inline
double Lattice::ar() const
{
    return mar;
}


inline
double Lattice::br() const
{
    return mbr;
}


inline
double Lattice::cr() const
{
    return mcr;
}


inline
double Lattice::alphar() const
{
    return malphar;
}


inline
double Lattice::betar() const
{
    return mbetar;
}


inline
double Lattice::gammar() const
{
    return mgammar;
}


inline
double Lattice::cosalphar() const
{
    return mcosar;
}


inline
double Lattice::cosbetar() const
{
    return mcosbr;
}


inline
double Lattice::cosgammar() const
{
    return mcosgr;
}


inline
double Lattice::sinalphar() const
{
    return msinar;
}


inline
double Lattice::sinbetar() const
{
    return msinbr;
}


inline
double Lattice::singammar() const
{
    return msingr;
}


inline
const R3::Vector& Lattice::var() const
{
    return mvar;
}


inline
const R3::Vector& Lattice::vbr() const
{
    return mvbr;
}


inline
const R3::Vector& Lattice::vcr() const
{
    return mvcr;
}

// lattice related tensors

inline
const R3::Matrix& Lattice::metrics() const
{
    return mmetrics;
}


inline
const R3::Matrix& Lattice::base() const
{
    return mbase;
}


inline
const R3::Matrix& Lattice::stdbase() const
{
    return mstdbase;
}


inline
const R3::Matrix& Lattice::baserot() const
{
    return mbaserot;
}


inline
const R3::Matrix& Lattice::recbase() const
{
    return mrecbase;
}


template <class V>
double Lattice::dot(const V& u, const V& v) const
{
    const R3::Matrix& M = this->metrics();
    double dp =
        u[0] * v[0] * M(0,0) +
        u[1] * v[1] * M(1,1) +
        u[2] * v[2] * M(2,2) +
        (u[0] * v[1] + u[1] * v[0]) * M(0,1) +
        (u[0] * v[2] + u[2] * v[0]) * M(0,2) +
        (u[2] * v[1] + u[1] * v[2]) * M(1,2);
    return dp;
}


template <class V>
double Lattice::norm(const V& u) const
{
    return sqrt(this->dot(u, u));
}


template <class V>
double Lattice::distance(const V& u, const V& v) const
{
    static R3::Vector duv;
    duv[0] = u[0] - v[0];
    duv[1] = u[1] - v[1];
    duv[2] = u[2] - v[2];
    return this->norm(duv);
}


template <class V>
double Lattice::angledeg(const V& u, const V& v) const
{
    using diffpy::mathutils::acosd;
    double ca = dot(u, v)/(norm(u) * norm(v));
    return acosd(ca);
}


template <class V>
double Lattice::anglerad(const V& u, const V& v) const
{
    double ca = dot(u, v)/(norm(u) * norm(v));
    return acos(ca);
}


template <class V>
const R3::Vector& Lattice::cartesian(const V& lv) const
{
    static R3::Vector lvcopy;
    lvcopy = double(lv[0]), double(lv[1]), double(lv[2]);
    return this->cartesian(lvcopy);
}


template <class V>
const R3::Vector& Lattice::fractional(const V& cv) const
{
    static R3::Vector cvcopy;
    cvcopy = double(cv[0]), double(cv[1]), double(cv[2]);
    return this->fractional(cvcopy);
}


template <class V>
const R3::Vector& Lattice::ucvCartesian(const V& cv) const
{
    static R3::Vector cvcopy;
    cvcopy = double(cv[0]), double(cv[1]), double(cv[2]);
    return this->ucvCartesian(cvcopy);
}


template <class V>
const R3::Vector& Lattice::ucvFractional(const V& cv) const
{
    static R3::Vector cvcopy;
    cvcopy = double(cv[0]), double(cv[1]), double(cv[2]);
    return ucvFractional(cvcopy);
}


}   // namespace srreal
}   // namespace diffpy

#endif  // LATTICE_HPP_INCLUDED
