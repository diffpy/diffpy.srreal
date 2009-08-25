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
* Various common routines useful for PDF calculation:
*     meanSquareDisplacement
*     bandPassFilter
*
* $Id$
*
*****************************************************************************/

#include <stdexcept>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_fft_complex.h>

#include <diffpy/srreal/PDFUtils.hpp>
#include <diffpy/mathutils.hpp>

using namespace std;
using diffpy::mathutils::eps_eq;

namespace diffpy {
namespace srreal {


double meanSquareDisplacement(const R3::Matrix& Uijcartn,
        const R3::Vector& s, bool anisotropy)
{
    double rv;
    if (anisotropy)
    {
        assert(R3::norm(s) > 0);
        assert(eps_eq(Uijcartn(0,1), Uijcartn(1,0)));
        assert(eps_eq(Uijcartn(0,2), Uijcartn(2,0)));
        assert(eps_eq(Uijcartn(1,2), Uijcartn(2,1)));
        static R3::Vector sn;
        sn = s / R3::norm(s);
        rv = Uijcartn(0,0) * sn(0) * sn(0) +
             Uijcartn(1,1) * sn(1) * sn(1) +
             Uijcartn(2,2) * sn(2) * sn(2) +
             2 * Uijcartn(0,1) * sn(0) * sn(1) +
             2 * Uijcartn(0,2) * sn(0) * sn(2) +
             2 * Uijcartn(1,2) * sn(1) * sn(2);
    }
    else
    {
        assert(eps_eq(Uijcartn(0,0), Uijcartn(1,1)));
        assert(eps_eq(Uijcartn(0,0), Uijcartn(2,2)));
        rv = Uijcartn(0,0);
    }
    return rv;
}

void bandPassFilterCValarray(valarray<double>& ycpa, double dr,
        double qmin, double qmax)
{
    // error message for FT failure
    const char* emsgft = "Fourier Transformation failed.";
    double* yc = &(ycpa[0]);
    // ycpa is a complex array, its actual length is half the size
    int padlen = ycpa.size() / 2;
    // apply fft
    int status;
    status = gsl_fft_complex_radix2_forward(yc, 1, padlen);
    if (status != GSL_SUCCESS)
    {
        throw invalid_argument(emsgft);
    }
    // Q step for yc
    double dQ = 2 * M_PI / ((padlen - 1) * dr);
    // Cut away high-Q frequencies -
    // loQmaxidx, hiQmaxidx correspond to Qmax and -Qmax frequencies
    // they need to be integer to catch cases with huge qmax/dQ
    int loQmaxidx = int( ceil(qmax/dQ) );
    int hiQmaxidx = padlen + 1 - loQmaxidx;
    hiQmaxidx = min(padlen, hiQmaxidx);
    // zero high Q components in yc
    for (int i = loQmaxidx; i < hiQmaxidx; ++i)
    {
	yc[2 * i] = yc[2 * i + 1] = 0.0;
    }
    // Cut away low-Q frequencies, while keeping the absolut offset.
    // loQminidx corresponds to the Qmin frequency.
    int loQminidx = (int) min(ceil(qmin / dQ), padlen / 2.0);
    for (int i = 1; i < loQminidx; ++i)
    {
        assert(2 * i + 1 < padlen);
        yc[2 * i] = yc[2 * i + 1] = 0.0;
        assert(padlen - 2 * i >= 0);
        yc[padlen - 2 * i] = yc[padlen - 2 * i + 1] = 0.0;
    }
    // transform back
    status = gsl_fft_complex_radix2_inverse(yc, 1, padlen);
    if (status != GSL_SUCCESS)
    {
        throw invalid_argument(emsgft);
    }
}

}   // namespace srreal
}   // namespace diffpy

// End of file
