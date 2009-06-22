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
* Implementation of shared template functions for PDF calculator.
*     bandPassFilter
*
* $Id$
*
*****************************************************************************/

#ifndef PDFUTILS_IPP_INCLUDED
#define PDFUTILS_IPP_INCLUDED

#include <cmath>

namespace diffpy {
namespace srreal {

template <class Ti>
void bandPassFilter(Ti first, Ti last, double dr, double qmin, double qmax)
{
    if (!(first < last))    return;
    size_t datalen = last - first;
    // pad data with the same number of zeros up to the next power of 2
    size_t padlen = (size_t) pow(2, ceil(log2(datalen) + 1));
    // ycpad is complex, so it needs to be twice as long
    std::valarray<double> ycpa(0.0, 2 * padlen);
    double* ycfirst = &(ycpa[0]);
    double* yci = ycfirst;
    for (Ti p = first; p != last; ++p, yci += 2)  { *yci = *p; }
    // perform the filtering
    bandPassFilterCValarray(ycpa, dr, qmin, qmax);
    // copy real components back to the input sequence
    yci = &(ycpa[0]);
    for (Ti p = first; p != last; ++p, yci += 2)  { *p = *yci; }
}

}   // namespace srreal
}   // namespace diffpy

// vim:ft=cpp:

#endif  // PDFUTILS_IPP_INCLUDED
