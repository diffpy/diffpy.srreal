/***********************************************************************
* $Id$
***********************************************************************/

#include <cmath>

#include "bonditerator.h"
#include "pdfcalculator.h"
#include "bondwidthcalculator.h"

namespace {

    const float sqrt2pi = sqrt(2.0*M_PI);

}

using namespace SrReal;

size_t 
SrReal::
getNumPoints(float _rmin, float _rmax, float _dr)
{
    float rmin = _rmin;
    rmin = ( rmin < 0 ? 0 : rmin );
    float rmax = fabs(_rmax);
    if(rmax < rmin) swap(rmin, rmax);
    float dr = _dr;
    if( dr <= 0 ) dr = 0.01;
    size_t numpoints = static_cast<size_t>( (rmax-rmin)/dr );
    return numpoints;

}

/* This calculates the RDF */
float *
SrReal::
calculateRDF(BondIterator &bonditer, 
        float _rmin, float _rmax, float _dr)
{
    float rmin = _rmin;
    rmin = ( rmin < 0 ? 0 : rmin );
    float rmax = _rmax;
    if(rmax < rmin) swap(rmin, rmax);
    float dr = _dr;
    if( dr <= 0 ) dr = 0.01;
    size_t numpoints = static_cast<size_t>( (rmax-rmin)/dr );
    rmax = rmin + numpoints*dr;

    std::cout << "numpoints = " << numpoints << std::endl;

    BondPair bp;
    const ObjCryst::Crystal &crystal = bonditer.getCrystal();

    JeongBWCalculator *bwcalc = new JeongBWCalculator;
    bwcalc->delta2 = 5.0;

    // Calculate the bonds within the calculation range
    float d, sigma, r, gnorm;
    float grmin, grmax;
    size_t gimin, gimax;

    float *profile = new float[numpoints];
    for(size_t i = 0; i < numpoints; ++i) profile[i] = 0.0;

    size_t nsc = bonditer.getNumScat();

    const ObjCryst::ScatteringComponentList &scl 
        = crystal.GetScatteringComponentList();

    for(int i=0; i < scl.GetNbComponent(); ++i)
    {
        std::cout << "i = " << i << std::endl;
        bonditer.setScatteringComponent(scl(i));

        for(bonditer.rewind(); !bonditer.finished(); bonditer.next())
        {
            bp = bonditer.getBondPair();

            d = bp.getDistance();
            if(d == 0) continue;

            sigma = bwcalc->calculate(bp);

            //std::cout << "r = " << d << std::endl;
            //std::cout << "sigma = " << sigma << std::endl;
            //std::cout << "rmin = " << rmin << std::endl;
            //std::cout << "rmax = " << rmax << std::endl;

            // Only continue if we're within five DW factors of the cutoff
            if( d > rmin-5*sigma and d < rmax+5*sigma ) {

                // calculate the gaussian 
                gnorm = 1.0/(sqrt2pi*sigma);
                gnorm *= bp.getMultiplicity();
                gnorm /= nsc;

                // calculate out to 5*sigma
                grmin = d - 5*sigma;
                grmax = d + 5*sigma;
                if(grmin < rmin) grmin = rmin;
                if(grmax > rmax) grmax = rmax;
                //std::cout << "grminmax " << grmin << ' ' << grmax << std::endl;
                gimin = static_cast<size_t>( (grmin - rmin)/dr );
                gimax = static_cast<size_t>( (grmax - rmin)/dr );
                //std::cout << "iminmax " << gimin << ' ' << gimax << std::endl;
                for(size_t l=gimin; l<gimax; ++l) {
                    r = rmin + l*dr;
                    profile[l] += gnorm * exp(-0.5*pow((r-d)/sigma,2));
                }
            }
        }
    }

    delete bwcalc;

    return profile;
}

/* This calculates the RDF */
float *
SrReal::
calculatePDF(BondIterator &bonditer,
        float _rmin, float _rmax, float _dr)

{

    float *rdf = calculateRDF(bonditer, _rmin, _rmax, _dr);

    float rmin = _rmin;
    rmin = ( rmin < 0 ? 0 : rmin );
    float rmax = _rmax;
    if(rmax < rmin) swap(rmin, rmax);
    float dr = _dr;
    if( dr <= 0 ) dr = 0.01;
    size_t numpoints = static_cast<size_t>( (rmax-rmin)/dr );
    // calculate rdf/r - 4*pi*r*rho0;
    const ObjCryst::Crystal &crystal = bonditer.getCrystal();
    // Calculate density
    float rho0 = bonditer.getNumScat() / crystal.GetVolume();
    float r = 0;
    size_t l = 0;
    // Don't want to divide by 0
    if( rmin == 0 ) l = 1;
    for(; l < numpoints; ++l)
    {
        r = rmin + dr*l;
        rdf[l] = rdf[l]/r - 4 * M_PI * rho0 * r;
    }

    return rdf;

}
