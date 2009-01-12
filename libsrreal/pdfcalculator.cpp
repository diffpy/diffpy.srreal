/***********************************************************************
* $Id$
***********************************************************************/

#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_errno.h>
#include <cassert>

#include "profilecalculator.h"
#include "pdfcalculator.h"

#define NDEBUG 1
/****************************************************************************/

namespace {

    const float sqrt2pi = sqrt(2*M_PI);
    const float eps = 1e-8;
}

SrReal::PDFCalculator::
PDFCalculator(
    SrReal::BondIterator& _bonditer, SrReal::BondWidthCalculator& _bwcalc)
        : SrReal::ProfileCalculator(_bonditer, _bwcalc)
{
    rdf = pdf = NULL;
    ffta = NULL;
    cnumpoints = 0;
    crmin = crmax = cdr = 0;
    bavg = 0;
    numscat = 0;
    qmaxidxlo = qmaxidxhi = qminidxhi = 0;


    qbroad = qdamp = 0.0;
    scale = 1.0;

    /* Create the RefinablePar objects for qbraod, qdamp and scale */
    ResetParList();

    // qbroad
    {
    ObjCryst::RefinablePar* tmp = new ObjCryst::RefinablePar("qbroad", &qbroad, 0.0, 1.0, 
        &SrReal::profilerefpartype, ObjCryst::REFPAR_DERIV_STEP_ABSOLUTE, 
        false, false, true, false, 1.0, 1);
    tmp->AssignClock(mClockMaster);
    AddPar(tmp);
    // Share this with the bwcalculator
    bwcalc.AddPar(tmp);
    }

    // qdamp
    {
    ObjCryst::RefinablePar* tmp = new ObjCryst::RefinablePar("qdamp", &qdamp, 0.0, 1.0, 
        &SrReal::profilerefpartype, ObjCryst::REFPAR_DERIV_STEP_ABSOLUTE, 
        false, false, true, false, 1.0, 1);
    tmp->AssignClock(mClockMaster);
    AddPar(tmp);
    }

    // scale
    {
    ObjCryst::RefinablePar* tmp = new ObjCryst::RefinablePar("scale", &scale, 0.0, 1.0, 
        &SrReal::profilerefpartype, ObjCryst::REFPAR_DERIV_STEP_ABSOLUTE, 
        false, false, true, false, 1.0, 1);
    tmp->AssignClock(mClockMaster);
    AddPar(tmp);
    }

}

SrReal::PDFCalculator::
~PDFCalculator()
{
    if(rdf != NULL)
    {
        delete [] rdf;
    }

    if(pdf != NULL)
    {
        delete [] pdf;
    }

    if(ffta != NULL)
    {
        delete [] ffta;
    }
}

/* Get the calculated RDF over the requested calculation range.
 *
 * This returns a new array (from getCorrectedProfile). It is up to the client
 * to manage it.
 */ 
float*
SrReal::PDFCalculator::
getRDF()
{
    calculateRDF();
    float* profile = getCorrectedProfile(rdf);
    return profile;
}

/* Get the calculated PDF over the requested calculation range.
 *
 * This returns a new array (from getCorrectedProfile). It is up to the client
 * to manage it.
 */ 
float*
SrReal::PDFCalculator::
getPDF()
{

    calculatePDF();
    float* profile = getCorrectedProfile(pdf);
    return profile;
}

/* Get the corrected profile. 
 *
 * This returns a new array.
 *
 * This performs the last steps of the profile calculation.
 * - Apply termination ripples
 * - Interpolate the calculated PDF over the requested range.
 * - Apply global resolution corrections
 * - Apply scale factor
 */
float*
SrReal::PDFCalculator::
getCorrectedProfile(float* df)
{
    /* apply termination ripples */
    // blank the fft array
    for (size_t i=0; i<2*fftlen; ++i) ffta[i] = 0.0;
    // copy df to real components of ffta
    for (size_t i=0; i<cnumpoints; ++i) ffta[2*i] = df[i];
    applyTerminationRipples();

    /* interpolate from ffta to the profile over requested calculation range 
     * and apply the scale and resolution correction while we're at it
    */
    float* profile = new float [numpoints];

    size_t cridx = 0;
    float cr = crmin;
    float y1 = 0, y2 = 0;
    for(size_t l = 0; l < numpoints; ++l)
    {
        // Find the next calculated r-value greater than rvals[l];
        while(cr <= rvals[l])
        {
            ++cridx;
            cr = crmin + cdr*cridx;
        }
        //std::cout << "cridx = " << cridx; 
        //std::cout << ", rvals[l] = " << rvals[l];
        //std::cout <<  ", cr = " << cr << std::endl;
        assert(cridx < cnumpoints);

        // Interpolate (linear) the new data point base on profile values calculated
        // below and above the requested r value.
        // FIXME - This should use a cubic spline
        y1 = ffta[2*(cridx-1)];
        y2 = ffta[2*cridx];
        profile[l] = y2 + (y1-y2) * (cr-rvals[l])/cdr;

        // Apply the resolution correction and scale factor
        profile[l] *= scale;
        if(qdamp != 0)
        {
            profile[l] *= exp(-0.5*pow(qdamp*rvals[l],2));
        }
    }

    return profile;
}

void
SrReal::PDFCalculator::
calculateRDF()
{

    const ObjCryst::Crystal &crystal = bonditer.getCrystal();

    // Calculate the bonds within the calculation range
    float r, d, sigma, gnorm;
    float grmin, grmax;
    size_t gimin, gimax;

    for(size_t i = 0; i < numpoints; ++i) rdf[i] = 0.0;

    calcAvgScatPow();

    SrReal::BondPair bp;

    const ObjCryst::ScatteringComponentList &scl 
        = crystal.GetScatteringComponentList();

    for(int i=0; i < scl.GetNbComponent(); ++i)
    {
        bonditer.setScatteringComponent(scl(i));

        for(bonditer.rewind(); !bonditer.finished(); bonditer.next())
        {
            bp = bonditer.getBondPair();

            d = bp.getDistance();
            if(d == 0) continue;

            // Get the DW factor
            sigma = bwcalc.calculate(bp);

            //std::cout << "r = " << d << std::endl;
            //std::cout << "sigma = " << sigma << std::endl;
            //std::cout << "rmin = " << crmin << std::endl;
            //std::cout << "rmax = " << crmax << std::endl;

            // Only continue if we're within five DW factors of the cutoff
            if( d > crmin-5*sigma and d < crmax+5*sigma ) {

                // calculate the gaussian 
                gnorm = 1.0/(sqrt2pi*sigma);
                gnorm *= bp.getMultiplicity();
                gnorm *= getPairScatPow(bp);
                gnorm /= bavg*bavg;
                gnorm /= numscat;

                // calculate out to 5*sigma
                grmin = d - 5*sigma;
                grmax = d + 5*sigma;
                if(grmin < crmin) grmin = crmin;
                if(grmax > crmax) grmax = crmax;
                //std::cout << "grminmax " << grmin << ' ' << grmax << std::endl;
                gimin = static_cast<size_t>( (grmin - crmin)/cdr );
                gimax = static_cast<size_t>( (grmax - crmin)/cdr );
                for(size_t l=gimin; l<gimax; ++l) {
                    r = crmin + l*cdr;
                    rdf[l] += gnorm * exp(-0.5*pow((r-d)/sigma,2));
                }
            }
        }
    }

    return;
}

void
SrReal::PDFCalculator::
calculatePDF()
{

    calculateRDF();
    // calculate rdf/r - 4*pi*r*rho0;
    const ObjCryst::Crystal &crystal = bonditer.getCrystal();
    float rho0 = numscat / crystal.GetVolume();
    float r;
    // Avoid dividing by 0
    size_t l = 0;
    if( crmin == 0 ) l = 1;
    for(; l < cnumpoints; ++l)
    {
        r = crmin + l*cdr;
        pdf[l] = rdf[l]/r - 4 * M_PI * rho0 * r;
    }

    return;
}

void 
SrReal::PDFCalculator::
setCalculationPoints(const float* _rvals, const size_t _numpoints)
{
    numpoints = _numpoints;
    float rmin = _rvals[0];
    float rmax = _rvals[numpoints-1];
    //std::cout << "rmax = " << rmax << std::endl;
    //std::cout << "rmin = " << rmin << std::endl;
    //std::cout << "numpoints = " << numpoints << std::endl;
    // Extend the range to include other bonds that may overlap
    const ObjCryst::Crystal& crystal = bonditer.getCrystal();
    float diam;
    diam = pow((double) crystal.GetVolume(), (double) 1.0/3.0);
    bonditer.setBondRange(rmin-diam, rmax+diam);
    // Prepare rvals
    if( rvals != NULL )
    {
        delete [] rvals;
    }
    rvals = new float [numpoints];

    // The FFT that is used in the termination ripple calculation requires an
    // equispaced grid. Calculate this grid on the smallest stride in _rvals.
    float dr = 0;
    cdr = rmax;
    rvals[0] = rmin;
    for(size_t l=1; l<numpoints; ++l)
    {
        rvals[l] = _rvals[l];
        dr = _rvals[l] - _rvals[l-1];
        cdr = (dr < cdr ? dr : cdr);
    }
    if(cdr < eps)
    {
        cdr = 0.01;
    }
    //std::cout << "cdr = " << cdr << std::endl;

    // Figure out the min and max of the calculation points. We do this by
    // extending the calculation range by at least 6 termination ripples. We'll
    // assume qmax >= 15 and use this to calculate the extention range.
    const int nripples = 6;
    const float _qmax = 15.0;
    float rext = nripples*2*M_PI/_qmax;
    crmin = max( (float) 0.0, rmin-rext);
    crmax = cdr*int(ceil(rmax + rext)/cdr);
    cnumpoints = static_cast<size_t>((crmax-crmin+eps)/cdr);

    // Create the arrays for handling the PDF and RDF
    rdf = new float [cnumpoints];
    pdf = new float [cnumpoints];

    //std::cout << "crmin = " << crmin << std::endl;
    //std::cout << "crmax = " << crmax << std::endl;
    //std::cout << "cnumpoints = " << cnumpoints << std::endl;

    setupFFT();
    return;
}

void
SrReal::PDFCalculator::
setQmax(float val)
{
    SrReal::ProfileCalculator::setQmax(val);
    setupFFT();
}

void
SrReal::PDFCalculator::
setQmin(float val)
{
    SrReal::ProfileCalculator::setQmin(val);
    setupFFT();
}

// This must be called whenever setupCalculation has been called.
void 
SrReal::PDFCalculator::
setupFFT() {

    if( cnumpoints > 0 )
    {
        // Want this to be a power of 2. 
        // This also guarantees that fftlen >= cnumpoints.
        fftlen = static_cast<int>(pow(2.0, ceil(log2(2*cnumpoints))));
        //std::cout << "fftlen = " << fftlen << std::endl;

        if( ffta != NULL ) {
            delete [] ffta;
        }

        // ffta is complex, so it needs to be twice as long
        ffta = new double [2*fftlen];

        // Q-spacing of the fft array
        float dQ = 2*M_PI/((fftlen-1)*cdr);
        // these indices correspond to Qmax and -Qmax frequencies.
        // They need to be integer to catch cases with huge qmax/dQ
        // ditto for qmin, except qminidxlo = 0.
        qmaxidxlo = min(fftlen, size_t( ceil(qmax/dQ) ));
        qmaxidxhi = fftlen + 1 - qmaxidxlo;
        qminidxhi = min(fftlen, size_t( ceil(qmin/dQ) ));
        //std::cout << "qmaxidxlo = " << qmaxidxlo << std::endl;
        //std::cout << "qmaxidxhi = " << qmaxidxhi << std::endl;
        //std::cout << "qminidxhi = " << qminidxhi << std::endl;
            
    }
}

/* Calculate the termination ripples of the passed profile.
 *
 * The profile to be modified must be placed in the even slots of fft before
 * calling this function.
 */
void 
SrReal::PDFCalculator::
applyTerminationRipples()
{
    if(qmax > 0 or qmin > 0)
    {
        // apply fft
        int fftstatus = gsl_fft_complex_radix2_forward(ffta, 1, fftlen);

        if (fftstatus != GSL_SUCCESS)
        {
            // FIXME should be exception
            std::cerr << "Forward FFT failed!";
            exit(1);
        }

        if( qmax > 0 )
        {

            // zero high Q components in ffta
            for (size_t i = qmaxidxlo; i < qmaxidxhi; ++i)
            {
                ffta[2*i] = ffta[2*i+1] = 0.0;
            }
        }

        if( qmin > 0)
        {
            // zero low Q components in ffta
            for (size_t i = 0; i < qminidxhi; ++i)
            {
                ffta[2*i] = ffta[2*i+1] = 0.0;
            }
        }

        // transform back
        fftstatus = gsl_fft_complex_radix2_inverse(ffta, 1, fftlen);

        if (fftstatus != GSL_SUCCESS)
        {
            // FIXME should be exception
            std::cerr << "Inverse FFT failed!";
            exit(1);
        }
    }

    return;
}

void 
SrReal::PDFCalculator::
calcAvgScatPow() {

    std::vector<ShiftedSC> unitcell = bonditer.getUnitCell();
    std::vector<ShiftedSC>::iterator it1;

    bavg = 0.0;
    numscat = 0.0;

    for(it1 = unitcell.begin(); it1 != unitcell.end(); ++it1)
    {
        numscat += it1->sc->mOccupancy;
        bavg += it1->sc->mpScattPow->GetForwardScatteringFactor(radtype) *
                it1->sc->mOccupancy;
    }
    bavg /= numscat;
    return;
}

float 
SrReal::PDFCalculator::
getPairScatPow(SrReal::BondPair &bp)
{
    float scatpow = 1;
    scatpow *= bp.getSC1()->mpScattPow->GetForwardScatteringFactor(radtype);
    scatpow *= bp.getSC1()->mOccupancy;
    scatpow *= bp.getSC2()->mpScattPow->GetForwardScatteringFactor(radtype);
    scatpow *= bp.getSC2()->mOccupancy;

    return scatpow;
}
