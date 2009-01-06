/***********************************************************************
* $Id$
***********************************************************************/

#ifndef PDFCALCULATOR_H
#define PDFCALCULATOR_H

#include "ObjCryst/General.h"
#include "ObjCryst/Crystal.h"

namespace SrReal
{

size_t getNumPoints(float _rmin, float _rmax, float _dr);

float *calculateRDF(BondIterator &bonditer, 
        float _rmin, float _rmax, float _dr);

float *calculatePDF(BondIterator &bonditer,
        float _rmin, float _rmax, float _dr);


// get the scattering power for a bond pair
float getPairScatPow(BondPair &bp, const ObjCryst::RadiationType rt);

// get the average scattering power for a unit cell of a crystal
inline float getAvgScatPow(BondIterator &biter,
        const ObjCryst::RadiationType rt);

// get the number of scatterers in the unit cell calculated from occupancy
inline float getOccupancy(BondIterator &bonditer);

} // End namespace SrReal

#endif //PDFCALCULATOR_H
