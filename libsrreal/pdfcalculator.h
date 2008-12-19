/***********************************************************************
* $Id$
***********************************************************************/

#ifndef PDFCALCULATOR_H
#define PDFCALCULATOR_H

namespace SrReal
{

size_t getNumPoints(float _rmin, float _rmax, float _dr);

float *calculateRDF(BondIterator &bonditer, 
        float _rmin, float _rmax, float _dr);

float *calculatePDF(BondIterator &bonditer,
        float _rmin, float _rmax, float _dr);

} // End namespace SrReal

#endif //PDFCALCULATOR_H
