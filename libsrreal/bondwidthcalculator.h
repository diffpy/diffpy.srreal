/***********************************************************************
* $Id$
***********************************************************************/

/* Calculators for bond widths. Each of these classes contains public data that
 * tunes the calculation parameters of a peak width. The calculate function
 * takes a SrReal::BondPair instance and returns a floating point number that
 * in some cases represents sigma, the correlated Debye-Waller factor.
 */

#ifndef BONDWIDTHCALCULATOR_H
#define BONDWIDTHCALCULATOR_H

#include "bonditerator.h"

namespace SrReal
{


class BondWidthCalculator
{

    public:
    BondWidthCalculator();
    virtual ~BondWidthCalculator();

    // Returns the bond width in angstroms
    virtual float calculate(SrReal::BondPair &bp);
};

class JeongBWCalculator : 
    public BondWidthCalculator
{

    public:
    JeongBWCalculator();
    virtual ~JeongBWCalculator();

    virtual float calculate(SrReal::BondPair &bp);

    float delta1; // The low-temperature coefficient (linear in 1/r) 
    float delta2; // The high-temperature coefficient (linear in 1/r^2)
};

} // end namespace SrReal

#endif // BONDWIDTHCALCULATOR_H
