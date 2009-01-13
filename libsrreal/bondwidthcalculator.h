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

#include <string>

#include "bonditerator.h"
#include "RefinableObj/RefinableObj.h" // From ObjCryst

namespace SrReal
{


/* Base class for bond width calculators.
 *
 * This calculates the uncorrelated bond width from the Biso values of the
 * constituents. This class is derived from ObjCryst::RefinableObj so that we
 * can make use of the ObjCryst clock mechanism.
 */
class BondWidthCalculator 
    : public ObjCryst::RefinableObj
{

    public:
    BondWidthCalculator();
    virtual ~BondWidthCalculator();

    // Returns the bond width in angstroms
    virtual float calculate(SrReal::BondPair& bp);
};

/* Bond width calculator using the formula from I.-K. Jeong, et al., Phys. Rev.
 * B 67, 104301 (2003)
 */
class JeongBWCalculator 
    : public BondWidthCalculator
{

    public:
    JeongBWCalculator();
    virtual ~JeongBWCalculator();

    virtual float calculate(SrReal::BondPair& bp);

    protected:

    /* Refinable parameters */
    // These are accessible through the refinable parameter interface inherited
    // from RefinableObj. The parameter qbroad is also shared with the
    // profile calculator using this instance.
    float delta1; // The low-temperature coefficient (of 1/r) 
    float delta2; // The high-temperature coefficient (of 1/r^2)
    float qbroad; // A resolution-based broadening factor

};

// Refinable parameter type for BondWidthCalculator parameters
const ObjCryst::RefParType bwrefpartype(string("bwrefpartype"));

} // end namespace SrReal

#endif // BONDWIDTHCALCULATOR_H
