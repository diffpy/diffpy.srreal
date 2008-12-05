/***********************************************************************
* $Id$
***********************************************************************/

#ifndef BONDITERATOR_H
#define BONDITERATOR_H

#include <vector>

#include "ObjCryst/Crystal.h"
#include "ObjCryst/Scatterer.h"
#include "PointsInSphere.h"


namespace SrReal
{

/* struct for a shifted scattering component 
 *
 * xyz are in cartesian coordinates.
 * */
struct ShiftedSC
{

    // id is for debugging
    ShiftedSC(const ObjCryst::ScatteringComponent *_sc,
        const float x, const float y, const float z, const int _id = 0) :
    sc(_sc), id(_id)
    {
        //sc->Print();
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;
        //std::cout << x << ' ' << y << ' ' << z << endl;
    }

    // Pointer to a ScatteringComponent
    const ObjCryst::ScatteringComponent *sc;

    /// Fractionnal coordinates
    float xyz[3];
    int id;


    /* Operators */

    bool operator<(const ShiftedSC &rhs) const
    {

        //std::cout << id << " vs " << rhs.id << endl;

        return ((xyz[0] < rhs.xyz[0]) 
            || (xyz[1] < rhs.xyz[1]) 
            || (xyz[2] < rhs.xyz[2])
            || (*sc != *(rhs.sc)));
    }

    // Compares equality.
    bool operator==(const ShiftedSC &rhs) const
    {

        //std::cout << id << " vs " << rhs.id << endl;

        return ((xyz[0] == rhs.xyz[0]) 
            && (xyz[1] == rhs.xyz[1]) 
            && (xyz[2] == rhs.xyz[2])
            && (*sc == *(rhs.sc)));
    }
   
};

std::ostream& operator<<(ostream &os, const ShiftedSC &sc)
{
    os << sc.id << ": ";
    os << sc.xyz[0] << " ";
    os << sc.xyz[1] << " ";
    os << sc.xyz[2];
    return os;
}

/* struct for holding bond pair information for use with the BondIterator
 *
 * xyz are in cartesian coordinates.
 */
class BondPair
{
    public:
    // Cartesian coordinates of the scatterers
    float xyz1[3];
    float xyz2[3];
    const ObjCryst::ScatteringComponent* sc1;
    const ObjCryst::ScatteringComponent* sc2;
    size_t multiplicity;
};


class BondIterator
{
    public:

    BondIterator
        (ObjCryst::Crystal &_crystal, 
         const float _rmin, const float _rmax);

    BondIterator(const BondIterator &);

    ~BondIterator();

    // Rewind the iterator
    void rewind();

    // Advance the iterator
    void next();
 
    // Check if the iterator is finished
    bool finished(); 

    // Update and reset the iterator given a status change in the crystal
    // structure or the calculation criteria.
    void reset(); 

    // Get the current pair.
    BondPair getBondPair();

    // Get the crystal and bounds on the iterator
    inline float getRmin() { return rmin; }
    inline float getRmax() { return rmax; }
    inline const ObjCryst::Crystal &getCrystal() { return crystal; }

    //FIXME:TESTING private:

    // Initialize punit and sunit
    void init();
    // Bonds from punit to punit
    bool incrementpp();
    // Bonds from punit to sunit
    bool incrementps();
    // Bonds from sunit to sunit
    bool incrementss();
    // Bonds from punit to image of punit
    bool incrementppi();
    // Bonds from punit to image of sunit
    bool incrementpsi(); 
    // Bonds from sunit to image of sunit
    bool incrementssi(); 
    
    // Check if the sphere is at 0, 0, 0
    inline bool sphAtOrigin() 
    {
        return (sph->mno[0]==0 && sph->mno[1]==0 && sph->mno[2]==0);
    }

    // Place cartesian coords in location defined by PointsInSphere iterator
    void placeInSphere(float *xyz);

    /**** Data members ****/

    // Reference to crystal
    const ObjCryst::Crystal &crystal;

    // Minimum and maximum r values
    const float rmin;
    const float rmax;

    // For holding the current BondPair
    BondPair bp;

    // Holds ScatteringComponents in the primitive unit
    std::vector<ShiftedSC> punit;
    // Holds ScatteringComponents that are created from symmetry operations.
    // This specifically excludes atoms in the punit.
    std::vector<ShiftedSC> sunit;

    // Degeneracy of each primitive atom in the conventional cell
    std::map<ShiftedSC,size_t> degen;

    // Iterators for punit and sunit;
    std::vector<ShiftedSC>::iterator iteri;
    std::vector<ShiftedSC>::iterator iterj;

    // Points in sphere iterator
    NS_POINTSINSPHERE::PointsInSphere *sph;

    // Enumerate the state of the iterator
    enum IncState {
        PP,
        PS,
        SS,
        PPI,
        PSI,
        SSI,
        FINISHED
    };

    // The current state of the iterator
    IncState state;
    
};

} // end namespace SrReal

#endif
