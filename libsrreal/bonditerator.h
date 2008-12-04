/***********************************************************************
* $Id$
***********************************************************************/

#ifndef BONDITERATOR_H
#define BONDITERATOR_H

namespace ObjCryst
{
    class Crystal;
    class ScatteringComponent;
    class RefinableObjClock;
} // end namespace ObjCryst


namespace SrReal
{


/* struct for a shifted scattering component */
struct ShiftedSC
{
   ShiftedSC(const ObjCryst::ScatteringComponent *ssc,
           double _xyz[3]):
   sc(ssc),xyz(_xyz);
   {}
   ShiftedSC(const ObjCryst::ScatteringComponent *ssc,
           double x, double y, double z) :
   sc(ssc);
   {
       xyz[0] = x;
       xyz[1] = y;
       xyz[2] = z;
   }

   /* Data members */

   // Pointer to a ScatteringComponent
   const ObjCryst::ScatteringComponent *sc;

   /// Fractionnal coordinates
   double xyz[3];

   /* Operators */

   bool operator<(const ShiftedSC &rhs) const
   {
    return ((xyz[0] < rhs.xyz[0]) 
        || (xyz[1] < rhs.xyz[1]) 
        || (xyz[2] < rhs.xyz[2])
        || (sc != rhs.sc));
   }

   // Compares identity. This is needed for when atoms with different
   // ScatteringComponents land on the same site, which is possible in a doped
   // material.
   bool operator==(const ShiftedSC &rhs) const
   {
    return ((xyz[0] == rhs.xyz[0]) 
        && (xyz[1] == rhs.xyz[1]) 
        && (xyz[2] == rhs.xyz[2])
        && sc == rhs.sc);
   }
   
};

/* struct for holding bond pair information for use with the BondIterator */
struct BondPair
{
    // Cartesian coordinates of the scatterers
    double xyz1[3];
    double xyz2[3];
    ObjCryst::ScatteringComponent* sc1;
    ObjCryst::ScatteringComponent* sc2;
    size_t multiplicity;
}


class BondIterator
{
    public:

    BondIterator
        (ObjCryst::Crystal &_crystal, float _rmin, float _rmax);

    BondIterator(const BondIterator &);

    // Rewind the iterator
    void rewind();

    // Advance the iterator
    void next();
 
    // Check if the iterator is finished
    bool finished(); 

    // Update and reset the iterator given a status change in the crystal
    // structure or the calculation criteria.
    void update(); 

    // Get the current pair.
    BondPair getBondPair();
    

    // Get the crystal and bounds on the iterator
    inline float getRmin() { return rmin; }
    inline float getRmax() { return rmax; }
    inline ObjCryst::Crystal &getCrystal() { return crystal; }

    private:

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

    // Reference to crystal
    ObjCryst::Crystal &crystal;

    // Minimum and maximum r values
    float rmin;
    float rmax;

    // For holding the ShiftedSC of the current pair.
    ShiftedSC sc1;
    ShiftedSC sc2;
    size_t multiplicity;

    // Holds ScatteringComponents in the primitive unit
    std::vector<ShiftedSC> punit;
    // Holds ScatteringComponents that are created from symmetry operations.
    // This specifically excludes atoms in the punit.
    std::vector<ShiftedSC> sunit;

    // Degeneracy of each primitive atom in the conventional cell
    std::vector<size_t> degen;

    // Iterators for punit and sunit;
    std::vector<ShiftedSC>::iterator iteri;
    std::vector<ShiftedSC>::iterator iterj;

    // Points in sphere iterator
    NS_POINTSINSPHERE::PointsInSphere sph;

    // These enumerate the state of the iterator
    enum IncState {
        PP,
        PS,
        SS,
        PPI,
        PSI,
        SSI,
        FINISHED
    }

    // This record the current state of the iterator
    IncState state;
};

} // end namespace SrReal

#endif
