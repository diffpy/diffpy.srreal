/***********************************************************************
* $Id$
***********************************************************************/

#ifndef BONDITERATOR_H
#define BONDITERATOR_H

#include <vector>
#include <map>

#include "ObjCryst/Crystal.h"
#include "ObjCryst/Scatterer.h"
#include "ObjCryst/General.h"
#include "PointsInSphere.h"


namespace SrReal
{

class ShiftedSC;
class BondPair;
class BondIterator;

// Very useful utility function
std::vector<ShiftedSC> getUnitCell(const ObjCryst::Crystal&);

/* Container class for holding a "shifted" ScatteringComponent. It holds a
 * reference to the ScatteringComponent and its shifted (Cartesian) position.
 * This is used when expanding the scatterers in a crystal into the conventional
 * unit cell.
*/
class ShiftedSC
{

    public:
    ShiftedSC(const ObjCryst::ScatteringComponent* _sc,
        const float x, const float y, const float z, const int _id = 0);

    ShiftedSC(const ShiftedSC& _ssc);
    ShiftedSC();

    /* Data members */

    // Pointer to a ScatteringComponent
    const ObjCryst::ScatteringComponent* sc;

    // Orthonormal coordinates
    float xyz[3];

    // Id for testing purposes
    int id;

    /* Operators */

    bool operator<(const ShiftedSC& rhs) const;

    // Compares equality.
    bool operator==(const ShiftedSC& rhs) const;

    /* Friends */
    friend class BondIterator;
    friend std::ostream& operator<<(ostream& os, const ShiftedSC& ssc);

};

std::ostream& operator<<(ostream& os, const SrReal::ShiftedSC& ssc);

/* Container class for holding bond pair information
 *
 * xyz are in cartesian coordinates.
 */
class BondPair
{
    public:

    BondPair() 
    {
        for(size_t l = 0; l < 3; ++l)
        {
            xyz1[l] = xyz2[l] = 0;
        }
        sc1 = sc2 = NULL;
        multiplicity = 0;
        r = -1;
    };

    inline void setXYZ1(float* _xyz)
    {
        for(size_t l = 0; l < 3; ++l) xyz1[l] = _xyz[l];
        r = -1;
    }

    inline void setXYZ1(size_t i, float val)
    {
        xyz1[i] = val;
        r = -1;
    }

    inline float* getXYZ1() { return xyz1; }

    inline float getXYZ1(size_t i) { return xyz1[i]; }

    inline void setXYZ2(float* _xyz)
    {
        for(size_t l = 0; l < 3; ++l) xyz2[l] = _xyz[l];
        r = -1;
    }

    inline void setXYZ2(size_t i, float val)
    {
        xyz2[i] = val;
        r = -1;
    }

    inline float* getXYZ2() { return xyz2; }

    inline float getXYZ2(size_t i) { return xyz2[i]; }

    inline void setSC1(ObjCryst::ScatteringComponent* _sc1) { sc1 = _sc1; }

    inline const ObjCryst::ScatteringComponent* getSC1() { return sc1; }

    inline void setSC2(ObjCryst::ScatteringComponent* _sc2) { sc2 = _sc2; }

    inline const ObjCryst::ScatteringComponent* getSC2() { return sc2; }

    inline void setMultiplicity(size_t _m) { multiplicity = _m; }

    inline size_t getMultiplicity() { return multiplicity; }

    inline float getDistance()
    {
        if(r == -1) 
        {
            r = 0;
            for(size_t l = 0; l < 3; ++l) 
            {
                r += pow(xyz1[l]-xyz2[l], 2);
            }
            r = sqrt(r);
        }
        return r;
    }

    private:
    // Cartesian coordinates of the scatterers
    float xyz1[3];
    float xyz2[3];
    const ObjCryst::ScatteringComponent* sc1;
    const ObjCryst::ScatteringComponent* sc2;
    size_t multiplicity;
    float r;

    /* Friends */
    friend class BondIterator;
    friend std::ostream& operator<<(ostream& os, const BondPair& bp);

};

std::ostream& operator<<(ostream& os, const SrReal::BondPair& bp);

class BondIterator
{
    public:

    BondIterator(ObjCryst::Crystal& _crystal);
    BondIterator(ObjCryst::Crystal& _crystal, float _rmin, float _rmax);
    BondIterator(const BondIterator&);

    ~BondIterator();

    // Set the range of the iterator
    void setBondRange(float _rmin, float _rmax);

    // Set one scatterer in the bond
    void setScatteringComponent(const ObjCryst::ScatteringComponent& _sc);

    // Rewind the iterator. This detects changes in the crystal and updates
    // itself as necessary.
    void rewind();

    // Advance the iterator
    void next();
 
    // Check if the iterator is finished
    bool finished(); 

    // Get the current pair.
    BondPair getBondPair();

    // Get the crystal and bounds on the iterator
    inline float getRmin() { return rmin; }
    inline float getRmax() { return rmax; }
    inline const ObjCryst::Crystal& getCrystal() { return crystal; }

    // Get the unit cell calculated internally
    inline std::vector<ShiftedSC> getUnitCell() { update(); return sscvec; }

    //FIXME:TESTING private:

    // Update the iterator
    void update();

    // Increment the iterator
    bool increment();
    
    // Check if the PointsInSphere is at 0, 0, 0
    inline bool sphAtOrigin() 
    {
        return (sph->mno[0]==0 && sph->mno[1]==0 && sph->mno[2]==0);
    }

    // Place cartesian coords in location defined by PointsInSphere iterator
    void placeInSphere(float* xyz);

    // Calculate the degeneracy of the scattering component in the unit cell
    void calculateDegeneracy();
    /**** Data members ****/

    // Reference to crystal
    const ObjCryst::Crystal& crystal;

    // Reference to one scattering component in the bond
    const ObjCryst::ScatteringComponent* sc;

    // Minimum and maximum r values
    float rmin;
    float rmax;

    // For holding the current BondPair
    BondPair bp;

    // flag indicating when we're finished.
    bool isfinished;

    // Clock for comparing with scattering components
    ObjCryst::RefinableObjClock itclock;
    // Clock for comparing with lattice clock
    ObjCryst::RefinableObjClock latclock;


    // Holds ScatteringComponents in the primitive unit
    std::vector<ShiftedSC> sscvec;
    // Iterators for punit and sunit;
    std::vector<ShiftedSC>::iterator iteri;
    // degeneracy
    size_t degen;

    // Points in sphere iterator
    NS_POINTSINSPHERE::PointsInSphere* sph;

};

} // end namespace SrReal


#endif
