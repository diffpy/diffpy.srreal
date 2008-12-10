/***********************************************************************
* $Id$
***********************************************************************/

#ifndef BONDITERATOR_H
#define BONDITERATOR_H

#include <vector>
#include <map>

#include "ObjCryst/Crystal.h"
#include "ObjCryst/Scatterer.h"
#include "PointsInSphere.h"


namespace SrReal
{

class ShiftedSC;
class BondPair;
class BondIterator;

// Very useful utility function
std::vector<ShiftedSC> getUnitCell(const ObjCryst::Crystal &);

class ShiftedSC
{


    public:
    ShiftedSC(const ObjCryst::ScatteringComponent *_sc,
        const float x, const float y, const float z, const int _id = 0);

    ShiftedSC();

    private:
    /* Data members */

    // Pointer to a ScatteringComponent
    const ObjCryst::ScatteringComponent *sc;

    // Fractional coordinates
    float xyz[3];

    // Id for testing purposes
    int id;

    public:

    ShiftedSC(const ShiftedSC &_ssc);

    /* Operators */

    bool operator<(const ShiftedSC &rhs) const;

    // Compares equality.
    bool operator==(const ShiftedSC &rhs) const;

    /* Friends */
    friend class BondIterator;
    friend std::ostream& operator<<(ostream &os, const ShiftedSC &sc);

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

    BondPair() 
    {
        for(size_t l = 0; l < 3; ++l)
        {
            xyz1[l] = xyz2[l] = 0;
        }
        sc1 = sc2 = NULL;
        multiplicity = 0;
    };

    inline void setXYZ1(float* _xyz)
    {
        for(size_t l = 0; l < 3; ++l) xyz1[l] = _xyz[l];
    }

    inline float* getXYZ1() { return xyz1; }

    inline float getXYZ1(size_t i) { return xyz1[i]; }

    inline void setXYZ2(float* _xyz)
    {
        for(size_t l = 0; l < 3; ++l) xyz2[l] = _xyz[l];
    }
    inline float* getXYZ2() { return xyz2; }

    inline float getXYZ2(size_t i) { return xyz2[i]; }

    inline void setSC1(ObjCryst::ScatteringComponent *_sc1) { sc1 = _sc1; }

    inline const ObjCryst::ScatteringComponent* getSC1() { return sc1; }

    inline void setSC2(ObjCryst::ScatteringComponent *_sc2) { sc2 = _sc2; }

    inline const ObjCryst::ScatteringComponent* getSC2() { return sc2; }

    inline void setMultiplicity(size_t _m) { multiplicity = _m; }

    inline size_t getMultiplicity() { return multiplicity; }

    private:
    // Cartesian coordinates of the scatterers
    float xyz1[3];
    float xyz2[3];
    const ObjCryst::ScatteringComponent* sc1;
    const ObjCryst::ScatteringComponent* sc2;
    size_t multiplicity;

    /* Friends */
    friend class BondIterator;
    friend std::ostream& operator<<(ostream &os, const BondPair &bp);

};

std::ostream& operator<<(ostream &os, const BondPair &bp)
{
    os << "(" << bp.multiplicity << ") ";
    os << "[";
    os << bp.xyz1[0] << ", ";
    os << bp.xyz1[1] << ", ";
    os << bp.xyz1[2] << "]";
    os << " -- ";
    os << "[";
    os << bp.xyz2[0] << ", ";
    os << bp.xyz2[1] << ", ";
    os << bp.xyz2[2] << "]";

    return os;
}


class BondIterator
{
    public:

    BondIterator
        (ObjCryst::Crystal &_crystal, 
         const float _rmin, const float _rmax);

    BondIterator(const BondIterator &);

    ~BondIterator();

    // Set one scatterer in the bond
    void setScatteringComponent(ObjCryst::ScatteringComponent &_sc);

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
    // Increment the iterator
    bool increment();
    
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

    // Reference to one scattering component in the bond
    ObjCryst::ScatteringComponent *sc;

    // Minimum and maximum r values
    const float rmin;
    const float rmax;

    // For holding the current BondPair
    BondPair bp;

    // flag indicating when we're finished.
    bool isfinished;

    // Holds ScatteringComponents in the primitive unit
    std::vector<ShiftedSC> sscvec;
    // Iterators for punit and sunit;
    std::vector<ShiftedSC>::iterator iteri;

    // Points in sphere iterator
    NS_POINTSINSPHERE::PointsInSphere *sph;

};

} // end namespace SrReal


#endif
