/***********************************************************************
* $Id$
***********************************************************************/

#ifndef BONDITERATOR_H
#define BONDITERATOR_H

#include <vector>

#include "ObjCryst/Crystal.h"
#include "ObjCryst/Scatterer.h"
#include "PointsInSphere.h"

namespace
{

/* The sign of a value
 * signof(-) == signof(0) == 0
 * signof(+) == 1
 */
int signof(const float v)
{
    return v > 0 ? 1 : 0;
}

size_t quadrant(const float * _xyz)
{
    // Check if _xyz is at the origin
    if( _xyz[0] == _xyz[1] &&
        _xyz[1] == _xyz[2] &&
        _xyz[2] == 0)
        return 0;
    // Return the quadrant
    size_t q = 0;
    for(size_t l = 0; l < 3; ++l)
    {
        q += signof(_xyz[l]) << l;
    }
    return q;
}

}

namespace SrReal
{

// Here's a private class
class ShiftedSC
{

    private:

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

    // Be careful of dangling references
    ShiftedSC()
    {
        xyz[0] = xyz[1] = xyz[2] = 0;
        id = -1;
        sc = NULL;
    }

    // Pointer to a ScatteringComponent
    const ObjCryst::ScatteringComponent *sc;

    /// Fractionnal coordinates
    float xyz[3];
    int id;

    public:

    /* Operators */

    bool operator<(const ShiftedSC &rhs) const
    {

        //std::cout << id << " vs " << rhs.id << endl;
        // FIXME - I need a more stable criterion
        // Do this by quadrant first
        // (0, 0, 0) < q1 < q2 < q3 ... < q8
        // Then by distance

        size_t q1, q2;
        q1 = quadrant(xyz);
        q2 = quadrant(rhs.xyz);

        if( q1 != q2 ) return (q1 < q2);

        float d1, d2;
        for(size_t l = 0; l < 3; ++l)
        {
            d1 += xyz[l]*xyz[l];
            d2 += rhs.xyz[l]*rhs.xyz[l];
        }
        return d1 < d2;
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

    /* Friends */
    friend class BondIterator;
    friend class std::vector<ShiftedSC>;
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

/* struct for a shifted scattering component 
 *
 * xyz are in cartesian coordinates.
 * */
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

    void setXYZ1(float* _xyz)
    {
        for(size_t l = 0; l < 3; ++l) xyz1[l] = _xyz[l];
    }
    float* getXYZ1() { return xyz1; }
    float getXYZ1(size_t i) { return xyz1[i]; }

    void setXYZ2(float* _xyz)
    {
        for(size_t l = 0; l < 3; ++l) xyz2[l] = _xyz[l];
    }
    float* getXYZ2() { return xyz2; }
    float getXYZ2(size_t i) { return xyz2[i]; }

    void setSC1(ObjCryst::ScatteringComponent *_sc1)
    {
        sc1 = _sc1;
    }

    const ObjCryst::ScatteringComponent* getSC1() { return sc1; }

    void setSC2(ObjCryst::ScatteringComponent *_sc2)
    {
        sc2 = _sc2;
    }

    const ObjCryst::ScatteringComponent* getSC2() { return sc2; }

    void setMultiplicity(size_t _m)
    {
        multiplicity = _m;
    }

    size_t getMultiplicity() { return multiplicity; }

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
