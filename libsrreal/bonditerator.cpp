/***********************************************************************
* $Id$
***********************************************************************/

#include <vector>

#include "bonditerator.h"
#include "PointsInSphere.h"

// From ObjCryst distribution
#include "CrystVector/CrystVector.h"
#include "ObjCryst/ScatteringPower.h"

#include "assert.h"

using namespace SrReal;
using std::vector;

namespace {

float rtod = 180.0/M_PI;

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
        q += (_xyz[l] > 0 ? 1 : 0 ) << l;
    }
    return q;
}

} // End anonymous namespace

/******************************************************************************
***** BondIterator implementation *********************************************
******************************************************************************/

BondIterator::
BondIterator (ObjCryst::Crystal &_crystal, 
    const float _rmin, const float _rmax)
    : crystal(_crystal), rmin(_rmin), rmax(_rmax)
    
{
    sph = NULL;
    init();
}


BondIterator::
BondIterator(const BondIterator &other) 
    : crystal(other.crystal), rmin(other.rmin), rmax(other.rmax)
{
    sph = NULL;
    init();
}

BondIterator::
~BondIterator()
{
    if( sph != NULL )
    {
        delete sph;
    }
}

void
BondIterator::
rewind()
{
    // Set the iterators up for the beginning of the next functions, which is
    // nextss.
    iteri = punit.begin();
    iterj = punit.begin() + 1;
    state = PP;

    // Initialize the first bond
    next();

}

/* This sets bp to the next bond in the iteration sequence. The sequence is as
 * follows:
 * incrementpp
 * incrementps
 * incrementss
 * incrementppi
 * incrementpsi
 * incrementssi
 *
 * The procedures and meaning of these iteration incrementers are described
 * before their implementations below.
 *
 * The duty of this function is to call the appropriate incrementer based on the
 * state of the iterator. It is the responsibility of the incrementer to set
 * bp and indicate if it was successful.
 */
void
BondIterator::
next()
{
    // Call the appropriate incrementer based on the state of the iterator. If
    // the incrementer returns false, then the state is incremented, the
    // ShiftedSC iterators are set for the next incrementor, and we try again.
    //
    //FIXME - Check the start conditions in the switch
    switch (state)
    {
        case PP :
            if( !incrementpp() ) 
            {
                state = PS;
                iteri = punit.begin();
                iterj = sunit.begin();
                next();
            }
            break;

        case PS :
            if( !incrementps() )
            { 
                state = SS;
                iteri = sunit.begin();
                iterj = sunit.begin() + 1;
                next();
            }
            break;

        case SS :
            if( !incrementss() ) 
            {
                state = PPI;
                iteri = punit.begin();
                iterj = punit.begin();
                sph->rewind();
                if( sphAtOrigin() ) sph->next();
                next();
            }
            break;

        case PPI :
            if( !incrementppi() ) 
            {
                state = PSI;
                iteri = punit.begin();
                iterj = sunit.begin();
                sph->rewind();
                if( sphAtOrigin() ) sph->next();
                next();
            }
            break;

        case PSI :
            if( !incrementpsi() ) 
            {
                state = SSI;
                iteri = sunit.begin();
                iterj = sunit.begin() + 1;
                sph->rewind();
                if( sphAtOrigin() ) sph->next();
                next();
            }
            break;

        case SSI :
            if( !incrementssi() ) state = FINISHED;
            break;

        case FINISHED :
            return;

    }

    return;
}

bool
BondIterator::
finished()
{
    return (state == FINISHED);
}

/* Resets everything and rewinds. 
 * Call this in response to a change in the crystal.
 */
void
BondIterator::
reset()
{
    if( sph != NULL )
    {
        delete sph;
    }
    init();
    return;
}

/* Get the bond pair from the iterator */
BondPair
BondIterator::
getBondPair()
{
    return bp;
}

/*****************************************************************************/

/* This expands primitive cell of the crystal and fills punit and sunit and then
 * calls rewind().
 */
void
BondIterator::
init()
{
    // Make sure these are clear
    sunit.clear();
    punit.clear();
    degen.clear();

    // Expand each scattering component in the primitive cell and record the new
    // atoms.
    const ObjCryst::ScatteringComponentList &mScattCompList 
        = crystal.GetScatteringComponentList();

    size_t nbComponent = mScattCompList.GetNbComponent();

    // Get the origin of the primitive unit
    const float x0 = crystal.GetSpaceGroup().GetAsymUnit().Xmin();
    const float y0 = crystal.GetSpaceGroup().GetAsymUnit().Ymin();
    const float z0 = crystal.GetSpaceGroup().GetAsymUnit().Zmin();

    size_t nbSymmetrics = crystal.GetSpaceGroup().GetNbSymmetrics();
    std::cout << "nbComponent = " << nbComponent << std::endl;
    std::cout << "nbSymmetrics = " << nbSymmetrics << std::endl;

    float x, y, z;
    float junk;
    CrystMatrix<float> symmetricsCoords;
    vector<ShiftedSC> workvec(nbSymmetrics);
    vector<ShiftedSC>::iterator it1;
    vector<ShiftedSC>::iterator it2;
    // For each scattering component, find its position in the primitive cell
    // and expand that position. Record this as a ShiftedSC.
    for(size_t i=0;i<nbComponent;++i)
    {

        symmetricsCoords = crystal.GetSpaceGroup().GetAllSymmetrics(
            mScattCompList(i).mX, 
            mScattCompList(i).mY, 
            mScattCompList(i).mZ
            );

        // Put each symmetric position in the unit cell
        for(size_t j=0;j<nbSymmetrics;++j)
        {
            x=modf(symmetricsCoords(j,0)-x0,&junk);
            y=modf(symmetricsCoords(j,1)-y0,&junk);
            z=modf(symmetricsCoords(j,2)-z0,&junk);
            if(x<0) x += 1.;
            if(y<0) y += 1.;
            if(z<0) z += 1.;

            // Get this in cartesian
            crystal.FractionalToOrthonormalCoords(x,y,z);
            // Store it in the working vector
            workvec[j] =  ShiftedSC(&mScattCompList(i),x,y,z,j);
        }

        // Now find the unique scatterers and record the degeneracy. It doesn't
        // matter if we record the actual primitive ssc in punit, as long as we
        // only put one there then we're fine.
        sort(workvec.begin(), workvec.end());
        //for(it2=workvec.begin();it2!=workvec.end();++it2)
        //    std::cout << *it2 << std::endl;
        //std::cout << std::endl;
        it1 = unique(workvec.begin(), workvec.end());
        //for(it2=workvec.begin();it2!=workvec.end();++it2)
        //    std::cout << *it2 << std::endl;
        //it2 = workvec.begin();
        // Put the first ssc in the punit
        if( it2 != it1 )
        {
            degen[workvec[0]] = 1;
            punit.push_back(*it2);
            //std::cout << *it2 << std::endl;
        }
        // Put the rest in the sunit and count the degeneracy.
        for(++it2; it2!=it1; ++it2)
        {
            ++degen[workvec[0]];
            sunit.push_back(*it2);
            //std::cout << *it2 << std::endl;
        }
    }

    // Set up the PointsInSphere iterator
    // FIXME - make sure that we're putting in the right rmin, rmax
    sph = new PointsInSphere((float) rmin, (float) rmax, 
        crystal.GetLatticePar(0),
        crystal.GetLatticePar(1),
        crystal.GetLatticePar(2),
        rtod*crystal.GetLatticePar(3),
        rtod*crystal.GetLatticePar(4),
        rtod*crystal.GetLatticePar(5)
       );

    // Get the iterators ready. This is placed here to guarantee a consistent
    // state whenever init is called.
    rewind();

}

/* This gets the current bond from the primitive unit cell into the primitive
 * cell and then increments. This is an exclusive iterator as it loops over
 * unique ssc pairs. The iterator loops over unique pairs once and records the
 * multiplicity for each pair as 2.
 *
 * To guarantee the consistency of the iterator, this should only be called by
 * the next() function.
 *
 * Returns true when the incrementer progressed, false otherwise.
 */
bool
BondIterator::
incrementpp()
{

    if(punit.size() == 0) return false;
    /* iteri = slow iterator (outer loop)
     * iterj = fast iterator (inner loop)
     */

    // Check the termination condition. We handle here the case where there is
    // only one atom in the primitive unit.
    if(iteri == punit.end()-1 && iterj == punit.end())
    {
        return false;
    }

    // If we got here, then we can record the bond
    for(size_t l=0;l<3;++l)
    {
        bp.xyz1[l] = iteri->xyz[l];
        bp.xyz2[l] = iterj->xyz[l];
    }
    bp.sc1 = iteri->sc;
    bp.sc2 = iterj->sc;
    bp.multiplicity = 2;

    // Now we can increment the fast iterator.
    ++iterj;

    // If the fast iterator is at the end then increment the slow iterator and
    // set the fast iterator one step beyond it.
    if(iterj == punit.end())
    {
        ++iteri;
        iterj = iteri;
        ++iterj;
    }

    return true;
}

/* This gets the current bond from the primitive unit cell into the symmetric
 * cell and then increments. The iterator loops over unique pairs once and
 * records the multiplicity for each pair as 2.
 *
 * To guarantee the consistency of the iterator, this should only be called by
 * the next() function.
 *
 * Returns true when the incrementer progressed, false otherwise.
 */
bool
BondIterator::
incrementps()
{
    /* iteri = slow iterator (outer loop, punit)
     * iterj = fast iterator (inner loop, sunit)
     */

    if(punit.size() == 0 || sunit.size() == 0) return false;
    // Check the termination condition.
    if(iteri == punit.end())
    {
        return false;
    }

    // If we got here, then we can record the bond
    for(size_t l=0;l<3;++l)
    {
        bp.xyz1[l] = iteri->xyz[l];
        bp.xyz2[l] = iterj->xyz[l];
    }
    bp.sc1 = iteri->sc;
    bp.sc2 = iterj->sc;
    bp.multiplicity = 2;

    // Now we can increment the fast iterator (iterj).
    ++iterj;

    // If the fast iterator is at the end then increment the slow iterator
    // (iteri) and set the fast iterator to the beginning.
    if(iterj == sunit.end())
    {
        ++iteri;
        iterj = sunit.begin();
    }

    return true;
}

/* This gets the current bond from the symmetric unit cell into the symmetric
 * cell and then increments. This is an exclusive iterator as it loops over
 * unique ssc pairs. The iterator loops over unique pairs once and records the
 * multiplicity for each pair as 2.
 *
 * To guarantee the consistency of the iterator, this should only be called by
 * the next() function.
 *
 * Returns true when the incrementer progressed, false otherwise.
 */
bool
BondIterator::
incrementss()
{

    /* iteri = slow iterator (outer loop)
     * iterj = fast iterator (inner loop)
     */

    if(sunit.size() < 2) return false;
    // Check the termination condition.
    if(iteri == sunit.end()-1 && iterj == sunit.end())
    {
        return false;
    }

    // If we got here, then we can record the bond
    for(size_t l=0;l<3;++l)
    {
        bp.xyz1[l] = iteri->xyz[l];
        bp.xyz2[l] = iterj->xyz[l];
    }
    bp.sc1 = iteri->sc;
    bp.sc2 = iterj->sc;
    bp.multiplicity = 2;

    // Now we can increment the fast iterator.
    ++iterj;

    // If the fast iterator is at the end then increment the slow iterator and
    // set the fast iterator one step beyond it.
    if(iterj == sunit.end())
    {
        ++iteri;
        iterj = iteri +1;
    }

    return true;
}

/* This gets the current bond from the primitive unit cell into an image of the
 * primitive unit cell and then increments. The iterator loops over all pairs
 * and records the multiplicity for each pair. If the ssc in the primitive unit
 * and the image are the same, then the multiplicity is set as the degeneracy of
 * the primitive scatterer in the conventional unit. Otherwise, the multiplicity
 * is 2 since we enforce that iterj >= iteri.
 *
 * To guarantee the consistency of the iterator, this should only be called by
 * the next() function.
 *
 * Returns true when the incrementer progressed, false otherwise.
 */
bool
BondIterator::
incrementppi()
{
    /* iteri = slow iterator (outer loop)
     * iterj = fast iterator (inner loop)
     * sph = fastest iterator (inner-inner loop)
     */

    if(punit.size() == 0) return false;
    // Check the termination condition. We end when the outer loop finishes, but
    // we must also check the case where the PointsInSphere iterator has only
    // one point at the origin. In this case, there are no images so we
    // terminate.
    if( iteri == punit.end() || sph->finished())
    {
        return false;
    }

    // If we got here, then we can record the bond
    for(size_t l=0;l<3;++l)
    {
        bp.xyz1[l] = iteri->xyz[l];
        bp.xyz2[l] = iterj->xyz[l];
    }
    placeInSphere(bp.xyz2);
    bp.sc1 = iteri->sc;
    bp.sc2 = iterj->sc;

    if( *iteri == *iterj )
    {
        bp.multiplicity = degen[*iteri];
    }
    else
    {
        bp.multiplicity = 2;
    }

    // Increment sph past the origin (if possible)
    sph->next();
    if(sphAtOrigin()) sph->next();

    // If sph is finished, then we reset it and increment iterj
    if( sph->finished() )
    {
        sph->rewind();
        if(sphAtOrigin()) sph->next();
        ++iterj;
    }

    // If iterj is finished, we increment iteri and set iterj equal to it
    if( iterj == punit.end() )
    {
        ++iteri;
        iterj = iteri;
    }

    return true;
}

/* This gets the current bond from the primitive unit cell into an image of the
 * symmetric unit cell and then increments. The iterator loops over all pairs
 * and records the multiplicity as 2 for each pair. Note that we may miss some
 * p--si bonds for a given mno, but we will find equivalent bonds at -(mno),
 * which is why the multiplicity is 2.
 *
 * To guarantee the consistency of the iterator, this should only be called by
 * the next() function.
 *
 * Returns true when the incrementer progressed, false otherwise.
 */
bool
BondIterator::
incrementpsi()
{
    /* iteri = slow iterator (outer loop)
     * iterj = fast iterator (inner loop)
     * sph = fastest iterator (inner-inner loop)
     */

    if(punit.size() == 0 || sunit.size() == 0) return false;
    // Check the termination condition. We end when the outer loop finishes, but
    // we must also check the case where the PointsInSphere iterator has only
    // one point at the origin. In this case, there are no images so we
    // terminate.
    if(iteri == punit.end() || sph->finished())
    {
        return false;
    }

    // If we got here, then we can record the bond
    for(size_t l=0;l<3;++l)
    {
        bp.xyz1[l] = iteri->xyz[l];
        bp.xyz2[l] = iterj->xyz[l];
    }
    placeInSphere(bp.xyz2);
    bp.sc1 = iteri->sc;
    bp.sc2 = iterj->sc;
    bp.multiplicity = 2;

    // Increment sph past the origin (if possible)
    sph->next();
    if(sphAtOrigin()) sph->next();

    // If sph is finished, then we reset it and increment iterj
    if( sph->finished() )
    {
        sph->rewind();
        if(sphAtOrigin()) sph->next();
        ++iterj;
    }

    // If iterj is finished, we increment iteri and reset iterj
    if( iterj == sunit.end() )
    {
        ++iteri;
        iterj = sunit.begin();
    }

    return true;
}

/* This gets the current bond from the symmetric unit cell into an image of the
 * symmetric unit cell and then increments. The iterator is exclusive as pairs
 * involving the same ssc are handled in incrementssi. The multiplicity
 * is 2 since we enforce that iterj >= iteri.
 *
 * To guarantee the consistency of the iterator, this should only be called by
 * the next() function.
 *
 * Returns true when the incrementer progressed, false otherwise.
 */
bool
BondIterator::
incrementssi()
{
    /* iteri = slow iterator (outer loop)
     * iterj = fast iterator (inner loop)
     * sph = fastest iterator (inner-inner loop)
     */

    if(sunit.size() == 0) return false;
    // Check the termination condition. We end when the outer loop finishes, but
    // we must also check the case where the PointsInSphere iterator has only
    // one point at the origin. In this case, there are no images so we
    // terminate.
    if(iteri == sunit.end() || sph->finished())
    {
        return false;
    }

    // If we got here, then we can record the bond
    for(size_t l=0;l<3;++l)
    {
        bp.xyz1[l] = iteri->xyz[l];
        bp.xyz2[l] = iterj->xyz[l];
    }
    placeInSphere(bp.xyz2);
    bp.sc1 = iteri->sc;
    bp.sc2 = iterj->sc;
    bp.multiplicity = 2;

    // Increment sph past the origin (if possible)
    sph->next();
    if(sphAtOrigin()) sph->next();

    // If sph is finished, then we reset it and increment iterj
    if( sph->finished() )
    {
        sph->rewind();
        if(sphAtOrigin()) sph->next();
        ++iterj;
    }

    // If iterj is finished, we increment iteri and set iterj just above it
    if( iterj == sunit.end() )
    {
        ++iteri;
        iterj = iteri;
        ++iterj;
    }

    return true;
}

/* Place cartesian coordinates xyz in the location defined by the PointsInSphere
 * iterator.
 */
void
BondIterator::
placeInSphere(float *xyz)
{ 
    static float dxyz[3];
    for(size_t l=0; l<3; ++l)
    {
        dxyz[l] = sph->mno[l];
    }
    crystal.FractionalToOrthonormalCoords(dxyz[0], dxyz[1], dxyz[2]);
    for(size_t l=0; l<3; ++l) 
    {
        xyz[l] += dxyz[l];
    }
    return;
}

/******************************************************************************
***** ShiftedSC implementation ************************************************
******************************************************************************/

ShiftedSC::
ShiftedSC(const ObjCryst::ScatteringComponent *_sc,
    const float x, const float y, const float z, const int _id) :
    sc(_sc), id(_id)
{
    //sc->Print();
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;
    //std::cout << x << ' ' << y << ' ' << z << endl;
}

// Be careful of dangling references
ShiftedSC::
ShiftedSC()
{
    xyz[0] = xyz[1] = xyz[2] = 0;
    id = -1;
    sc = NULL;
}

bool
ShiftedSC::
operator<(const ShiftedSC &rhs) const
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

bool
ShiftedSC::
operator==(const ShiftedSC &rhs) const
{

    //std::cout << id << " vs " << rhs.id << endl;

    return ((xyz[0] == rhs.xyz[0]) 
        && (xyz[1] == rhs.xyz[1]) 
        && (xyz[2] == rhs.xyz[2])
        && (*sc == *(rhs.sc)));
}
