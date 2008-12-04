/***********************************************************************
* $Id$
***********************************************************************/

#include "bonditerator.h"

using namespace SrReal

BondIterator::
BondIterator (ObjCryst::Crystal &_crystal, float _rmin, float _rmax)
    : crystal(_crystal), rmin(_rmin), rmax(_rmax)
    
{
    init();
}


BondIterator::
BondIterator(const BondIterator &other)
{
    BondIterator(other.crystal, other.rmin, other.rmax);
}

void
BondIterator::
rewind()
{
    // Set the iterators up for the beginning of the next functions, whcih is
    // nextss.
    iteri = sunit.begin();
    iterj = sunit.begin(); ++iterj;
    sph.rewind();
    state = PP;

    // Initialize the first bond
    next();

}

/* This sets sc1 and sc2 to the next pair in the iteration sequence. The
 * sequence is as follows:
 * nextpp
 * nextps
 * nextss
 * nextppi
 * nextpsi
 * nextssi
 *
 * The procedures and meaning of these iteration incrementers are described
 * before their implementations below.
 *
 * The duty of this function is to call the appropriate incrementer based on the
 * state of the iterator. It is the responsibility of the incrementer to set sc1
 * and sc2 and indicate if it was successful.
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
                iteri = sunit.begin();
                iterj = punit.begin();
                next();
            }
            break;

        case PS :
            if( !incrementps() )
            { 
                state = SS;
                iteri = sunit.begin();
                iterj = sunit.begin();
                ++iterj;
                next();
            }
            break;

        case SS :
            if( !incrementsp() ) 
            {
                state = PPI;
                iteri = punit.begin();
                iterj = punit.begin();
                sph.rewind();
                next();
            }
            break;

        case PPI :
            if( !incrementppi() ) 
            {
                state = PSI;
                iteri = punit.begin();
                iterj = sunit.begin();
                sph.rewind();
                next();
            }
            break;

        case PSI :
            if( !incrementpsi() ) 
            {
                state = SSI;
                iteri = sunit.begin();
                iterj = sunit.begin();
                sph.rewind();
                next();
            }
            break;

        case SSI :
            if( !incrementssi() ) state = FINISHED;
            break;

    }

    return;
}

bool
BondIterator::
finished()
{
    return (state == FINISHED);
}

void
BondIterator::
reset()
{
    sunit.clear();
    punit.clear();
    degen.clear();
    init();
    return;
}

/* This expands primitive cell of the crystal and fills punit and sunit and then
 * calls reset().
 * FIXME - Need to count the multiplicity of each site from the primitive unit
 * cell into the expanded unit cell.
 */
void
BondIterator::
init()
{

    // Expand the primitive cell and store the resulting scattering components.crystal 
    const ObjCryst::ScatteringComponentList &mScattCompList 
        = crystal.GetScatteringComponentList();

    long nbComponent = mScattCompList.GetNbComponent();

    // Get the origin of the primitive unit
    const float x0 = crystal.GetSpaceGroup().GetAsymUnit().Xmin();
    const float y0 = crystal.GetSpaceGroup().GetAsymUnit().Ymin();
    const float z0 = crystal.GetSpaceGroup().GetAsymUnit().Zmin();

    // Number of symmetric positions
    const int nbSymmetrics=crystal.GetSpaceGroup().GetNbSymmetrics();
    std::cout << "nbSymmetrics = " << nbSymmetrics << std::endl;

    float x, y, z;
    float junk;
    // Get the list of all atoms within or near the asymmetric unit
    CrystMatrix<float> symmetricsCoords;
    for(long i=0;i<nbComponent;i++)
    {
    symmetricsCoords = crystal.GetSpaceGroup().GetAllSymmetrics(
            mScattCompList(i).mX, 
            mScattCompList(i).mY, 
            mScattCompList(i).mZ
            );

       for(int j=0;j<nbSymmetrics;j++)
       {
          x=modf(symmetricsCoords(j,0)-x0,&junk);
          y=modf(symmetricsCoords(j,1)-y0,&junk);
          z=modf(symmetricsCoords(j,2)-z0,&junk);
          if(x<0) x +=1.;
          if(y<0) y +=1.;
          if(z<0) z +=1.;

          //std::cout << &mScattCompList(i) << ' ' << x << ' ' << y << ' ' << z << std::endl;
          

          // Store it in the primitive cell vector if its the primitive scatterer.
          if( j == 0 )
          {
              punit.push_back(ShiftedSC(&mScattCompList(i),x,y,z));
          }
          // Store this ShiftedSC in the symmetry cell
          else
          {
              sunit.push_back(ShiftedSC(&mScattCompList(i),x,y,z));
          }
       }
    }

    // Clear out all duplicate entries in the sunit
    sort(punit.begin(), punit.end());
    sort(sunit.begin(), sunit.end());
    vector<ShiftedSC>::iterator it1;
    it1 = unique(sunit.begin(), sunit.end());
    sunit.erase(it1, sunit.end());

    // Count the degeneracy of each primitive atom in the conventional unit. We
    // do this by seeing if they have the same scattering component.
    // FIXME - Check that this is doing what I think it is.
    vector<ShiftedSC>::iterator it2;
    degen.resize(nbComponent);
    size_t counter = 0;
    for(it1=punit.begin(); it1!=punit.end(); ++it1)
    {
        degen[counter] = 1;
        for(it2=sunit.begin(); it2!=sunit.end(); ++it2)
        {
            if( it1->sc == it2->sc ) ++degen[counter];
        }
        ++counter;
    }

    // Set up the PointsInSphere iterator
    sph(rmin, rmax, 
        crystal.GetLatticePar(0)
        crystal.GetLatticePar(1)
        crystal.GetLatticePar(2)
        crystal.GetLatticePar(3)
        crystal.GetLatticePar(4)
        crystal.GetLatticePar(5)
       );

    // Get the iterators ready
    rewind();

}

/* This gets the current bond from the primitive unit cell into the conventional
 * unit cell and then increments. In its lifetime, the incrementer loops over
 * the N(N-1)/2 pairs in the primitive unit cell and counts each bond
 * twice.
 *
 * Returns true when the incrementer progressed, false otherwise.
 */
bool
BondIterator::
incrementpp()
{
    return false;
}

bool
BondIterator::
incrementps()
{
    return false;
}

bool
BondIterator::
incrementss()
{
    return false;
}

bool
BondIterator::
incrementppi()
{
    return false;
}

bool
BondIterator::
incrementpsi()
{
    return false;
}

bool
BondIterator::
incrementssi()
{
    return false;
}

