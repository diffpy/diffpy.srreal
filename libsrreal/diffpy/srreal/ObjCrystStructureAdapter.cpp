/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Chris Farrow
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* class ObjCrystStructureAdapter -- adapter to the Crystal class from
* ObjCryst++.
* class ObjCrystBondGenerator -- Generate bonds from ObjCrystStructureAdapter.
*     
* class ObjCrystBondGenerator -- bond generator
*
*
* $Id$
*
*****************************************************************************/

#include <algorithm>
#include <cassert>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include <ObjCryst/Crystal.h>
#include <ObjCryst/ScatteringPower.h>

#include <diffpy/srreal/Lattice.hpp>
#include <diffpy/srreal/ObjCrystStructureAdapter.hpp>
#include <diffpy/srreal/PDFCalculator.hpp>
#include <diffpy/srreal/PDFUtils.hpp>
#include <diffpy/srreal/PointsInSphere.hpp>

using namespace std;
using namespace diffpy::srreal;

namespace {

// Two coordinates are the same if they are within this tolerance
const double toler = 1e-5;
const double rtod = 180 / M_PI;
const double UtoB = 8 * M_PI * M_PI;
const double BtoU = 1.0 / UtoB;

}


//////////////////////////////////////////////////////////////////////////////
// class ObjCrystStructureAdapter
//////////////////////////////////////////////////////////////////////////////

// Constructor ---------------------------------------------------------------

ObjCrystStructureAdapter::
ObjCrystStructureAdapter(const ObjCryst::Crystal* cryst) : pcryst(cryst)
{

    getUnitCell();
    mlattice.setLatPar( cryst->GetLatticePar(0), 
                        cryst->GetLatticePar(1),
                        cryst->GetLatticePar(2), 
                        rtod*cryst->GetLatticePar(3),
                        rtod*cryst->GetLatticePar(4), 
                        rtod*cryst->GetLatticePar(5) );

}

// Public Methods ------------------------------------------------------------

BaseBondGenerator* 
ObjCrystStructureAdapter::
createBondGenerator() const
{
    BaseBondGenerator* bnds = new ObjCrystBondGenerator(this);
    return bnds;
}


int 
ObjCrystStructureAdapter::
countSites() const
{
    return vssc.size();
}


double 
ObjCrystStructureAdapter::
numberDensity() const
{
    double rv = this->totalOccupancy() / pcryst->GetVolume();
    return rv;
}


const Lattice& 
ObjCrystStructureAdapter::
getLattice() const
{
    return mlattice;
}


const R3::Vector& 
ObjCrystStructureAdapter::
siteCartesianPosition(int idx) const
{
    assert(0 <= idx && idx < this->countSites());
    
    return vssc[idx].xyz;
}


double 
ObjCrystStructureAdapter::
siteOccupancy(int idx) const
{
    assert(0 <= idx && idx < this->countSites());
    return vssc[idx].sc->mOccupancy;
}


bool 
ObjCrystStructureAdapter::
siteAnisotropy(int idx) const
{
    assert(0 <= idx && idx < this->countSites());
    return false;
}


const R3::Matrix&
ObjCrystStructureAdapter::
siteCartesianUij(int idx) const
{
    assert(0 <= idx && idx < this->countSites());
    return vssc[idx].uij;
}


const string& 
ObjCrystStructureAdapter::
siteAtomType(int idx) const
{
    assert(0 <= idx && idx < this->countSites());
    return vssc[idx].sc->mpScattPow->GetSymbol();
}


// Private Methods -----------------------------------------------------------

/* Get the conventional unit cell from the crystal. */
void
ObjCrystStructureAdapter::
getUnitCell()
{
    // Expand each scattering component in the primitive cell and record the
    // new scatterers.
    const ObjCryst::ScatteringComponentList& scl 
        = pcryst->GetScatteringComponentList();

    size_t nbComponent = scl.GetNbComponent();

    size_t nbSymmetrics = pcryst->GetSpaceGroup().GetNbSymmetrics();

    double x, y, z, junk;
    CrystMatrix<double> symmetricsCoords;
    set<ShiftedSC> workset;
    ShiftedSC workssc;
    // For each scattering component, find its position in the primitive cell
    // and expand that position. Record this as a ShiftedSC.
    // NOTE - I've also tried this algorithm by finding the unique elements in a
    // vector. The speed of that method is comparable to this one.
    for(size_t i=0;i<nbComponent;++i)
    {
        // We don't want to record dummy atoms...
        if(NULL == scl(i).mpScattPow)
        {
            continue;
        }

        // NOTE - there is an option in GetAllSymmetrics to return only distinct
        // atom positions. The way it is done below is faster.
        // NOTE - this identifies unique atoms by their scattering power.
        // Therefore, if multiple atoms exist on the same site and have the
        // same ScatteringPower, then only one will make it into the unit cell.
        symmetricsCoords = pcryst->GetSpaceGroup().GetAllSymmetrics(
            scl(i).mX, 
            scl(i).mY, 
            scl(i).mZ
            );

        // Put each symmetric position in the unit cell
        for(size_t j=0;j<nbSymmetrics;++j)
        {
            x=modf(symmetricsCoords(j,0),&junk);
            y=modf(symmetricsCoords(j,1),&junk);
            z=modf(symmetricsCoords(j,2),&junk);
            if(fabs(x) < toler) x = 0;
            if(fabs(y) < toler) y = 0;
            if(fabs(z) < toler) z = 0;
            if(x<0) x += 1.;
            if(y<0) y += 1.;
            if(z<0) z += 1.;

            // Get this in cartesian
            pcryst->FractionalToOrthonormalCoords(x,y,z);

            // Store it in the scatterer set. This will eliminate duplicates.
            workssc = ShiftedSC(&scl(i),x,y,z);
            workset.insert(workssc);
        }

    }

    // Now record the unique scatterers in vssc
    vssc.resize( workset.size() );
    copy(workset.begin(), workset.end(), vssc.begin());
}

//////////////////////////////////////////////////////////////////////////////
// class ObjCrystStructureAdapter::ShiftedSC
//////////////////////////////////////////////////////////////////////////////

/* Constructor */
ObjCrystStructureAdapter::ShiftedSC::
ShiftedSC(const ObjCryst::ScatteringComponent *_sc,
    const double& x, const double& y, const double& z) : sc(_sc)
{
    xyz[0] = x;
    xyz[1] = y;
    xyz[2] = z;

    double uiso = _sc->mpScattPow->GetBiso();
    uiso *= BtoU;
    uij(0,0) = uij(1,1) = uij(2,2) = uiso;
    uij(0,1) = uij(1,0) = 0;
    uij(0,2) = uij(2,0) = 0;
    uij(2,1) = uij(1,2) = 0;

}

/* Copy Constructor */
ObjCrystStructureAdapter::ShiftedSC::
ShiftedSC(const ShiftedSC& _ssc) : sc(_ssc.sc), xyz(_ssc.xyz), uij(_ssc.uij)
{}

/* Default Constructor - This is only used for sorting. It should not be called
 * directly.
*/
ObjCrystStructureAdapter::ShiftedSC::
ShiftedSC() : sc(NULL) 
{};

/* Ordering operator - required by the set used in getUnitCell */
bool
ObjCrystStructureAdapter::ShiftedSC::
operator<(const ShiftedSC& rhs) const
{
    // The sign of A-B is equal the sign of the first non-zero component of the
    // vector.

    size_t l;

    for(l = 0; l < 3; ++l)
    {
        if( fabs(xyz[l] - rhs.xyz[l]) > toler )
        {
            return xyz[l] < rhs.xyz[l];
        }
    }
    if(sc == NULL or rhs.sc == NULL) return false;

    // If we get here then the vectors are equal. We compare the addresses of
    // the ScatteringPower member of the ScatteringComponent
    return sc->mpScattPow < rhs.sc->mpScattPow;

}

/* Equality operator. */
bool
ObjCrystStructureAdapter::ShiftedSC::
operator==(const ShiftedSC& rhs) const
{

    bool poseq = ((xyz[0] == rhs.xyz[0]) 
        && (xyz[1] == rhs.xyz[1]) 
        && (xyz[2] == rhs.xyz[2]));

    bool sceq;
    if(sc == NULL or rhs.sc == NULL) sceq = true;
    else sceq = (*sc == *(rhs.sc));

    return poseq && sceq;
}


//////////////////////////////////////////////////////////////////////////////
// class ObjCrystBondGenerator
//////////////////////////////////////////////////////////////////////////////

// Constructor ---------------------------------------------------------------

ObjCrystBondGenerator::
ObjCrystBondGenerator(const ObjCrystStructureAdapter* adpt) 
    : BaseBondGenerator(adpt), pstructure(adpt)
{}

// Public Methods ------------------------------------------------------------

void 
ObjCrystBondGenerator::
rewind()
{
    // Delay msphere instantiation to here instead of in constructor,
    // so it is possible to use setRmin, setRmax.
    if (!msphere.get())
    {
        // Make a Lattice instance
        const Lattice& L = pstructure->getLattice();
        double buffzone = L.ucMaxDiagonalLength();
        double rsphmin = this->getRmin() - buffzone;
        double rsphmax = this->getRmax() + buffzone;
        msphere.reset(new PointsInSphere(rsphmin, rsphmax, L));
    }
    msphere->rewind();
    this->BaseBondGenerator::rewind();
}


void
ObjCrystBondGenerator::
setRmin(double rmin)
{
    // destroy msphere so it will be created on rewind with new rmin
    if (this->getRmin() != rmin)    msphere.reset(NULL);
    this->BaseBondGenerator::setRmin(rmin);
}


void 
ObjCrystBondGenerator::
setRmax(double rmax)
{
    // destroy msphere so it will be created on rewind with new rmax
    if (this->getRmax() != rmax)    msphere.reset(NULL);
    this->BaseBondGenerator::setRmax(rmax);
}


const R3::Vector& 
ObjCrystBondGenerator::
r1() const
{
    static R3::Vector rv;
    const Lattice& L = pstructure->getLattice();
    rv = this->BaseBondGenerator::r1() + L.cartesian(msphere->mno());
    return rv;
}


double 
ObjCrystBondGenerator::
msd0() const
{
    double rv = this->msdSiteDir(this->site0(), this->r01());
    return rv;
}


double 
ObjCrystBondGenerator::
msd1() const
{
    double rv = this->msdSiteDir(this->site1(), this->r01());
    return rv;
}


bool 
ObjCrystBondGenerator::
iterateSymmetry()
{
    msphere->next();
    return !msphere->finished();
}


void 
ObjCrystBondGenerator::
rewindSymmetry()
{
    msphere->rewind();
}


double 
ObjCrystBondGenerator::
msdSiteDir(int siteidx, const R3::Vector& s) const
{
    // FIXME We don't need to worry about anisotropy for now
    //const R3::Matrix& Uijcartn = pstructure->siteCartesianUij(siteidx);
    //bool anisotropy = pstructure->siteAnisotropy(siteidx);
    //double rv = meanSquareDisplacement(Uijcartn, s, anisotropy);
    return pstructure->siteCartesianUij(siteidx)(0,0);
}

// End of file
