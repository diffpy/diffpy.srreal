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
const double rtod = 180 / M_PI;
const double UtoB = 8 * M_PI * M_PI;
const double BtoU = 1.0 / UtoB;

}



//////////////////////////////////////////////////////////////////////////////
// class ObjCrystStructureAdapter
//////////////////////////////////////////////////////////////////////////////

// Constructor ---------------------------------------------------------------

const double ObjCrystStructureAdapter::toler = 1e-5;

ObjCrystStructureAdapter::
ObjCrystStructureAdapter(const ObjCryst::Crystal& cryst) : pcryst(&cryst)
{
    mlattice.setLatPar( pcryst->GetLatticePar(0), 
                        pcryst->GetLatticePar(1),
                        pcryst->GetLatticePar(2), 
                        rtod * pcryst->GetLatticePar(3),
                        rtod * pcryst->GetLatticePar(4), 
                        rtod * pcryst->GetLatticePar(5) );
    this->getUnitCell();
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
    return vsc.size();
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
    return *(vsym[idx].begin());
}


double 
ObjCrystStructureAdapter::
siteOccupancy(int idx) const
{
    assert(0 <= idx && idx < this->countSites());
    return vsc[idx].mOccupancy;
}


bool 
ObjCrystStructureAdapter::
siteAnisotropy(int idx) const
{
    assert(0 <= idx && idx < this->countSites());
    return vsc[idx].mpScattPow->IsIsotropic();
}

double
ObjCrystStructureAdapter::
siteMultiplicity(int idx) const
{

    assert(0 <= idx && idx < this->countSites());
    return vsym[idx].size();
}

const R3::Matrix&
ObjCrystStructureAdapter::
siteCartesianUij(int idx) const
{
    assert(0 <= idx && idx < this->countSites());
    return vuij[idx];
}


const string& 
ObjCrystStructureAdapter::
siteAtomType(int idx) const
{
    assert(0 <= idx && idx < this->countSites());
    return vsc[idx].mpScattPow->GetSymbol();
}


// Private Methods -----------------------------------------------------------

/* Get the conventional unit cell from the crystal. */
void
ObjCrystStructureAdapter::
getUnitCell()
{
    // Expand each scattering component in the primitive cell and record the
    // new scatterers.
    const ObjCryst::ScatteringComponentList& scl =
        pcryst->GetScatteringComponentList();
    size_t nbComponent = scl.GetNbComponent();

    size_t nbSymmetrics = pcryst->GetSpaceGroup().GetNbSymmetrics();

    double x, y, z, junk;
    const ObjCryst::ScatteringPower* sp  = NULL;
    CrystMatrix<double> symmetricsCoords;

    vsc.clear();
    vsym.clear();
    vuij.clear();
    vsc.resize(nbComponent);
    vsym.resize(nbComponent);
    vuij.resize(nbComponent);

    // For each scattering component, find its position in the primitive cell
    // and expand that position. Record the translation from the original
    // position.
    for(size_t i=0;i<nbComponent;++i)
    {


        vsc[i] = scl(i);

        // Get all the symmetric coordinates
        symmetricsCoords = pcryst->GetSpaceGroup().GetAllSymmetrics(
            scl(i).mX, 
            scl(i).mY, 
            scl(i).mZ
            );

        // Collect the unique symmetry operations.
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

            // Record the position
            R3::Vector xyz;
            xyz[0] = x;
            xyz[1] = y;
            xyz[2] = z;

            vsym[i].insert(xyz);

        }


        // Store the uij tensor
        R3::Matrix Ucart;
        sp = vsc[i].mpScattPow;
        if( sp->IsIsotropic() )
        {
            Ucart(0,0) = Ucart(1,1) = Ucart(2,2) = sp->GetBiso() * BtoU;
            Ucart(0,1) = Ucart(1,0) = 0;
            Ucart(0,2) = Ucart(2,0) = 0;
            Ucart(2,1) = Ucart(1,2) = 0;
        }
        else
        {
            R3::Matrix Ufrac;
            Ufrac(0,0) = sp->GetBij(1,1) * BtoU;
            Ufrac(1,1) = sp->GetBij(2,2) * BtoU;
            Ufrac(2,2) = sp->GetBij(3,3) * BtoU;
            Ufrac(0,1) = Ufrac(1,0) = sp->GetBij(1,2) * BtoU;
            Ufrac(0,2) = Ufrac(2,0) = sp->GetBij(1,3) * BtoU;
            Ufrac(1,2) = Ufrac(2,1) = sp->GetBij(2,3) * BtoU;
            Ucart = mlattice.cartesianMatrix(Ufrac);
        }
        vuij[i] = Ucart;
    }
}

//////////////////////////////////////////////////////////////////////////////
// class ObjCrystBondGenerator
//////////////////////////////////////////////////////////////////////////////

// Constructor ---------------------------------------------------------------

ObjCrystBondGenerator::
ObjCrystBondGenerator(const ObjCrystStructureAdapter* adpt) 
    : BaseBondGenerator(adpt), pstructure(adpt)
{
}

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
    symiter = pstructure->vsym[msite_first].begin();
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
    assert( symiter != pstructure->vsym[msite_current].end() );
    rv = *symiter + L.cartesian(msphere->mno());
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
    // Iterate the sphere. If it is finished, rewind and iterate the symmetry
    // iterator. If that is also finished, then we're done.
    msphere->next();
    if( msphere->finished() )
    {
        if(++symiter == pstructure->vsym[msite_current].end())
        {
            return false;
        }
        msphere->rewind();
    }
    return true;
}


void 
ObjCrystBondGenerator::
rewindSymmetry()
{
    msphere->rewind();
    symiter = pstructure->vsym[msite_current].begin();
}


double 
ObjCrystBondGenerator::
msdSiteDir(int siteidx, const R3::Vector& s) const
{
    const R3::Matrix& Uijcartn = pstructure->siteCartesianUij(siteidx);
    bool anisotropy = pstructure->siteAnisotropy(siteidx);
    double rv = meanSquareDisplacement(Uijcartn, s, anisotropy);
    return rv;
}



// End of file
