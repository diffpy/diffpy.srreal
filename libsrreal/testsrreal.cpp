#include "bonditerator.h"
#include "pdfcalculator.h"

#include "ObjCryst/Crystal.h" // From ObjCryst distribution
#include "ObjCryst/Atom.h" // From ObjCryst distribution
#include "ObjCryst/ScatteringPower.h" // From ObjCryst distribution

#include <string>
#include <vector>

using namespace std;
using namespace SrReal;

// FIXME 

ObjCryst::Crystal* makeNi()
{
    // Create the Ni structure
    ObjCryst::Crystal* crystal = new ObjCryst::Crystal(3.52, 3.52, 3.52, "225");
    ObjCryst::ScatteringPowerAtom* sp
        = new ObjCryst::ScatteringPowerAtom("Ni", "Ni");
    sp->SetBiso(8*M_PI*M_PI*0.003);
    // Atoms only belong to one crystal. They must be allocated in the heap.
    ObjCryst::Atom *atomp = new ObjCryst::Atom(0.0, 0.0, 0.0, "Ni", sp);
    crystal->AddScatterer(atomp);
    crystal->AddScatteringPower(sp);
    return crystal;
}

ObjCryst::Crystal* makeLaMnO3()
{
    ObjCryst::Crystal* crystal = 
        new ObjCryst::Crystal(5.486341, 5.619215, 7.628206, 90, 90, 90, "P b n m");
    ObjCryst::ScatteringPowerAtom* sp;
    ObjCryst::Atom *atomp;
    // Add the atoms
    // La1
    sp = new ObjCryst::ScatteringPowerAtom("La1", "La");
    sp->SetBiso(8*M_PI*M_PI*0.003);
    atomp = new ObjCryst::Atom(0.996096, 0.0321494, 0.25, "La1", sp);
    crystal->AddScatterer(atomp);
    crystal->AddScatteringPower(sp);
    // Mn1
    sp = new ObjCryst::ScatteringPowerAtom("Mn1", "Mn");
    sp->SetBiso(8*M_PI*M_PI*0.003);
    atomp = new ObjCryst::Atom(0, 0.5, 0, "Mn1", sp);
    crystal->AddScatterer(atomp);
    crystal->AddScatteringPower(sp);
    // O1
    sp = new ObjCryst::ScatteringPowerAtom("O1", "O");
    sp->SetBiso(8*M_PI*M_PI*0.003);
    atomp = new ObjCryst::Atom(0.0595746, 0.496164, 0.25, "O1", sp);
    crystal->AddScatterer(atomp);
    crystal->AddScatteringPower(sp);
    // O2
    sp = new ObjCryst::ScatteringPowerAtom("O2", "O");
    sp->SetBiso(8*M_PI*M_PI*0.003);
    atomp = new ObjCryst::Atom(0.720052, 0.289387, 0.0311126, "O2", sp);
    crystal->AddScatterer(atomp);
    crystal->AddScatteringPower(sp);

    return crystal;
}

ObjCryst::Crystal* makeZnS()
{
    // Create the ZnS structure
    ObjCryst::Crystal* crystal 
        = new ObjCryst::Crystal(3.811, 3.811, 6.234, 90, 90, 120, "P 63 m c");
    ObjCryst::ScatteringPowerAtom* zsp 
        = new ObjCryst::ScatteringPowerAtom("Zn", "Zn");
    ObjCryst::ScatteringPowerAtom* ssp 
        = new ObjCryst::ScatteringPowerAtom("S", "S");
    ObjCryst::ScatteringPowerAtom* csp 
        = new ObjCryst::ScatteringPowerAtom("C", "C");
    zsp->SetBiso(8*M_PI*M_PI*0.003);
    ssp->SetBiso(8*M_PI*M_PI*0.003);
    csp->SetBiso(8*M_PI*M_PI*0.003);
    // Atoms only belong to one crystal. They must be allocated in the heap.
    ObjCryst::Atom *zatomp = 
        new ObjCryst::Atom(1.0/3, 2.0/3, 0.0, "Zn", zsp);
    ObjCryst::Atom *satomp = 
        new ObjCryst::Atom(1.0/3, 2.0/3, 0.385, "S", ssp, 1);
    ObjCryst::Atom *catomp = 
        new ObjCryst::Atom(1.0/3, 2.0/3, 0.385, "C", csp, 0);
    crystal->AddScatterer(zatomp);
    crystal->AddScatterer(satomp);
    crystal->AddScatterer(catomp);
    return crystal;
}

void test1()
{
    ObjCryst::Crystal& crystal = *makeLaMnO3();

    BondIterator biter(crystal, 0, 5);
    vector<ShiftedSC> unitcell = biter.getUnitCell();
    for(size_t i=0; i<unitcell.size(); ++i)
    {
        cout << unitcell[i] << endl;
    }

    ObjCryst::ScatteringComponentList scl 
        = crystal.GetScatteringComponentList();
    BondPair bp;
    for(int i=0; i < scl.GetNbComponent(); ++i)
    {
        biter.setScatteringComponent(scl(i));
        cout << "---- " << i << " ----" << endl;

        for(biter.rewind(); !biter.finished(); biter.next())
        {
            bp = biter.getBondPair();

            cout << bp.getDistance() << " ";
            cout << bp << endl;
        }
    }

}

void test2()
{
    ObjCryst::Crystal& crystal = *makeZnS();

    std::vector<ShiftedSC> uc = getUnitCell(crystal);

    std::vector<ShiftedSC>::iterator ssciter;

    for(ssciter=uc.begin(); ssciter!=uc.end(); ++ssciter)
    {
        cout << *ssciter << endl;
    }

    BondIterator biter(crystal, 0, 5);
    ObjCryst::ScatteringComponentList scl 
        = crystal.GetScatteringComponentList();
    BondPair bp;
    double dist = 0;
    for(int i=0; i < scl.GetNbComponent(); ++i)
    {
        biter.setScatteringComponent(scl(i));
        cout << "---- " << i << " ----" << endl;

        for(biter.rewind(); !biter.finished(); biter.next())
        {
            bp = biter.getBondPair();
            dist = 0;

            for(int i = 0; i < 3; ++i )
            {
                dist += pow(bp.getXYZ1(i)-bp.getXYZ2(i),2);
            }
            dist = sqrt(dist);

            cout << dist << " ";
            cout << bp << endl;
        }
    }
    ObjCryst::RefinablePar x = crystal.GetScatt("Zn").GetPar("x");
    x.SetValue(0);
    biter.rewind();
    biter.rewind();

}

void test3()
{

    ObjCryst::Crystal& crystal = *makeLaMnO3();

    // Create the calculation points
    float rmin, rmax, dr;
    rmin = 0;
    rmax = 10;
    dr = 0.01;
    size_t numpoints = static_cast<size_t>(ceil((rmax-rmin)/dr));
    float *rvals = new float [numpoints];
    for(size_t i=0; i<numpoints; ++i)
    {
        // Test a non-uniform grid
        // float dshift = (rand()%9)/10.0;
        // rvals[i] = rmin + dr*(i+dshift);
        rvals[i] = rmin + dr*i;
    }

    // Create the iterator and calculators
    BondIterator biter(crystal);
    JeongBWCalculator bwcalc;
    bwcalc.setDelta2(0.0);
    PDFCalculator pdfcalc(biter, bwcalc);
    pdfcalc.setCalculationPoints(rvals, numpoints);

    float *pdf = pdfcalc.getPDF();

    for(size_t i=0; i<numpoints; ++i)
    {
        cout << rvals[i] << "  " << pdf[i] << endl;
    }

    delete [] rvals;
    delete [] pdf;

    return;

}

void speedTest()
{

    ObjCryst::Crystal& crystal = *makeLaMnO3();
    // Create the calculation points
    float rmin, rmax, dr;
    rmin = 0;
    rmax = 10;
    dr = 0.01;
    size_t numpoints = static_cast<size_t>(ceil((rmax-rmin)/dr));
    float *rvals = new float [numpoints];
    for(size_t i=0; i<numpoints; ++i)
    {
        // Test a non-uniform grid
        // float dshift = (rand()%9)/10.0;
        // rvals[i] = rmin + dr*(i+dshift);
        rvals[i] = rmin + dr*i;
    }

    BondIterator biter(crystal);
    JeongBWCalculator bwcalc;

    PDFCalculator pdfcalc(biter, bwcalc);
    pdfcalc.setCalculationPoints(rvals, numpoints);

    // Change the bwcalc.
    cout << "change delta2" << endl;
    bwcalc.setDelta2(5);
    pdfcalc.getPDF();

    // Change an x-coordinate
    cout << "change 1 scatt" << endl;
    ObjCryst::Scatterer& scatla = crystal.GetScatt("La1");
    scatla.GetClockScatterer().Print();
    scatla.SetX(0.8);
    scatla.GetClockScatterer().Print();
    pdfcalc.getPDF();

    // Change an thermal parameter
    cout << "Change Biso" << endl;
    ObjCryst::ScatteringPower& sp = crystal.GetScatteringPower("La1");
    sp.SetBiso(8*M_PI*M_PI*0.008);
    pdfcalc.getPDF();

    // Change another atom
    cout << "Change in atom coordinate" << endl;
    ObjCryst::Scatterer& scato1 = crystal.GetScatt("O1");
    scato1.GetClockScatterer().Print();
    scato1.SetX(0.05);
    scato1.GetClockScatterer().Print();
    pdfcalc.getPDF();

    // Change properties of two atoms. Should
    cout << "Change in two atoms" << endl;
    scatla.GetClockScatterer().Print();
    scatla.SetX(0.9);
    scatla.GetClockScatterer().Print();
    scato1.GetClockScatterer().Print();
    scato1.SetX(0.07);
    scato1.GetClockScatterer().Print();
    pdfcalc.getPDF();

    return;
}

int main()
{
    speedTest();
}
