#include "bonditerator.h"

#include "ObjCryst/Crystal.h" // From ObjCryst distribution
#include "ObjCryst/Atom.h" // From ObjCryst distribution
#include "ObjCryst/ScatteringPower.h" // From ObjCryst distribution

#include <string>
#include <vector>

using namespace std;
using namespace SrReal;

// FIXME 

void test1()
{
    string sgstr("224");
    string estr("Ni");

    // Create the Ni structure
    ObjCryst::Crystal crystal(3.52, 3.52, 3.52, sgstr);
    ObjCryst::ScatteringPowerAtom sp(estr, estr);
    sp.SetBiso(8*M_PI*M_PI*0.003);
    // Atoms only belong to one crystal. They must be allocated in the heap.
    ObjCryst::Atom *atomp = new ObjCryst::Atom(0.0, 0.0, 0.0, estr, &sp);
    crystal.AddScatterer(atomp);

    BondIterator biter(crystal, 0, 5);
    ObjCryst::ScatteringComponentList scl 
        = crystal.GetScatteringComponentList();
    BondPair bp;
    double dist = 0;
    for(size_t i=0; i < scl.GetNbComponent(); ++i)
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

}

void test2()
{
    string sgstr("P 63 m c");
    string zstr("Zn");
    string sstr("S");
    string cstr("C");

    // Create the ZnS structure
    ObjCryst::Crystal crystal(3.811, 3.811, 6.234, 90, 90, 120, sgstr);
    ObjCryst::ScatteringPowerAtom zsp(zstr, zstr);
    ObjCryst::ScatteringPowerAtom ssp(sstr, sstr);
    ObjCryst::ScatteringPowerAtom csp(cstr, cstr);
    zsp.SetBiso(8*M_PI*M_PI*0.003);
    ssp.SetBiso(8*M_PI*M_PI*0.003);
    csp.SetBiso(8*M_PI*M_PI*0.003);
    // Atoms only belong to one crystal. They must be allocated in the heap.
    ObjCryst::Atom *zatomp = new ObjCryst::Atom(1.0/3, 2.0/3, 0.0, zstr, &zsp);
    ObjCryst::Atom *satomp = new ObjCryst::Atom(1.0/3, 2.0/3, 0.385, sstr, &ssp);
    ObjCryst::Atom *catomp = new ObjCryst::Atom(1.0/3, 2.0/3, 0.385, cstr, &csp);
    crystal.AddScatterer(zatomp);
    crystal.AddScatterer(satomp);
    crystal.AddScatterer(catomp);

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
    for(size_t i=0; i < scl.GetNbComponent(); ++i)
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



}


int main()
{
    test1();
    test2();
}
