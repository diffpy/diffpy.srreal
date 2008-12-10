#include "bonditerator.h"

#include "ObjCryst/Crystal.h" // From ObjCryst distribution
#include "ObjCryst/Atom.h" // From ObjCryst distribution
#include "ObjCryst/ScatteringPower.h" // From ObjCryst distribution

#include <string>

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

    BondIterator biter(crystal, 0, 10);

    BondPair bp;

    double dist = 0;
    for(biter.rewind(); !biter.finished(); biter.next())
    {
        bp = biter.getBondPair();
        dist = 0;

        for(int i = 0; i < 3; ++i )
        {
            dist += pow(bp.getXYZ1(i)-bp.getXYZ2(i),2);
        }
        dist = sqrt(dist);

        //cout << dist << endl;
        cout << bp << endl;
    }

}

int main()
{
    test1();
}
