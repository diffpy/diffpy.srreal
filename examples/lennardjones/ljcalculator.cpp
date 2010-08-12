#include <iostream>
#include <fstream>
#include <string>
#include <diffpy/srreal/PairQuantity.hpp>
#include <diffpy/srreal/VR3Structure.hpp>

using namespace diffpy::srreal;
using namespace std;

class LennardJonesCalculator : public PairQuantity
{
    public:
        // Constructor - the size of the results array is always one,
        // because LJ potential is a scalar
        LennardJonesCalculator()
        {
            this->resizeValue(1);
        }

        // convenience template method.  This allows using an instantiated
        // LennardJonesCalculator as a function.  T must be a type for which
        // a createStructureAdapter(const T&) function exists.
        template <class T>
        double operator()(const T& stru)
        {
            this->eval(stru);
            return this->value()[0];
        }

    protected:

        // define contribution from a pair of atoms.  mvalue is a reference
        // to the internal array of result values.
        void addPairContribution(const BaseBondGenerator& bnds, int sumscale)
        {
            double rij = bnds.distance();
            double ljij = 4 * (pow(rij, -12) - pow(rij, -6));
            mvalue[0] += sumscale * ljij / 2.0;
        }

};


// helper function that loads a simple list of xyz coordinates
VR3Structure loadVR3Structure(string filename)
{
    VR3Structure rv;
    ifstream fp(filename.c_str());
    double x, y, z;
    while (fp >> x >> y >> z)
    {
        rv.push_back(R3::Vector(x, y, z));
    }
    return rv;
}


int main(int argc, char* argv[])
{
    using namespace std;
    string filename;
    filename = (argc > 1) ? argv[1] : "lj50.xyz";
    LennardJonesCalculator ljcalc;
    VR3Structure stru = loadVR3Structure(filename);
    cout << "LJ potential of " << filename << " is " << ljcalc(stru) << '\n';
    return 0;
}
