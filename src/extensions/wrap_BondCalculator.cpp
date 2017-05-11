/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2011 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* Bindings to the BondCalculator class.
*
*****************************************************************************/

#include <boost/python/class.hpp>

#include <diffpy/srreal/BondCalculator.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_BondCalculator {

namespace bp = boost::python;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_BondCalculator = "\
Calculator of bond distances in a specified structure.\n\
";

const char* doc_BondCalculator_distances = "\
Array of sorted bond distances in the evaluated structure.\n\
";

const char* doc_BondCalculator_directions = "\
List of bond directions in the evaluated structure.\n\
";

const char* doc_BondCalculator_sites0 = "\
List of zero-based indices of the first site in the pair.\n\
";

const char* doc_BondCalculator_sites1 = "\
List of zero-based indices of the second site in the pair.\n\
";

const char* doc_BondCalculator_types0 = "\
List of atom symbols for the first site in all pairs.\n\
";

const char* doc_BondCalculator_types1 = "\
List of atom symbols for the second site in all pairs.\n\
";

const char* doc_BondCalculator_filterCone = "\
Setup an additive bond filter in a specified direction cone.\n\
Second and further calls create more filters that permit more bonds.\n\
Use filterOff to create exclusive filter in a new direction.\n\
\n\
cartesiandir -- cone axis in Cartesian coordinates,\n\
                list, tuple or array.\n\
degrees      -- cone angle in degrees\n\
\n\
No return value.\n\
";

const char* doc_BondCalculator_filterOff = "\
Turn off bond filtering and destroy all cone filters.\n\
Permit all bonds in further calculations.  Also used to create\n\
exclusive cone filter in a new direction.\n\
";

// wrappers ------------------------------------------------------------------


void filter_cone(BondCalculator& obj,
        object cartesiandir, double degrees)
{
    if (len(cartesiandir) != 3)
    {
        const char* emsg = "cartesiandir must be a 3-element array.";
        PyErr_SetString(PyExc_ValueError, emsg);
        bp::throw_error_already_set();
    }
    R3::Vector cdir;
    cdir[0] = extract<double>(cartesiandir[0]);
    cdir[1] = extract<double>(cartesiandir[1]);
    cdir[2] = extract<double>(cartesiandir[2]);
    obj.filterCone(cdir, degrees);
}

}   // namespace nswrap_BondCalculator

// Wrapper definition --------------------------------------------------------

void wrap_BondCalculator()
{
    using namespace nswrap_BondCalculator;

    class_<BondCalculator, bases<PairQuantity>
        >("BondCalculator", doc_BondCalculator)
        .add_property("distances",
                distances_asarray<BondCalculator>,
                doc_BondCalculator_distances)
        .add_property("directions",
                directions_asarray<BondCalculator>,
                doc_BondCalculator_directions)
        .add_property("sites0",
                sites0_asarray<BondCalculator>,
                doc_BondCalculator_sites0)
        .add_property("sites1",
                sites1_asarray<BondCalculator>,
                doc_BondCalculator_sites1)
        .add_property("types0",
                types0_aschararray<BondCalculator>,
                doc_BondCalculator_types0)
        .add_property("types1",
                types1_aschararray<BondCalculator>,
                doc_BondCalculator_types1)
        .def("filterCone", filter_cone,
                (bp::arg("cartesiandir"), bp::arg("degrees")),
                doc_BondCalculator_filterCone)
        .def("filterOff", &BondCalculator::filterOff,
                doc_BondCalculator_filterOff)
        .def_pickle(PairQuantityPickleSuite<BondCalculator>())
        ;

}

}   // namespace srrealmodule

// End of file
