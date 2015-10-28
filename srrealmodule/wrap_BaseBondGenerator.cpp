/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* Bindings to the BaseBondGenerator class.  So far the wrapper is intended
* only for accessing the C++ created BaseBondGenerator instances and there
* is no support for method overrides from Python.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/register_ptr_to_python.hpp>

#include <string>

#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_BaseBondGenerator {

// docstrings ----------------------------------------------------------------

const char* doc_BaseBondGenerator = "\
This class generates atom pairs in the structure within specified distance\n\
bounds and returns the associated atom-pair data.  The generator walks over\n\
all neighbors of the anchor site accounting for symmetry and periodicity of\n\
the structure object.\n\
";

const char* doc_BaseBondGenerator_rewind = "\
Reset generator to the first atom pair containing the anchor atom.\n\
";

const char* doc_BaseBondGenerator_finished = "\
Return True if there are no more atom pairs containing the anchor atom.\n\
";

const char* doc_BaseBondGenerator_next = "\
Advance to the next bond of the anchor atom.\n\
";

const char* doc_BaseBondGenerator_selectAnchorSite = "\
Select the anchor site of the bond generator.\n\
\n\
anchor   -- integer index of the anchor site\n\
\n\
No return value.  Must be followed by a rewind call to get a valid state.\n\
";

const char* doc_BaseBondGenerator_selectSiteRange = "\
Select a range of neighbor sites for the anchor atom bonds.\n\
\n\
first    -- integer index of the first neighbor site\n\
last     -- index of the last neighbor site, not included\n\
\n\
No return value.  Must be followed by a rewind call to get a valid state.\n\
";

const char* doc_BaseBondGenerator_setRmin = "\
Select minimum distance for the generated bonds.\n\
\n\
rmin -- the minimum distance\n\
\n\
No return value.  Must be followed by a rewind call to get a valid state.\n\
";

const char* doc_BaseBondGenerator_setRmax = "\
Select the maximum distance limit of the generated bonds.\n\
\n\
rmax -- the maximum distance\n\
\n\
No return value.  Must be followed by a rewind call to get a valid state.\n\
";

const char* doc_BaseBondGenerator_getRmin = "\
Return the minimum length of the generated bonds.\n\
";

const char* doc_BaseBondGenerator_getRmax = "\
Return the maximum length of the generated bonds.\n\
";

const char* doc_BaseBondGenerator_site0 = "\
Anchor site index of the current bond.\n\
";

const char* doc_BaseBondGenerator_site1 = "\
Neighbor site index of the current bond.\n\
";

const char* doc_BaseBondGenerator_multiplicity = "\
Symmetry multiplicity of the anchor site.\n\
";

const char* doc_BaseBondGenerator_r0 = "\
Cartesian coordinates of the anchor site.\n\
";

const char* doc_BaseBondGenerator_r1 = "\
Cartesian coordinates of the current neighbor site, these can\n\
be adjusted for symmetry operations and periodic translations.\n\
";

const char* doc_BaseBondGenerator_distance = "\
Current distance between anchor and neighbor site.\n\
";

const char* doc_BaseBondGenerator_r01 = "\
Cartesian direction from anchor to neighbor site.\n\
";

const char* doc_BaseBondGenerator_Ucartesian0 = "\
Cartesian anisotropic displacement parameters matrix at the anchor site.\n\
Returns a 3x3 numpy array.\n\
";

const char* doc_BaseBondGenerator_Ucartesian1 = "\
Cartesian anisotropic displacement parameters matrix at the neighbor site.\n\
This can be adjusted for symmetry rotations at the neighbor site.\n\
Returns a 3x3 numpy array.\n\
";

const char* doc_BaseBondGenerator_msd = "\
Mean square displacement along the direction from anchor to neighbor site.\n\
This is proportional to the sum of Ucartesian0 and Ucartesian1 matrices.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYARRAY_METHOD_WRAPPER(r0, r0_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(r1, r1_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(r01, r01_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(Ucartesian0, Ucartesian0_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(Ucartesian1, Ucartesian1_asarray)

}   // namespace nswrap_BaseBondGenerator

// Wrapper definition --------------------------------------------------------

void wrap_BaseBondGenerator()
{
    using namespace boost::python;
    using namespace diffpy::srreal;
    using namespace nswrap_BaseBondGenerator;

    class_<BaseBondGenerator>("BaseBondGenerator", doc_BaseBondGenerator,
            init<StructureAdapterPtr>())
        .def("rewind", &BaseBondGenerator::rewind,
                doc_BaseBondGenerator_rewind)
        .def("finished", &BaseBondGenerator::finished,
                doc_BaseBondGenerator_finished)
        .def("next", &BaseBondGenerator::next,
                doc_BaseBondGenerator_next)
        .def("selectAnchorSite", &BaseBondGenerator::selectAnchorSite,
                arg("anchor"), doc_BaseBondGenerator_selectAnchorSite)
        .def("selectSiteRange", &BaseBondGenerator::selectSiteRange,
                (arg("first"), arg("last")),
                doc_BaseBondGenerator_selectSiteRange)
        .def("setRmin", &BaseBondGenerator::setRmin,
                arg("rmin"), doc_BaseBondGenerator_setRmin)
        .def("setRmax", &BaseBondGenerator::setRmax,
                arg("rmax"), doc_BaseBondGenerator_setRmax)
        .def("getRmin", &BaseBondGenerator::getRmin,
                return_value_policy<copy_const_reference>(),
                doc_BaseBondGenerator_getRmin)
        .def("getRmax", &BaseBondGenerator::getRmax,
                return_value_policy<copy_const_reference>(),
                doc_BaseBondGenerator_getRmax)
        .def("site0", &BaseBondGenerator::site0,
                doc_BaseBondGenerator_site0)
        .def("site1", &BaseBondGenerator::site1,
                doc_BaseBondGenerator_site1)
        .def("multiplicity", &BaseBondGenerator::multiplicity,
                doc_BaseBondGenerator_multiplicity)
        .def("r0", r0_asarray<BaseBondGenerator>,
                doc_BaseBondGenerator_r0)
        .def("r1", r1_asarray<BaseBondGenerator>,
                doc_BaseBondGenerator_r1)
        .def("distance", &BaseBondGenerator::distance,
                return_value_policy<copy_const_reference>(),
                doc_BaseBondGenerator_distance)
        .def("r01", r01_asarray<BaseBondGenerator>,
                doc_BaseBondGenerator_r01)
        .def("Ucartesian0", Ucartesian0_asarray<BaseBondGenerator>,
                doc_BaseBondGenerator_Ucartesian0)
        .def("Ucartesian1", Ucartesian1_asarray<BaseBondGenerator>,
                doc_BaseBondGenerator_Ucartesian1)
        .def("msd", &BaseBondGenerator::msd,
                doc_BaseBondGenerator_msd)
        ;

    register_ptr_to_python<BaseBondGeneratorPtr>();
}

}   // namespace srrealmodule

// End of file
