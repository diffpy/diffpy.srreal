/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Bindings to the BaseBondGenerator class.  So far the wrapper is intended
* only for accessing the C++ created BaseBondGenerator instances and there
* is no support for method overrides from Python.
*
* $Id$
*
*****************************************************************************/

#include <string>
#include <boost/python.hpp>

#include <diffpy/srreal/BaseBondGenerator.hpp>
#include <diffpy/srreal/StructureAdapter.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_BaseBondGenerator {

// docstrings ----------------------------------------------------------------

const char* doc_BaseBondGenerator = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_rewind = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_finished = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_next = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_nextsite = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_selectAnchorSite = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_selectSiteRange = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_setRmin = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_setRmax = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_getRmin = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_getRmax = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_site0 = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_site1 = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_multiplicity = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_r0 = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_r1 = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_distance = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_r01 = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_Ucartesian0 = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_Ucartesian1 = "\
FIXME\n\
";

const char* doc_BaseBondGenerator_msd = "\
FIXME\n\
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
            init<StructureAdapterConstPtr>())
        .def("rewind", &BaseBondGenerator::rewind,
                doc_BaseBondGenerator_rewind)
        .def("finished", &BaseBondGenerator::finished,
                doc_BaseBondGenerator_finished)
        .def("next", &BaseBondGenerator::next,
                doc_BaseBondGenerator_next)
        .def("nextsite", &BaseBondGenerator::nextsite,
                doc_BaseBondGenerator_nextsite)
        .def("selectAnchorSite", &BaseBondGenerator::selectAnchorSite,
                doc_BaseBondGenerator_selectAnchorSite)
        .def("selectSiteRange", &BaseBondGenerator::selectSiteRange,
                doc_BaseBondGenerator_selectSiteRange)
        .def("setRmin", &BaseBondGenerator::setRmin,
                doc_BaseBondGenerator_setRmin)
        .def("setRmax", &BaseBondGenerator::setRmax,
                doc_BaseBondGenerator_setRmax)
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
