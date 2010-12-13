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
* Bindings to the BVSCalculator class.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>
#include <diffpy/srreal/BVSCalculator.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_BVSCalculator {

using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_BVSCalculator_valences = "\
Return valences expected at each site of the evaluated structure.\n\
";

const char* doc_BVSCalculator_bvdiff = "\
Difference between expected and calculated valence magnitudes at each site.\n\
Positive for underbonding, negative for overbonding.\n\
";

const char* doc_BVSCalculator_bvmsdiff = "\
Mean square difference between expected and calculated valences.\n\
Adjusted for multiplicity and occupancy of atom sites in the structure.\n\
";

const char* doc_BVSCalculator_bvrmsdiff = "\
Root mean square difference between expected and calculated valences.\n\
Adjusted for multiplicity and occupancy of atom sites in the structure.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYARRAY_METHOD_WRAPPER(valences, valences_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(bvdiff, bvdiff_asarray)

}   // namespace nswrap_BVSCalculator

// Wrapper definition --------------------------------------------------------

void wrap_BVSCalculator()
{
    using namespace nswrap_BVSCalculator;

    class_<BVSCalculator, bases<PairQuantity> >("BVSCalculator_ext")
        .def("valences", valences_asarray<BVSCalculator>,
                doc_BVSCalculator_valences)
        .def("bvdiff", bvdiff_asarray<BVSCalculator>,
                doc_BVSCalculator_bvdiff)
        .def("bvmsdiff", &BVSCalculator::bvmsdiff,
                doc_BVSCalculator_bvmsdiff)
        .def("bvrmsdiff", &BVSCalculator::bvrmsdiff,
                doc_BVSCalculator_bvrmsdiff)
        .def_pickle(SerializationPickleSuite<BVSCalculator>())
        ;

}

}   // namespace srrealmodule

// End of file
