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
#include "srreal_docstrings.hpp"

namespace srrealmodule {
namespace nswrap_BVSCalculator {

using namespace boost::python;
using namespace diffpy::srreal;

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
        ;
}

}   // namespace srrealmodule

// End of file
