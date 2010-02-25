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
* bvs_ext - boost python wrap of the C++ bond valence sums calculators
*
* $Id$
*
*****************************************************************************/


#include <boost/python.hpp>

#include <numpy/arrayobject.h>

#include <diffpy/srreal/BVSCalculator.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>
#include "srreal_converters.hpp"

// Module Definitions --------------------------------------------------------

BOOST_PYTHON_MODULE(bvs_ext)
{
    using namespace boost::python;
    using diffpy::srreal::BVSCalculator;

    // initialize converters
    initialize_srreal_converters();

    class_<BVSCalculator>("BVSCalculator")
        .def("_getDoubleAttr", &BVSCalculator::getDoubleAttr)
        .def("_setDoubleAttr", &BVSCalculator::setDoubleAttr)
        .def("_hasDoubleAttr", &BVSCalculator::hasDoubleAttr)
        .def("_namesOfDoubleAttributes",
                &BVSCalculator::namesOfDoubleAttributes)
        .def("valences", &BVSCalculator::valences)
        .def("bvmsdiff", &BVSCalculator::bvmsdiff)
        .def("bvrmsdiff", &BVSCalculator::bvrmsdiff)
        .def("eval", &BVSCalculator::eval<object>,
                return_value_policy<copy_const_reference>())
        ;
}
