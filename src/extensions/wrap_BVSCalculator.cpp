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
* Bindings to the BVSCalculator class.
*
*****************************************************************************/

#include <boost/python/class.hpp>

#include <diffpy/srreal/BVSCalculator.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_BVSCalculator {

using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_BVSCalculator = "\
Calculator of bond valence sums in the specified structure.\n\
";

const char* doc_BVSCalculator_value = "\
Return bond valence sums per each atom site in the structure.\n\
";

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

const char* doc_BVSCalculator_bvparamtable = "\
BVParametersTable object used for bond valence parameters lookup.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYARRAY_METHOD_WRAPPER(valences, valences_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(bvdiff, bvdiff_asarray)

BVParametersTablePtr getbvparamtable(BVSCalculator& obj)
{
    return obj.getBVParamTable();
}

void setbvparamtable(BVSCalculator& obj, BVParametersTablePtr bptb)
{
    obj.setBVParamTable(bptb);
}

}   // namespace nswrap_BVSCalculator

// Wrapper definition --------------------------------------------------------

void wrap_BVSCalculator()
{
    using namespace nswrap_BVSCalculator;

    class_<BVSCalculator, bases<PairQuantity> >("BVSCalculator",
            doc_BVSCalculator)
        .add_property("value", value_asarray<BVSCalculator>,
                doc_BVSCalculator_value)
        .add_property("valences", valences_asarray<BVSCalculator>,
                doc_BVSCalculator_valences)
        .add_property("bvdiff", bvdiff_asarray<BVSCalculator>,
                doc_BVSCalculator_bvdiff)
        .add_property("bvmsdiff", &BVSCalculator::bvmsdiff,
                doc_BVSCalculator_bvmsdiff)
        .add_property("bvrmsdiff", &BVSCalculator::bvrmsdiff,
                doc_BVSCalculator_bvrmsdiff)
        .add_property("bvparamtable", getbvparamtable, setbvparamtable,
                doc_BVSCalculator_bvparamtable)
        .def_pickle(PairQuantityPickleSuite<BVSCalculator>())
        ;

}

}   // namespace srrealmodule

// End of file
