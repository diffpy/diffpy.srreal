/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2011 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Bindings to the OverlapCalculator class.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>
#include <diffpy/srreal/OverlapCalculator.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_OverlapCalculator {

using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_OverlapCalculator = "\
";

const char* doc_OverlapCalculator_overlaps = "\
";

const char* doc_OverlapCalculator_distances = "\
";

const char* doc_OverlapCalculator_directions = "\
";

const char* doc_OverlapCalculator_sites0 = "\
";

const char* doc_OverlapCalculator_sites1 = "\
";

const char* doc_OverlapCalculator_types0 = "\
";

const char* doc_OverlapCalculator_types1 = "\
";

const char* doc_OverlapCalculator_sitesquareoverlaps = "\
";

const char* doc_OverlapCalculator_totalsquareoverlap = "\
";

const char* doc_OverlapCalculator_meansquareoverlap = "\
";

const char* doc_OverlapCalculator_flipDiffTotal = "\
";

const char* doc_OverlapCalculator_flipDiffMean = "\
";

const char* doc_OverlapCalculator_gradients = "\
";

const char* doc_OverlapCalculator_atomradiitable = "\
AtomRadiiTable object used for radius lookup.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYARRAY_METHOD_WRAPPER(overlaps, overlaps_asarray)
DECLARE_PYLISTARRAY_METHOD_WRAPPER(siteSquareOverlaps, siteSquareOverlaps_asarray)
DECLARE_PYLISTARRAY_METHOD_WRAPPER(gradients, gradients_aslist)

AtomRadiiTablePtr getatomradiitable(OverlapCalculator& obj)
{
    return obj.getAtomRadiiTable();
}

void setatomradiitable(OverlapCalculator& obj, AtomRadiiTablePtr rtb)
{
    obj.setAtomRadiiTable(rtb);
}

}   // namespace nswrap_OverlapCalculator

// Wrapper definition --------------------------------------------------------

void wrap_OverlapCalculator()
{
    using namespace nswrap_OverlapCalculator;

    class_<OverlapCalculator, bases<PairQuantity> >("OverlapCalculator",
            doc_OverlapCalculator)
        .add_property("overlaps",
                overlaps_asarray<OverlapCalculator>,
                doc_OverlapCalculator_overlaps)
        .add_property("distances",
                distances_asarray<OverlapCalculator>,
                doc_OverlapCalculator_distances)
        .add_property("directions",
                directions_aslist<OverlapCalculator>,
                doc_OverlapCalculator_directions)
        .add_property("sites0",
                sites0_aslist<OverlapCalculator>,
                doc_OverlapCalculator_sites0)
        .add_property("sites1",
                sites1_aslist<OverlapCalculator>,
                doc_OverlapCalculator_sites1)
        .add_property("types0",
                types0_aslist<OverlapCalculator>,
                doc_OverlapCalculator_types0)
        .add_property("types1",
                types1_aslist<OverlapCalculator>,
                doc_OverlapCalculator_types1)
        .add_property("sitesquareoverlaps",
                siteSquareOverlaps_asarray<OverlapCalculator>,
                doc_OverlapCalculator_sitesquareoverlaps)
        .add_property("totalsquareoverlap",
                &OverlapCalculator::totalSquareOverlap,
                doc_OverlapCalculator_totalsquareoverlap)
        .add_property("meansquareoverlap",
                &OverlapCalculator::meanSquareOverlap,
                doc_OverlapCalculator_meansquareoverlap)
        .def("flipDiffTotal",
                &OverlapCalculator::flipDiffTotal,
                doc_OverlapCalculator_flipDiffTotal)
        .def("flipDiffMean",
                &OverlapCalculator::flipDiffMean,
                doc_OverlapCalculator_flipDiffMean)
        .add_property("gradients",
                gradients_aslist<OverlapCalculator>,
                doc_OverlapCalculator_gradients)
        .add_property("atomradiitable",
                getatomradiitable, setatomradiitable,
                doc_OverlapCalculator_atomradiitable)
        .def_pickle(SerializationPickleSuite<OverlapCalculator>())
        ;

    // Inject __init__ and __call__ methods with support for keyword arguments.
    // Add properties for the OverlapCalculator double attributes.
    import("diffpy.srreal.overlapcalculator");
}

}   // namespace srrealmodule

// End of file
