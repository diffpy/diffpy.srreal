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
FIXME\n\
";

const char* doc_OverlapCalculator_overlaps = "\
FIXME\n\
";

const char* doc_OverlapCalculator_distances = "\
FIXME\n\
";

const char* doc_OverlapCalculator_directions = "\
FIXME\n\
";

const char* doc_OverlapCalculator_sites0 = "\
FIXME\n\
";

const char* doc_OverlapCalculator_sites1 = "\
FIXME\n\
";

const char* doc_OverlapCalculator_types0 = "\
FIXME\n\
";

const char* doc_OverlapCalculator_types1 = "\
FIXME\n\
";

const char* doc_OverlapCalculator_sitesquareoverlaps = "\
FIXME\n\
";

const char* doc_OverlapCalculator_totalsquareoverlap = "\
FIXME\n\
";

const char* doc_OverlapCalculator_meansquareoverlap = "\
FIXME\n\
";

const char* doc_OverlapCalculator_flipDiffTotal = "\
FIXME\n\
";

const char* doc_OverlapCalculator_flipDiffMean = "\
FIXME\n\
";

const char* doc_OverlapCalculator_gradients = "\
FIXME\n\
";

const char* doc_OverlapCalculator_getNeighborSites = "\
FIXME\n\
";

const char* doc_OverlapCalculator_coordinations = "\
FIXME\n\
";

const char* doc_OverlapCalculator_coordinationByTypes = "\
FIXME\n\
";

const char* doc_OverlapCalculator_neighborhoods = "\
FIXME\n\
";

const char* doc_OverlapCalculator_atomradiitable = "\
AtomRadiiTable object used for radius lookup.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYARRAY_METHOD_WRAPPER(overlaps, overlaps_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(siteSquareOverlaps, siteSquareOverlaps_asarray)
DECLARE_PYLISTARRAY_METHOD_WRAPPER(gradients, gradients_aslist)
DECLARE_PYSET_METHOD_WRAPPER1(getNeighborSites, getNeighborSites_asset)
DECLARE_PYARRAY_METHOD_WRAPPER(coordinations, coordinations_asarray)
DECLARE_PYDICT_METHOD_WRAPPER1(coordinationByTypes, coordinationByTypes_asdict)
DECLARE_PYLISTSET_METHOD_WRAPPER(neighborhoods, neighborhoods_aslistset)

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
        .def("getNeighborSites",
                getNeighborSites_asset<OverlapCalculator,int>,
                doc_OverlapCalculator_getNeighborSites)
        .add_property("coordinations",
                coordinations_asarray<OverlapCalculator>,
                doc_OverlapCalculator_coordinations)
        .def("coordinationByTypes",
                coordinationByTypes_asdict<OverlapCalculator,int>,
                doc_OverlapCalculator_coordinationByTypes)
        .add_property("neighborhoods",
                neighborhoods_aslistset<OverlapCalculator>,
                doc_OverlapCalculator_neighborhoods)
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
