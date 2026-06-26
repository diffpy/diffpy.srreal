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
* Bindings to the OverlapCalculator class.
*
*****************************************************************************/

#include <nanobind/nanobind.h>

#include <diffpy/srreal/ConstantRadiiTable.hpp>
#include <diffpy/srreal/OverlapCalculator.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace nb = nanobind;

namespace srrealmodule {
namespace nswrap_OverlapCalculator {

using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_OverlapCalculator = "\
Calculate the overlap of atom radii and coordination numbers at each site.\n\
";

const char* doc_OverlapCalculator_overlaps = "\
Magnitudes of all non-zero atom radii overlaps in the structure.\n\
";

const char* doc_OverlapCalculator_distances = "\
Distances of all overlapping atom pairs in the structure.\n\
";

const char* doc_OverlapCalculator_directions = "\
Directions from the first to the second atom in the overlapping pairs.\n\
Returns an Nx3 array.\n\
";

const char* doc_OverlapCalculator_sites0 = "\
Indices of all first sites of the overlapping pairs.\n\
";

const char* doc_OverlapCalculator_sites1 = "\
Indices of all second sites of the overlapping pairs.\n\
";

const char* doc_OverlapCalculator_types0 = "\
List of atom symbols of all first sites of the overlapping pairs.\n\
";

const char* doc_OverlapCalculator_types1 = "\
List of atom symbols of all second sites of the overlapping pairs.\n\
";

const char* doc_OverlapCalculator_sitesquareoverlaps = "\
Sum of squared overlaps per each site in the structure.\n\
";

const char* doc_OverlapCalculator_totalsquareoverlap = "\
Total sum of squared overlaps in the structure adjusted\n\
for site multiplicities and occupancies.\n\
";

const char* doc_OverlapCalculator_meansquareoverlap = "\
Total square overlap per one atom in the structure.\n\
";

const char* doc_OverlapCalculator_flipDiffTotal = "\
Calculate change of the totalsquareoverlap after flipping the atom types\n\
at the i and j sites.\n\
\n\
i    -- zero-based index of the first site\n\
j    -- zero-based index of the second site\n\
\n\
Return float.\n\
";

const char* doc_OverlapCalculator_flipDiffMean = "\
Calculate change of the meansquareoverlap after flipping the atom types\n\
at the i and j sites.\n\
\n\
i    -- zero-based index of the first site\n\
j    -- zero-based index of the second site\n\
\n\
Return float.\n\
";

const char* doc_OverlapCalculator_gradients = "\
Gradients of the totalsquareoverlap per each site in the structure.\n\
Returns an Nx3 array.\n\
";

const char* doc_OverlapCalculator_getNeighborSites = "\
Get indices of all sites that neighbor with the specified site,\n\
either directly or as periodic or symmetry images.  Atoms are\n\
assumed neighbors if they have non-zero overlap.\n\
\n\
i    -- zero-based index of the evaluated site\n\
\n\
Return a set of integer indices, this may include i.\n\
";

const char* doc_OverlapCalculator_coordinations = "\
Return coordination numbers per each site in the structure.\n\
These may be non-integer if there are sites with partial occupancies.\n\
";

const char* doc_OverlapCalculator_coordinationByTypes = "\
Evaluate neighbor types and their occupancies at the specified site.\n\
\n\
i    -- zero-based index of the evaluated site\n\
\n\
Return a dictionary where the keys are atom type strings and the values\n\
their total occupancies.\n\
";

const char* doc_OverlapCalculator_neighborhoods = "\
Return all sets of connected site indices in the structure.\n\
Sites are assumed in the same neighborhood if they have overlapping\n\
neighbors or can be connected through neighbor links.  There are no\n\
overlapping pairs between sites from different neighborhoods.\n\
\n\
Return a list of site indices sets.\n\
";

const char* doc_OverlapCalculator_atomradiitable = "\
AtomRadiiTable object used for radius lookup.\n\
";

// wrappers ------------------------------------------------------------------

DECLARE_PYARRAY_METHOD_WRAPPER(overlaps, overlaps_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(siteSquareOverlaps, siteSquareOverlaps_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(gradients, gradients_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER(coordinations, coordinations_asarray)
DECLARE_PYDICT_METHOD_WRAPPER1(coordinationByTypes, coordinationByTypes_asdict)
DECLARE_PYLISTSET_METHOD_WRAPPER(neighborhoods, neighborhoods_aslistset)

AtomRadiiTablePtr getatomradiitable(OverlapCalculator& obj)
{
    return obj.getAtomRadiiTable();
}

DECLARE_BYTYPE_SETTER_WRAPPER(setAtomRadiiTable, setatomradiitable)


double flip_diff_total(const OverlapCalculator& obj, nb::object i, nb::object j)
{
    int i1 = extractint(i);
    int j1 = extractint(j);
    return obj.flipDiffTotal(i1, j1);
}


double flip_diff_mean(const OverlapCalculator& obj, nb::object i, nb::object j)
{
    int i1 = extractint(i);
    int j1 = extractint(j);
    return obj.flipDiffMean(i1, j1);
}


nb::object get_neighbor_sites(const OverlapCalculator& obj, nb::object i)
{
    int i1 = extractint(i);
    nb::object rv = convertToPythonSet(obj.getNeighborSites(i1));
    return rv;
}


class OverlapCalculatorPickleSuite :
    public PairQuantityPickleSuite<OverlapCalculator>
{
    private:

        typedef PairQuantityPickleSuite<OverlapCalculator> Super;

    public:

        template <typename C>
        static void bind(C& cls)
        {
            cls
                .def("__getstate__", getstate)
                .def("__setstate__", setstate)
                .def("__reduce__", reduce)
                ;
            cls.attr("__getstate_manages_dict__") = nb::none();
        }


        static nb::tuple getstate(nb::object obj)
        {
            OverlapCalculator& calc = nb::cast<OverlapCalculator&>(obj);
            ScopedSharedPtrReplacement<AtomRadiiTable> tb(
                calc.getAtomRadiiTable(), AtomRadiiTablePtr(new ConstantRadiiTable));
            try
            {
                nb::tuple rv = nb::make_tuple(
                        Super::getstate(obj),
                        tb.state_object()
                        );
                return rv;
            }
            catch (...)
            {
                tb.restore();
                throw;
            }
        }


        static void setstate(
                nb::object obj, nb::tuple state)
        {
            ensure_tuple_length(state, 2);
            // restore the state using boost serialization
            nb::tuple state0(state[0]);
            Super::setstate(obj, state0);
            // restore component that was kept out of Boost serialization
            OverlapCalculator& calc = nb::cast<OverlapCalculator&>(obj);
            assign_pointer_state(calc.getAtomRadiiTable(), state[1]);
        }

        static nb::tuple reduce(nb::object obj)
        {
            return Super::reduce(obj);
        }
};

}   // namespace nswrap_OverlapCalculator

// Wrapper definition --------------------------------------------------------

void wrap_OverlapCalculator(nb::module_& m)
{
    using namespace nswrap_OverlapCalculator;

    nb::class_<OverlapCalculator, PairQuantity> overlapcalculator(m, "OverlapCalculator",
            doc_OverlapCalculator);
    overlapcalculator
        .def(nb::init<>())
        .def_prop_ro("overlaps",
                overlaps_asarray<OverlapCalculator>,
                doc_OverlapCalculator_overlaps)
        .def_prop_ro("distances",
                distances_asarray<OverlapCalculator>,
                doc_OverlapCalculator_distances)
        .def_prop_ro("directions",
                directions_asarray<OverlapCalculator>,
                doc_OverlapCalculator_directions)
        .def_prop_ro("sites0",
                sites0_asarray<OverlapCalculator>,
                doc_OverlapCalculator_sites0)
        .def_prop_ro("sites1",
                sites1_asarray<OverlapCalculator>,
                doc_OverlapCalculator_sites1)
        .def_prop_ro("types0",
                types0_aschararray<OverlapCalculator>,
                doc_OverlapCalculator_types0)
        .def_prop_ro("types1",
                types1_aschararray<OverlapCalculator>,
                doc_OverlapCalculator_types1)
        .def_prop_ro("sitesquareoverlaps",
                siteSquareOverlaps_asarray<OverlapCalculator>,
                doc_OverlapCalculator_sitesquareoverlaps)
        .def_prop_ro("totalsquareoverlap",
                &OverlapCalculator::totalSquareOverlap,
                doc_OverlapCalculator_totalsquareoverlap)
        .def_prop_ro("meansquareoverlap",
                &OverlapCalculator::meanSquareOverlap,
                doc_OverlapCalculator_meansquareoverlap)
        .def("flipDiffTotal",
                flip_diff_total,
                nb::arg("i"), nb::arg("j"),
                doc_OverlapCalculator_flipDiffTotal)
        .def("flipDiffMean",
                flip_diff_mean,
                doc_OverlapCalculator_flipDiffMean)
        .def_prop_ro("gradients",
                gradients_asarray<OverlapCalculator>,
                doc_OverlapCalculator_gradients)
        .def("getNeighborSites",
                get_neighbor_sites,
                nb::arg("i"),
                doc_OverlapCalculator_getNeighborSites)
        .def_prop_ro("coordinations",
                coordinations_asarray<OverlapCalculator>,
                doc_OverlapCalculator_coordinations)
        .def("coordinationByTypes",
                coordinationByTypes_asdict<OverlapCalculator,int>,
                nb::arg("i"),
                doc_OverlapCalculator_coordinationByTypes)
        .def_prop_ro("neighborhoods",
                neighborhoods_aslistset<OverlapCalculator>,
                doc_OverlapCalculator_neighborhoods)
        .def_prop_rw("atomradiitable",
                getatomradiitable,
                setatomradiitable<OverlapCalculator,AtomRadiiTable>,
                doc_OverlapCalculator_atomradiitable)
        ;
        OverlapCalculatorPickleSuite::bind(overlapcalculator);
        
}

}   // namespace srrealmodule

// End of file
