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
* Bindings to the BondDistanceCalculator class.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>
#include <diffpy/srreal/BondDistanceCalculator.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_BondDistanceCalculator {

namespace bp = boost::python;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_BondDistanceCalculator = "\
FIXME\\n\
";

const char* doc_BondDistanceCalculator_rmin = "\
Lower bound for the atom distances.\n\
[0 A]\n\
";

const char* doc_BondDistanceCalculator_rmax = "\
Upper bound for the atom distances.\n\
[5 A]\n\
";

const char* doc_BondDistanceCalculator___call__ = "\
FIXME\\n\
";

const char* doc_BondDistanceCalculator_distances = "\
FIXME\\n\
";

const char* doc_BondDistanceCalculator_directions = "\
FIXME\\n\
";

const char* doc_BondDistanceCalculator_sites0 = "\
FIXME\\n\
";

const char* doc_BondDistanceCalculator_sites1 = "\
FIXME\\n\
";

const char* doc_BondDistanceCalculator_filterCone = "\
FIXME\\n\
";

const char* doc_BondDistanceCalculator_filterOff = "\
FIXME\\n\
";

// wrappers ------------------------------------------------------------------

bp::list callop_aslist(BondDistanceCalculator& obj, const object& a)
{
    bp::iterator<QuantityType> iter;
    bp::list rv(iter(obj(a)));
    return rv;
}


bp::list directions_aslist(const BondDistanceCalculator& obj)
{
    std::vector<R3::Vector> rv = obj.directions();
    std::vector<R3::Vector>::const_iterator ii;
    bp::list rvlist;
    for (ii = rv.begin(); ii != rv.end(); ++ii)
    {
        rvlist.append(convertToNumPyArray(*ii));
    }
    return rvlist;
}


void filter_cone(BondDistanceCalculator& obj,
        bp::object cartesiandir, double degrees)
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

DECLARE_PYLIST_METHOD_WRAPPER(distances, distances_aslist)
DECLARE_PYLIST_METHOD_WRAPPER(sites0, sites0_aslist)
DECLARE_PYLIST_METHOD_WRAPPER(sites1, sites1_aslist)

}   // namespace nswrap_BondDistanceCalculator

// Wrapper definition --------------------------------------------------------

void wrap_BondDistanceCalculator()
{
    using namespace nswrap_BondDistanceCalculator;

    class_<BondDistanceCalculator, bases<PairQuantity>
        >("BondDistanceCalculator", doc_BondDistanceCalculator)
        .def("__call__", callop_aslist, doc_BondDistanceCalculator___call__)
        .def("distances", distances_aslist<BondDistanceCalculator>,
                doc_BondDistanceCalculator_distances)
        .def("directions", directions_aslist,
                doc_BondDistanceCalculator_directions)
        .def("sites0", sites0_aslist<BondDistanceCalculator>,
                doc_BondDistanceCalculator_sites0)
        .def("sites1", sites1_aslist<BondDistanceCalculator>,
                doc_BondDistanceCalculator_sites1)
        .def("filterCone", filter_cone,
                doc_BondDistanceCalculator_filterCone)
        .def("filterOff", &BondDistanceCalculator::filterOff,
                doc_BondDistanceCalculator_filterOff)
        .def_pickle(SerializationPickleSuite<BondDistanceCalculator>())
        ;

    object propertyFromExtDoubleAttr =
        import("diffpy.srreal.wraputils").attr("propertyFromExtDoubleAttr");
    object bdc = scope().attr("BondDistanceCalculator");
    bdc.attr("rmin") =
        propertyFromExtDoubleAttr("rmin", doc_BondDistanceCalculator_rmin);
    bdc.attr("rmax") =
        propertyFromExtDoubleAttr("rmax", doc_BondDistanceCalculator_rmax);
}

}   // namespace srrealmodule

// End of file
