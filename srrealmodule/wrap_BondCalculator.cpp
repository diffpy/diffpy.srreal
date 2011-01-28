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
* Bindings to the BondCalculator class.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>
#include <diffpy/srreal/BondCalculator.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_BondCalculator {

namespace bp = boost::python;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_BondCalculator = "\
FIXME\\n\
";

const char* doc_BondCalculator_rmin = "\
Lower bound for the atom distances.\n\
[0 A]\n\
";

const char* doc_BondCalculator_rmax = "\
Upper bound for the atom distances.\n\
[5 A]\n\
";

const char* doc_BondCalculator___call__ = "\
FIXME\\n\
";

const char* doc_BondCalculator_distances = "\
FIXME\\n\
";

const char* doc_BondCalculator_directions = "\
FIXME\\n\
";

const char* doc_BondCalculator_sites0 = "\
FIXME\\n\
";

const char* doc_BondCalculator_sites1 = "\
FIXME\\n\
";

const char* doc_BondCalculator_filterCone = "\
FIXME\\n\
";

const char* doc_BondCalculator_filterOff = "\
FIXME\\n\
";

// wrappers ------------------------------------------------------------------

bp::list callop_aslist(BondCalculator& obj, const object& a)
{
    bp::iterator<QuantityType> iter;
    bp::list rv(iter(obj(a)));
    return rv;
}


bp::list directions_aslist(const BondCalculator& obj)
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


void filter_cone(BondCalculator& obj,
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

DECLARE_PYARRAY_METHOD_WRAPPER(distances, distances_asarray)
DECLARE_PYLIST_METHOD_WRAPPER(sites0, sites0_aslist)
DECLARE_PYLIST_METHOD_WRAPPER(sites1, sites1_aslist)

}   // namespace nswrap_BondCalculator

// Wrapper definition --------------------------------------------------------

void wrap_BondCalculator()
{
    using namespace nswrap_BondCalculator;

    class_<BondCalculator, bases<PairQuantity>
        >("BondCalculator", doc_BondCalculator)
        .def("__call__", callop_aslist, doc_BondCalculator___call__)
        .add_property("distances", distances_asarray<BondCalculator>,
                doc_BondCalculator_distances)
        .add_property("directions", directions_aslist,
                doc_BondCalculator_directions)
        .add_property("sites0", sites0_aslist<BondCalculator>,
                doc_BondCalculator_sites0)
        .add_property("sites1", sites1_aslist<BondCalculator>,
                doc_BondCalculator_sites1)
        .def("filterCone", filter_cone,
                doc_BondCalculator_filterCone)
        .def("filterOff", &BondCalculator::filterOff,
                doc_BondCalculator_filterOff)
        .def_pickle(SerializationPickleSuite<BondCalculator>())
        ;

    object propertyFromExtDoubleAttr =
        import("diffpy.srreal.wraputils").attr("propertyFromExtDoubleAttr");
    object bdc = scope().attr("BondCalculator");
    bdc.attr("rmin") =
        propertyFromExtDoubleAttr("rmin", doc_BondCalculator_rmin);
    bdc.attr("rmax") =
        propertyFromExtDoubleAttr("rmax", doc_BondCalculator_rmax);
}

}   // namespace srrealmodule

// End of file
