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


DECLARE_PYLIST_METHOD_WRAPPER(distances, distances_aslist)
DECLARE_PYLIST_METHOD_WRAPPER(sites0, sites0_aslist)
DECLARE_PYLIST_METHOD_WRAPPER(sites1, sites1_aslist)

}   // namespace nswrap_BondDistanceCalculator

// Wrapper definition --------------------------------------------------------

void wrap_BondDistanceCalculator()
{
    using namespace nswrap_BondDistanceCalculator;

    class_<BondDistanceCalculator, bases<PairQuantity>
        >("BondDistanceCalculator")
        .def("__call__", callop_aslist)
        .def("distances", distances_aslist<BondDistanceCalculator>)
        .def("directions", directions_aslist)
        .def("sites0", sites0_aslist<BondDistanceCalculator>)
        .def("sites1", sites1_aslist<BondDistanceCalculator>)
        .def_pickle(SerializationPickleSuite<BondDistanceCalculator>())
        ;

}

}   // namespace srrealmodule

// End of file
