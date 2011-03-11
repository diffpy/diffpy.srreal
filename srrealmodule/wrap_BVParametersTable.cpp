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
* Bindings to the BVParametersTable and BVParam classes.
*
* $Id$
*
*****************************************************************************/

#include <string>
#include <boost/python.hpp>
#include <diffpy/srreal/BVParam.hpp>
#include <diffpy/srreal/BVParametersTable.hpp>
#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_BVParametersTable {

namespace bp = boost::python;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_BVParam = "\
";

const char* doc_BVParam___init__ = "FIXME";

const char* doc_BVParam_bondvalence = "FIXME";

const char* doc_BVParam_bondvalenceToDistance = "FIXME";

const char* doc_BVParam_setFromCifLine = "FIXME";

const char* doc_BVParam_atom0 = "FIXME";

const char* doc_BVParam_valence0 = "FIXME";

const char* doc_BVParam_atom1 = "FIXME";

const char* doc_BVParam_valence1 = "FIXME";

const char* doc_BVParam_Ro = "FIXME";

const char* doc_BVParam_B = "FIXME";

const char* doc_BVParam_ref_id = "FIXME";

// wrappers ------------------------------------------------------------------

//BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(getpwm_overloads,
//        getBVParametersTable, 0, 0)
//
//DECLARE_PYSET_FUNCTION_WRAPPER(BVParametersTable::getRegisteredTypes,
//        getBVParametersTableTypes_asset)

}   // namespace nswrap_BVParametersTable

// Wrapper definition --------------------------------------------------------

void wrap_BVParametersTable()
{
    using namespace nswrap_BVParametersTable;

    class_<BVParam>("BVParam", doc_BVParam)
        .def(init<const std::string&, int, const std::string&, int,
                double, double, std::string>(doc_BVParam___init__,
                    (arg("atom0"), arg("valence0"),
                    arg("atom1"), arg("valence1"), arg("Ro")=0.0, arg("B")=0.0,
                    arg("ref_id")="")))
        .def("bondvalence", &BVParam::bondvalence, doc_BVParam_bondvalence)
        .def("bondvalenceToDistance", &BVParam::bondvalenceToDistance,
                doc_BVParam_bondvalenceToDistance)
        .def("setFromCifLine", &BVParam::setFromCifLine,
                doc_BVParam_setFromCifLine)
        .def_readonly("atom0", &BVParam::matom0, doc_BVParam_atom0)
        .def_readonly("valence0", &BVParam::mvalence0, doc_BVParam_valence0)
        .def_readonly("atom1", &BVParam::matom1, doc_BVParam_atom1)
        .def_readonly("valence1", &BVParam::mvalence1, doc_BVParam_valence1)
        .def_readwrite("Ro", &BVParam::mRo, doc_BVParam_Ro)
        .def_readwrite("B", &BVParam::mB, doc_BVParam_B)
        .def_readwrite("ref_id", &BVParam::mref_id, doc_BVParam_ref_id)
        ;

}

}   // namespace srrealmodule

// End of file
