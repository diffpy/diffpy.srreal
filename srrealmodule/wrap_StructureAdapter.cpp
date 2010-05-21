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
* Bindings to the StructureAdapter class.  So far the wrapper is intended
* only for accessing the C++ created StructureAdapter instances and there
* is no support for method overrides from Python.
*
* $Id$
*
*****************************************************************************/

#include <boost/python.hpp>

#include <diffpy/srreal/PythonStructureAdapter.hpp>
#include <diffpy/srreal/NoSymmetryStructureAdapter.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_StructureAdapter {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// wrappers ------------------------------------------------------------------

DECLARE_PYARRAY_METHOD_WRAPPER1(siteCartesianPosition,
        siteCartesianPosition_asarray)
DECLARE_PYARRAY_METHOD_WRAPPER1(siteCartesianUij,
        siteCartesianUij_asarray)

// Helper class necessary for wrapping a pure virtual methods

class StructureAdapterWrap :
    public StructureAdapter,
    public wrapper<StructureAdapter>
{
    public:

        BaseBondGenerator* createBondGenerator() const
        {
            return this->get_override("createBondGenerator")();
        }


        int countSites() const
        {
            return this->get_override("countSites")();
        }


        const R3::Vector& siteCartesianPosition(int idx) const
        {
            static R3::Vector rv;
            python::object pos =
                this->get_override("siteCartesianPosition")(idx);
            for (int i = 0; i < R3::Ndim; ++i)
            {
                rv[i] = python::extract<double>(pos[i]);
            }
            return rv;
        }


        bool siteAnisotropy(int idx) const
        {
            return this->get_override("siteAnisotropy")(idx);
        }


        const R3::Matrix& siteCartesianUij(int idx) const
        {
            static R3::Matrix rv;
            python::object uij =
                this->get_override("siteCartesianUij")(idx);
            double* pdata = rv.data();
            int mxlen = R3::Ndim * R3::Ndim;
            for (int i = 0; i < mxlen; ++i, ++pdata)
            {
                *pdata = python::extract<double>(uij[i]);
            }
            return rv;
        }

};  // class StructureAdapterWrap

}   // namespace nswrap_StructureAdapter

// Wrapper definition --------------------------------------------------------

void wrap_StructureAdapter()
{
    using namespace nswrap_StructureAdapter;

    class_<StructureAdapterWrap, noncopyable>("StructureAdapter")
        .def("createBondGenerator",
                &StructureAdapter::createBondGenerator,
                return_value_policy<manage_new_object>())
        .def("countSites", &StructureAdapter::countSites)
        .def("totalOccupancy", &StructureAdapter::totalOccupancy)
        .def("numberDensity", &StructureAdapter::numberDensity)
        .def("siteAtomType", &StructureAdapter::siteAtomType,
                return_value_policy<copy_const_reference>())
        .def("siteCartesianPosition",
                siteCartesianPosition_asarray<StructureAdapter,int>)
        .def("siteMultiplicity", &StructureAdapter::siteMultiplicity)
        .def("siteOccupancy", &StructureAdapter::siteOccupancy)
        .def("siteAnisotropy", &StructureAdapter::siteAnisotropy)
        .def("siteCartesianUij",
                siteCartesianUij_asarray<StructureAdapter,int>)
        ;

    register_ptr_to_python<StructureAdapterPtr>();
    
    def("nosymmetry", nosymmetry<object>);
}

}   // namespace srrealmodule

// End of file
