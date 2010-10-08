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
    public wrapper_srreal<StructureAdapter>
{
    public:

        BaseBondGenerator* createBondGenerator() const
        {
            return this->get_pure_virtual_override("createBondGenerator")();
        }


        int countSites() const
        {
            return this->get_pure_virtual_override("countSites")();
        }


        double totalOccupancy() const
        {
            override f = this->get_override("totalOccupancy");
            if (f)  return f();
            return this->default_totalOccupancy();
        }

        double default_totalOccupancy() const
        {
            return this->StructureAdapter::totalOccupancy();
        }


        double numberDensity() const
        {
            override f = this->get_override("numberDensity");
            if (f)  return f();
            return this->default_numberDensity();
        }

        double default_numberDensity() const
        {
            return this->StructureAdapter::numberDensity();
        }


        const std::string& siteAtomType(int idx) const
        {
            static std::string rv;
            override f = this->get_override("siteAtomType");
            if (f)
            {
                python::object atp = f(idx);
                rv = python::extract<std::string>(atp);
                return rv;
            }
            return this->default_siteAtomType(idx);
        }

        const std::string& default_siteAtomType(int idx) const
        {
            return this->StructureAdapter::siteAtomType(idx);
        }


        const R3::Vector& siteCartesianPosition(int idx) const
        {
            static R3::Vector rv;
            python::object pos =
                this->get_pure_virtual_override("siteCartesianPosition")(idx);
            for (int i = 0; i < R3::Ndim; ++i)
            {
                rv[i] = python::extract<double>(pos[i]);
            }
            return rv;
        }


        int siteMultiplicity(int idx) const
        {
            override f = this->get_override("siteMultiplicity");
            if (f)  return f(idx);
            return this->default_siteMultiplicity(idx);
        }

        int default_siteMultiplicity(int idx) const
        {
            return this->StructureAdapter::siteMultiplicity(idx);
        }


        double siteOccupancy(int idx) const
        {
            override f = this->get_override("siteOccupancy");
            if (f)  return f(idx);
            return this->default_siteOccupancy(idx);
        }

        double default_siteOccupancy(int idx) const
        {
            return this->StructureAdapter::siteOccupancy(idx);
        }


        bool siteAnisotropy(int idx) const
        {
            return this->get_pure_virtual_override("siteAnisotropy")(idx);
        }


        const R3::Matrix& siteCartesianUij(int idx) const
        {
            static R3::Matrix rv;
            python::object uij =
                this->get_pure_virtual_override("siteCartesianUij")(idx);
            double* pdata = rv.data();
            int mxlen = R3::Ndim * R3::Ndim;
            for (int i = 0; i < mxlen; ++i, ++pdata)
            {
                *pdata = python::extract<double>(uij[i]);
            }
            return rv;
        }


        void customPQConfig(PairQuantity* pq) const
        {
            override f = this->get_override("_customPQConfig");
            if (f)  f(pq);
            else    this->default_customPQConfig(pq);
        }

        void default_customPQConfig(PairQuantity* pq) const
        {
            this->StructureAdapter::customPQConfig(pq);
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
                return_internal_reference<>())
        .def("countSites", &StructureAdapter::countSites)
        .def("totalOccupancy",
                &StructureAdapter::totalOccupancy,
                &StructureAdapterWrap::default_totalOccupancy)
        .def("numberDensity", &StructureAdapter::numberDensity,
                &StructureAdapterWrap::default_numberDensity)
        .def("siteAtomType", &StructureAdapter::siteAtomType,
                &StructureAdapterWrap::default_siteAtomType,
                return_value_policy<copy_const_reference>())
        .def("siteCartesianPosition", 
                    siteCartesianPosition_asarray<StructureAdapter,int>)
        .def("siteMultiplicity",
                &StructureAdapter::siteMultiplicity,
                &StructureAdapterWrap::default_siteMultiplicity)
        .def("siteOccupancy",
                &StructureAdapter::siteOccupancy,
                &StructureAdapterWrap::default_siteOccupancy)
        .def("siteAnisotropy",
                &StructureAdapter::siteAnisotropy)
        .def("siteCartesianUij",
                siteCartesianUij_asarray<StructureAdapter,int>)
        .def("_customPQConfig",
                &StructureAdapter::customPQConfig,
                &StructureAdapterWrap::default_customPQConfig)
        ;

    register_ptr_to_python<StructureAdapterPtr>();
    
    def("nosymmetry", nosymmetry<object>);
    def("createStructureAdapter", createStructureAdapter);
}

}   // namespace srrealmodule

// End of file
