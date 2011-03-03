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

#include <sstream>
#include <boost/python.hpp>

#include <diffpy/serialization.hpp>
#include <diffpy/srreal/PythonStructureAdapter.hpp>
#include <diffpy/srreal/NoMetaStructureAdapter.hpp>
#include <diffpy/srreal/NoSymmetryStructureAdapter.hpp>

#include "srreal_converters.hpp"

namespace srrealmodule {
namespace nswrap_StructureAdapter {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

// FIXME:
const char* doc_StructureAdapter = "";
const char* doc_StructureAdapter___init__ = "";
const char* doc_StructureAdapter_createBondGenerator = "";
const char* doc_StructureAdapter_countSites = "";
const char* doc_StructureAdapter_totalOccupancy = "";
const char* doc_StructureAdapter_numberDensity = "";
const char* doc_StructureAdapter_siteAtomType = "";
const char* doc_StructureAdapter_siteCartesianPosition = "";
const char* doc_StructureAdapter_siteMultiplicity = "";
const char* doc_StructureAdapter_siteOccupancy = "";
const char* doc_StructureAdapter_siteAnisotropy = "";
const char* doc_StructureAdapter_siteCartesianUij = "";
const char* doc_StructureAdapter__customPQConfig = "";
const char* doc_nometa = "";
const char* doc_nosymmetry = "";
const char* doc_createStructureAdapter = "";

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
            for (int i = 0; i < R3::Ndim; ++i)
            {
                for (int j = 0; j < R3::Ndim; ++j)
                {
                    rv(i, j) = python::extract<double>(uij[i][j]);
                }
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

// pickle support

StructureAdapterPtr
createStructureAdapterFromString(const std::string& content)
{
    using namespace std;
    istringstream storage(content, ios::binary);
    diffpy::serialization::iarchive ia(storage, ios::binary);
    StructureAdapterPtr adpt;
    ia >> adpt;
    return adpt;
}


class StructureAdapterPickleSuite : public pickle_suite
{
    public:

        static python::tuple getinitargs(StructureAdapterPtr adpt)
        {
            using namespace std;
            ostringstream storage(ios::binary);
            diffpy::serialization::oarchive oa(storage, ios::binary);
            oa << adpt;
            return python::make_tuple(storage.str());
        }

};  // class StructureAdapterPickleSuite

}   // namespace nswrap_StructureAdapter

// Wrapper definition --------------------------------------------------------

void wrap_StructureAdapter()
{
    using namespace nswrap_StructureAdapter;

    class_<StructureAdapterWrap, noncopyable>(
            "StructureAdapter", doc_StructureAdapter)
        .def("__init__", make_constructor(createStructureAdapterFromString),
                doc_StructureAdapter___init__)
        .def("createBondGenerator",
                &StructureAdapter::createBondGenerator,
                return_internal_reference<>(),
                doc_StructureAdapter_createBondGenerator)
        .def("countSites", &StructureAdapter::countSites,
                doc_StructureAdapter_countSites)
        .def("totalOccupancy",
                &StructureAdapter::totalOccupancy,
                &StructureAdapterWrap::default_totalOccupancy,
                doc_StructureAdapter_totalOccupancy)
        .def("numberDensity", &StructureAdapter::numberDensity,
                &StructureAdapterWrap::default_numberDensity,
                doc_StructureAdapter_numberDensity)
        .def("siteAtomType", &StructureAdapter::siteAtomType,
                &StructureAdapterWrap::default_siteAtomType,
                return_value_policy<copy_const_reference>(),
                doc_StructureAdapter_siteAtomType)
        .def("siteCartesianPosition",
                    siteCartesianPosition_asarray<StructureAdapter,int>,
                    doc_StructureAdapter_siteCartesianPosition)
        .def("siteMultiplicity",
                &StructureAdapter::siteMultiplicity,
                &StructureAdapterWrap::default_siteMultiplicity,
                doc_StructureAdapter_siteMultiplicity)
        .def("siteOccupancy",
                &StructureAdapter::siteOccupancy,
                &StructureAdapterWrap::default_siteOccupancy,
                doc_StructureAdapter_siteOccupancy)
        .def("siteAnisotropy",
                &StructureAdapter::siteAnisotropy,
                doc_StructureAdapter_siteAnisotropy)
        .def("siteCartesianUij",
                siteCartesianUij_asarray<StructureAdapter,int>,
                doc_StructureAdapter_siteCartesianUij)
        .def("_customPQConfig",
                &StructureAdapter::customPQConfig,
                &StructureAdapterWrap::default_customPQConfig,
                doc_StructureAdapter__customPQConfig)
        .def_pickle(StructureAdapterPickleSuite())
        ;

    register_ptr_to_python<StructureAdapterPtr>();

    def("nometa", nometa<object>, doc_nometa);
    def("nosymmetry", nosymmetry<object>, doc_nosymmetry);
    def("createStructureAdapter", createStructureAdapter,
            doc_createStructureAdapter);
}

}   // namespace srrealmodule

// End of file
