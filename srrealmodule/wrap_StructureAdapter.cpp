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
*****************************************************************************/

#include <boost/python.hpp>

#include <diffpy/srreal/PythonStructureAdapter.hpp>
#include <diffpy/srreal/NoMetaStructureAdapter.hpp>
#include <diffpy/srreal/NoSymmetryStructureAdapter.hpp>
#include <diffpy/srreal/PairQuantity.hpp>
#include <diffpy/serialization.ipp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_StructureAdapter {

using namespace boost;
using namespace boost::python;
using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_StructureAdapter = "\
An adaptor to structure representation compatible with PairQuantity.\n\
This class provides a uniform interface to structure data that is\n\
understood by all PairQuantity calculators.\n\
";

const char* doc_StructureAdapter_clone = "\
Return a deep copy of this StructureAdapter instance.\n\
\n\
This method must be overloaded in a derived class.\n\
";

const char* doc_StructureAdapter_createBondGenerator = "\
Create a bond generator instance linked to this structure adapter\n\
\n\
Return a BaseBondGenerator object.\n\
";

const char* doc_StructureAdapter_countSites = "\
Return number of symmetry independent atom sites in the structure.\n\
";

const char* doc_StructureAdapter_totalOccupancy = "\
Return total atom occupancy in the structure accounting for symmetry\n\
multiplicity and fractional occupancies.\n\
";

const char* doc_StructureAdapter_numberDensity = "\
Number density of atoms in periodic crystal structures.\n\
Return zero for non-periodic structures.\n\
";

const char* doc_StructureAdapter_siteAtomType = "\
Element, isotope or ion symbol at the specified atom site.\n\
\n\
i    -- zero-based atom site index.\n\
\n\
Return a string symbol.\n\
";

const char* doc_StructureAdapter_siteCartesianPosition = "\
Return absolute cartesian coordinates of the specified atom site.\n\
\n\
i    -- zero-based atom site index.\n\
\n\
Return an array of 3 cartesian positions.\n\
";

const char* doc_StructureAdapter_siteMultiplicity = "\
Symmetry multiplicity of the specified atom site in the structure.\n\
\n\
i    -- zero-based atom site index.\n\
\n\
Return integer multiplicity.\n\
";

const char* doc_StructureAdapter_siteOccupancy = "\
Fractional occupancy of the specified site in the structure.\n\
\n\
i    -- zero-based atom site index.\n\
\n\
Return float.\n\
";

const char* doc_StructureAdapter_siteAnisotropy = "\
Flag for anisotropic displacement parameters at the specified site.\n\
\n\
i    -- zero-based atom site index.\n\
\n\
Return boolean flag.\n\
";

const char* doc_StructureAdapter_siteCartesianUij = "\
Matrix of displacement parameters expressed in cartesian coordinates.\n\
\n\
i    -- zero-based atom site index.\n\
\n\
Return a 3 by 3 array.\n\
";

const char* doc_StructureAdapter__customPQConfig = "\
Support for optional custom configuration of the PairQuantity object.\n\
This method is called from the setStructure and eval methods of the owner\n\
PairQuantity object.\n\
\n\
pqobj    -- the owner PairQuantity object.  The function should check for the\n\
            exact type of pqobj and apply configuration accordingly.\n\
\n\
No return value.  This method can be overloaded in the derived class.\n\
No action by default.\n\
";

const char* doc_nometa = "\
Return a proxy to StructureAdapter with _customPQConfig method disabled.\n\
This creates a thin wrapper over a source StructureAdapter object that\n\
disables _customPQConfig.\n\
\n\
stru -- StructureAdapter object or an object convertible to StructureAdapter.\n\
\n\
Return a proxy StructureAdapter with disabled _customPQConfig.\n\
";

const char* doc_nosymmetry = "\
Return a proxy StructureAdapter with crystal symmetry disabled.\n\
For crystals the new adapter generates bonds only within the asymmetric\n\
unit ignoring any translational or other symmetries.\n\
\n\
stru -- StructureAdapter object or an object convertible to StructureAdapter.\n\
\n\
Return a proxy StructureAdapter with disabled symmetry expansion.\n\
";

const char* doc_createStructureAdapter = "\
Create StructureAdapter from a Python object.\n\
\n\
stru -- an object that is convertible to StructureAdapter, i.e., it has\n\
        a registered factory that converts Python structure object to\n\
        StructureAdapter.  Return stru if already a StructureAdapter.\n\
\n\
Return a StructureAdapter instance.\n\
Raise TypeError if stru cannot be converted to StructureAdapter.\n\
";

const char* doc__emptyStructureAdapter = "\
Factory for an empty structure singleton.\n\
\n\
Return a singleton instance of empty StructureAdapter.\n\
";

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

        StructureAdapterPtr clone() const
        {
            return this->get_pure_virtual_override("clone")();
        }


        BaseBondGeneratorPtr createBondGenerator() const
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

    private:

        // serialization
        friend class boost::serialization::access;
        template<class Archive>
            void serialize(Archive& ar, const unsigned int version)
        {
            ar & boost::serialization::base_object<StructureAdapter>(*this);
        }

};  // class StructureAdapterWrap


class StructureAdapterPickleSuite2 :
    public SerializationPickleSuite<StructureAdapter, DICT_PICKLE>
{
    public:

        static boost::python::tuple getinitargs(
                diffpy::srreal::StructureAdapterPtr adpt)
        {
            python::tuple rv;
            // if adapter has been created from Python, we can use the default
            // Python constructor, i.e., __init__ with no arguments.
            bool frompython = dynamic_pointer_cast<StructureAdapterWrap>(adpt);
            if (frompython)  return rv;
            // otherwise the instance is from a non-wrapped C++ adapter,
            // and we need to reconstruct it using boost::serialization
            std::string content = diffpy::serialization_tostring(adpt);
            rv = python::make_tuple(content);
            return rv;
        }


        static boost::python::object constructor()
        {
            return StructureAdapterPickleSuite::constructor();
        }

};  // class StructureAdapterPickleSuite2


}   // namespace nswrap_StructureAdapter

// Wrapper definition --------------------------------------------------------

void wrap_StructureAdapter()
{
    using namespace nswrap_StructureAdapter;

    class_<StructureAdapterWrap, noncopyable>(
            "StructureAdapter", doc_StructureAdapter)
        .def("__init__", StructureAdapterPickleSuite::constructor(),
                doc_StructureAdapter___init__fromstring)
        .def("clone",
                &StructureAdapter::clone,
                doc_StructureAdapter_clone)
        .def("createBondGenerator",
                &StructureAdapter::createBondGenerator,
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
                python::arg("pqobj"),
                doc_StructureAdapter__customPQConfig)
        .def_pickle(StructureAdapterPickleSuite2())
        ;

    register_ptr_to_python<StructureAdapterPtr>();

    def("nometa", nometa<object>, doc_nometa);
    def("nosymmetry", nosymmetry<object>, doc_nosymmetry);
    def("createStructureAdapter", createStructureAdapter,
            doc_createStructureAdapter);
    def("_emptyStructureAdapter", emptyStructureAdapter,
            doc__emptyStructureAdapter);
}

// Export shared docstrings

const char* doc_StructureAdapter__customPQConfig =
    nswrap_StructureAdapter::doc_StructureAdapter__customPQConfig;

}   // namespace srrealmodule

using srrealmodule::nswrap_StructureAdapter::StructureAdapterWrap;
BOOST_CLASS_EXPORT(StructureAdapterWrap)

// End of file
