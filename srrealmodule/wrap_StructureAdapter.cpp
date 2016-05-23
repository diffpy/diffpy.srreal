/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* Bindings to the StructureAdapter class.  So far the wrapper is intended
* only for accessing the C++ created StructureAdapter instances and there
* is no support for method overrides from Python.
*
*****************************************************************************/

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/copy_const_reference.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/python/implicit.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

#include <diffpy/srreal/StructureDifference.hpp>
#include <diffpy/srreal/NoMetaStructureAdapter.hpp>
#include <diffpy/srreal/NoSymmetryStructureAdapter.hpp>
#include <diffpy/srreal/PairQuantity.hpp>
#include <diffpy/serialization.ipp>

namespace srrealmodule {

// declarations
void sync_StructureDifference(boost::python::object obj);

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

const char* doc_StructureAdapter___init__fromstring = "\
Construct StructureAdapter object from a string.  This is used\n\
internally by the pickle protocol and should not be called directly.\n\
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
No return value.  This method can be overloaded in a derived class.\n\
No action by default.\n\
";

const char* doc_StructureAdapter_diff = "\
Evaluate difference between this and other StructureAdapter.\n\
\n\
other    -- another StructureAdapter instance to compare with self\n\
\n\
Return StructureDifference object sd, where sd.stru0 is self,\n\
sd.stru1 other, sd.pop0 are indices of atom sites that are in self,\n\
but not in the other and sd.add1 atom indices that are only in self.\n\
This method can be overloaded in a derived class.\n\
";

const char* doc_NoMetaStructureAdapter = "\
StructureAdapter proxy which disables _customPQConfig method.\n\
";

const char* doc_NoMetaStructureAdapter_init = "\
Create proxy to StructureAdapter that disables _customPQConfig.\n\
\n\
adapter  -- StructureAdapter object to be proxied.  The new adapter\n\
            will avoid calling of adapter._customPQConfig method.\n\
";

const char* doc_NoSymmetryStructureAdapter = "\
StructureAdapter proxy which disables crystal and periodic symmetry.\n\
";

const char* doc_NoSymmetryStructureAdapter_init = "\
Create proxy to StructureAdapter that disables symmetry expansion.\n\
\n\
adapter  -- StructureAdapter object to be proxied.  Any iteration over\n\
            atom pairs from periodic or space-group symmetry will be\n\
            disabled in the proxy adapter.\n\
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

const char* doc__emptyStructureAdapter = "\
Factory for an empty structure singleton.\n\
\n\
Return a singleton instance of empty StructureAdapter.\n\
";

// Local Helpers - forward declarations --------------------------------------

namespace {

void checkindex(const StructureAdapter& adpt, int i);

}   // namespace

// wrappers ------------------------------------------------------------------

const std::string& siteAtomType_safe(const StructureAdapter& adpt, int i)
{
    checkindex(adpt, i);
    return adpt.siteAtomType(i);
}


DECLARE_PYARRAY_METHOD_WRAPPER1(siteCartesianPosition,
        siteCartesianPosition_asarray)

python::object siteCartesianPosition_safe(const StructureAdapter& adpt, int i)
{
    checkindex(adpt, i);
    return siteCartesianPosition_asarray(adpt, i);
}


int siteMultiplicity_safe(const StructureAdapter& adpt, int i)
{
    checkindex(adpt, i);
    return adpt.siteMultiplicity(i);
}


double siteOccupancy_safe(const StructureAdapter& adpt, int i)
{
    checkindex(adpt, i);
    return adpt.siteOccupancy(i);
}


bool siteAnisotropy_safe(const StructureAdapter& adpt, int i)
{
    checkindex(adpt, i);
    return adpt.siteAnisotropy(i);
}


DECLARE_PYARRAY_METHOD_WRAPPER1(siteCartesianUij,
        siteCartesianUij_asarray)

python::object siteCartesianUij_safe(const StructureAdapter& adpt, int i)
{
    checkindex(adpt, i);
    return siteCartesianUij_asarray(adpt, i);
}

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
            if (f)  f(ptr(pq));
            else    this->default_customPQConfig(pq);
        }

        void default_customPQConfig(PairQuantity* pq) const
        {
            this->StructureAdapter::customPQConfig(pq);
        }


        StructureDifference diff(StructureAdapterConstPtr other) const
        {
            override f = this->get_override("diff");
            if (f)
            {
                python::object sdobj = f(other);
                sync_StructureDifference(sdobj);
                StructureDifference& sd =
                    python::extract<StructureDifference&>(sdobj);
                return sd;
            }
            return this->default_diff(other);
        }

        StructureDifference default_diff(StructureAdapterConstPtr other) const
        {
            return this->StructureAdapter::diff(other);
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


template <class T>
class StructureProxyPickleSuite : public boost::python::pickle_suite
{
    public:

        static boost::python::tuple getinitargs(
                diffpy::srreal::StructureAdapterPtr adpt)
        {
            using namespace boost;
            shared_ptr<T> tadpt = dynamic_pointer_cast<T>(adpt);
            StructureAdapterPtr srcadpt = tadpt->getSourceStructure();
            python::tuple rv = python::make_tuple(srcadpt);
            return rv;
        }

};

// Local Helpers -------------------------------------------------------------

namespace {

void checkindex(const StructureAdapter& adpt, int i)
{
    // Leave it up to Python wrapped classes to handle possible index issues.
    if (dynamic_cast<const StructureAdapterWrap*>(&adpt))  return;
    // Prevent out-of-bounds crash from C++ objects.
    if (0 <= i && i < adpt.countSites())  return;
    throw std::out_of_range("Index out of range.");
}

}   // namespace


}   // namespace nswrap_StructureAdapter

// Wrapper definition --------------------------------------------------------

void wrap_StructureAdapter()
{
    using namespace nswrap_StructureAdapter;

    class_<StructureAdapterWrap, noncopyable>(
            "StructureAdapter", doc_StructureAdapter)
        .def("__init__", StructureAdapter_constructor(),
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
        .def("siteAtomType",
                siteAtomType_safe,
                return_value_policy<copy_const_reference>(),
                doc_StructureAdapter_siteAtomType)
        .def("siteAtomType",
                &StructureAdapterWrap::default_siteAtomType,
                return_value_policy<copy_const_reference>())
        .def("siteCartesianPosition",
                    siteCartesianPosition_safe,
                    doc_StructureAdapter_siteCartesianPosition)
        .def("siteMultiplicity",
                siteMultiplicity_safe,
                doc_StructureAdapter_siteMultiplicity)
        .def("siteMultiplicity",
                &StructureAdapterWrap::default_siteMultiplicity)
        .def("siteOccupancy",
                siteOccupancy_safe,
                doc_StructureAdapter_siteOccupancy)
        .def("siteOccupancy",
                &StructureAdapterWrap::default_siteOccupancy)
        .def("siteAnisotropy",
                siteAnisotropy_safe,
                doc_StructureAdapter_siteAnisotropy)
        .def("siteCartesianUij",
                siteCartesianUij_safe,
                doc_StructureAdapter_siteCartesianUij)
        .def("_customPQConfig",
                &StructureAdapter::customPQConfig,
                &StructureAdapterWrap::default_customPQConfig,
                python::arg("pqobj"),
                doc_StructureAdapter__customPQConfig)
        .def("diff",
                &StructureAdapter::diff,
                &StructureAdapterWrap::default_diff,
                python::arg("other"),
                doc_StructureAdapter_diff)
        .def_pickle(StructureAdapterPickleSuite<StructureAdapterWrap>())
        ;

    register_ptr_to_python<StructureAdapterPtr>();
    implicitly_convertible<StructureAdapterPtr, StructureAdapterConstPtr>();

    typedef boost::shared_ptr<NoMetaStructureAdapter>
        NoMetaStructureAdapterPtr;
    class_<NoMetaStructureAdapter,
        bases<StructureAdapter>, NoMetaStructureAdapterPtr>(
            "NoMetaStructureAdapter", doc_NoMetaStructureAdapter)
        .def(init<StructureAdapterPtr>(python::arg("adapter"),
                    doc_NoMetaStructureAdapter_init))
        .def_pickle(StructureProxyPickleSuite<NoMetaStructureAdapter>())
        ;

    typedef boost::shared_ptr<NoSymmetryStructureAdapter>
        NoSymmetryStructureAdapterPtr;
    class_<NoSymmetryStructureAdapter,
        bases<StructureAdapter>, NoSymmetryStructureAdapterPtr>(
            "NoSymmetryStructureAdapter", doc_NoSymmetryStructureAdapter)
        .def(init<StructureAdapterPtr>(python::arg("adapter"),
                    doc_NoSymmetryStructureAdapter_init))
        .def_pickle(StructureProxyPickleSuite<NoSymmetryStructureAdapter>())
        ;

    def("nometa", nometa<object>, doc_nometa);
    def("nosymmetry", nosymmetry<object>, doc_nosymmetry);
    def("_emptyStructureAdapter", emptyStructureAdapter,
            doc__emptyStructureAdapter);
}

// Export shared docstrings

const char* doc_StructureAdapter___init__fromstring =
    nswrap_StructureAdapter::doc_StructureAdapter___init__fromstring;
const char* doc_StructureAdapter__customPQConfig =
    nswrap_StructureAdapter::doc_StructureAdapter__customPQConfig;
const char* doc_StructureAdapter_diff =
    nswrap_StructureAdapter::doc_StructureAdapter_diff;

}   // namespace srrealmodule

using srrealmodule::nswrap_StructureAdapter::StructureAdapterWrap;
BOOST_CLASS_EXPORT(StructureAdapterWrap)

// End of file
