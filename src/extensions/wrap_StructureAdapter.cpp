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

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/shared_ptr.h>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

#include <diffpy/srreal/StructureDifference.hpp>
#include <diffpy/srreal/NoMetaStructureAdapter.hpp>
#include <diffpy/srreal/NoSymmetryStructureAdapter.hpp>
#include <diffpy/srreal/PairQuantity.hpp>
#include <diffpy/serialization.ipp>

namespace nb = nanobind;

namespace srrealmodule {

// declarations
void sync_StructureDifference(nb::object obj);

namespace nswrap_StructureAdapter {

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

nb::object siteCartesianPosition_safe(const StructureAdapter& adpt, int i)
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

nb::object siteCartesianUij_safe(const StructureAdapter& adpt, int i)
{
    checkindex(adpt, i);
    return siteCartesianUij_asarray(adpt, i);
}


PyObject* restoreStructureAdapter(PyObject*, PyObject* args)
{
    PyObject* content = nullptr;
    if (!PyArg_UnpackTuple(args, "_restoreStructureAdapter", 1, 1, &content))
        return nullptr;
    if (!PyBytes_Check(content))
    {
        PyErr_SetString(PyExc_TypeError, "expected bytes");
        return nullptr;
    }

    char* buffer = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(content, &buffer, &size) < 0)
        return nullptr;

    try
    {
        StructureAdapterPtr adapter =
            createStructureAdapterFromString(std::string(buffer, size));
        nb::object pyadapter = nb::cast(adapter);
        return pyadapter.release().ptr();
    }
    catch (const std::exception& e)
    {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}


PyMethodDef restoreStructureAdapterDef = {
    "_restoreStructureAdapter",
    restoreStructureAdapter,
    METH_VARARGS,
    "Restore native C++ StructureAdapter from serialized bytes."
};


// Helper class necessary for wrapping a pure virtual methods

class StructureAdapterWrap :
    public StructureAdapter,
    public PythonTrampolineTag
{
    public:

        NB_TRAMPOLINE(StructureAdapter, 13);

        StructureAdapterPtr clone() const override
        {
            NB_OVERRIDE_PURE(clone);
        }


        BaseBondGeneratorPtr createBondGenerator() const override
        {
            NB_OVERRIDE_PURE(createBondGenerator);
        }


        int countSites() const override
        {
            NB_OVERRIDE_PURE(countSites);
        }


        // TODO: totalOccupancy is not marked as virtual function
        // need to check behaviour
        double totalOccupancy() const
        {
            return this->default_totalOccupancy();
        }

        double default_totalOccupancy() const
        {
            return this->StructureAdapter::totalOccupancy();
        }


        double numberDensity() const override
        {
            NB_OVERRIDE(numberDensity);
        }

        double default_numberDensity() const
        {
            return this->StructureAdapter::numberDensity();
        }


        const std::string& siteAtomType(int idx) const override
        {
            static std::string rv;
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "siteAtomType", false);

            if (ticket.key.is_valid()) 
            {
                nb::object atp = nb_trampoline.base().attr(ticket.key)(idx);
                rv = nb::cast<std::string>(atp);
                return rv;
            }

            return this->default_siteAtomType(idx);
        }

        const std::string& default_siteAtomType(int idx) const
        {
            return this->StructureAdapter::siteAtomType(idx);
        }


        const R3::Vector& siteCartesianPosition(int idx) const override
        {
            static R3::Vector rv;
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "siteCartesianPosition", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method StructureAdapter.siteCartesianPosition() called"
                );
            }

            nb::object pos = nb_trampoline.base().attr(ticket.key)(idx);
            for (int i = 0; i < R3::Ndim; ++i)
            {
                rv[i] = nb::cast<double>(pos[i]);
            }
            return rv;
        }


        int siteMultiplicity(int idx) const override
        {
            NB_OVERRIDE(siteMultiplicity, idx);
        }

        int default_siteMultiplicity(int idx) const
        {
            return this->StructureAdapter::siteMultiplicity(idx);
        }


        double siteOccupancy(int idx) const override
        {
            NB_OVERRIDE(siteOccupancy, idx);
        }

        double default_siteOccupancy(int idx) const
        {
            return this->StructureAdapter::siteOccupancy(idx);
        }


        bool siteAnisotropy(int idx) const override
        {
            NB_OVERRIDE_PURE(siteAnisotropy, idx);
        }


        const R3::Matrix& siteCartesianUij(int idx) const override
        {
            static R3::Matrix rv;
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "siteCartesianUij", true);

            if (!ticket.key.is_valid()) 
            {
                throw nb::type_error(
                    "pure virtual method StructureAdapter.siteCartesianUij() called"
                );
            }

            nb::object uij = nb_trampoline.base().attr(ticket.key)(idx);
            for (int i = 0; i < R3::Ndim; ++i)
            {
                for (int j = 0; j < R3::Ndim; ++j)
                {
                    rv(i, j) = nb::cast<double>(uij[i][j]);
                }
            }
            return rv;
        }


        void customPQConfig(PairQuantity* pq) const override
        {
            NB_OVERRIDE_NAME("_customPQConfig", customPQConfig, pq);
        }

        void default_customPQConfig(PairQuantity* pq) const
        {
            this->StructureAdapter::customPQConfig(pq);
        }


        StructureDifference diff(StructureAdapterConstPtr other) const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "diff", false);

            if (ticket.key.is_valid()) {
                nb::object sdobj = nb_trampoline.base().attr(ticket.key)(other);
                sync_StructureDifference(sdobj);
                StructureDifference &sd =
                    nb::cast<StructureDifference&>(sdobj);
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


double numberDensity_dispatch(const StructureAdapter& adpt)
{
    const StructureAdapterWrap *wrap =
        dynamic_cast<const StructureAdapterWrap *>(&adpt);
    return wrap ? wrap->default_numberDensity() : adpt.numberDensity();
}


const std::string& siteAtomType_dispatch(const StructureAdapter& adpt, int i)
{
    const StructureAdapterWrap *wrap =
        dynamic_cast<const StructureAdapterWrap *>(&adpt);
    return wrap ? wrap->default_siteAtomType(i) : siteAtomType_safe(adpt, i);
}


int siteMultiplicity_dispatch(const StructureAdapter& adpt, int i)
{
    const StructureAdapterWrap *wrap =
        dynamic_cast<const StructureAdapterWrap *>(&adpt);
    return wrap ?
        wrap->default_siteMultiplicity(i) :
        siteMultiplicity_safe(adpt, i);
}


double siteOccupancy_dispatch(const StructureAdapter& adpt, int i)
{
    const StructureAdapterWrap *wrap =
        dynamic_cast<const StructureAdapterWrap *>(&adpt);
    return wrap ?
        wrap->default_siteOccupancy(i) :
        siteOccupancy_safe(adpt, i);
}


BaseBondGeneratorPtr createBondGenerator_shared(StructureAdapterPtr adpt)
{
    return adpt->createBondGenerator();
}


template <class T>
class StructureProxyPickleSuite
{
    public:

        template <typename C>
        static void bind(C& cls)
        {
            cls
                .def("__reduce__", [](nb::object self)
                {
                    T& adapter = nb::cast<T&>(self);
                    StructureAdapterPtr src = adapter.getSourceStructure();
                    nb::object dict = get_instance_dict(self);

                    if (dict.is_none() || nb::len(dict) == 0)
                    {
                        return nb::make_tuple(
                            runtime_type(self),
                            nb::make_tuple(src)
                        );
                    }

                    return nb::make_tuple(
                        runtime_type(self),
                        nb::make_tuple(src),
                        nb::make_tuple(nb::none(), dict)
                    );
                })
                ;
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
    PyErr_SetString(PyExc_IndexError, "Index out of range.");
    throw nb::python_error();
}

}   // namespace


}   // namespace nswrap_StructureAdapter

// Wrapper definition --------------------------------------------------------

void wrap_StructureAdapter(nb::module_& m)
{
    using namespace nswrap_StructureAdapter;

    nb::class_<StructureAdapter, StructureAdapterWrap> structureadapter(m,
            "StructureAdapter", doc_StructureAdapter);
    structureadapter
        .def(nb::init<>())
        .def("clone",
                &StructureAdapter::clone,
                doc_StructureAdapter_clone)
        .def("createBondGenerator",
                createBondGenerator_shared,
                doc_StructureAdapter_createBondGenerator)
        .def("countSites", &StructureAdapter::countSites,
                doc_StructureAdapter_countSites)
        .def("totalOccupancy",
                &StructureAdapter::totalOccupancy,
                doc_StructureAdapter_totalOccupancy)
        .def("numberDensity", numberDensity_dispatch,
                doc_StructureAdapter_numberDensity)
        .def("siteAtomType",
                siteAtomType_dispatch,
                doc_StructureAdapter_siteAtomType)
        .def("siteCartesianPosition",
                    siteCartesianPosition_safe,
                    doc_StructureAdapter_siteCartesianPosition)
        .def("siteMultiplicity",
                siteMultiplicity_dispatch,
                doc_StructureAdapter_siteMultiplicity)
        .def("siteOccupancy",
                siteOccupancy_dispatch,
                doc_StructureAdapter_siteOccupancy)
        .def("siteAnisotropy",
                siteAnisotropy_safe,
                doc_StructureAdapter_siteAnisotropy)
        .def("siteCartesianUij",
                siteCartesianUij_safe,
                doc_StructureAdapter_siteCartesianUij)
        .def("_customPQConfig",
                &StructureAdapter::customPQConfig,
                nb::arg("pqobj"),
                doc_StructureAdapter__customPQConfig)
        .def("diff",
                &StructureAdapter::diff,
                nb::arg("other"),
                doc_StructureAdapter_diff)
        ;
        StructureAdapterPickleSuite<StructureAdapter, StructureAdapterWrap>::bind(
            structureadapter);

    typedef std::shared_ptr<NoMetaStructureAdapter>
        NoMetaStructureAdapterPtr;
    nb::class_<NoMetaStructureAdapter, StructureAdapter> nometastructureadapter(m,
            "NoMetaStructureAdapter", doc_NoMetaStructureAdapter);
    nometastructureadapter
        .def(nb::init<StructureAdapterPtr>(), nb::arg("adapter"),
                    doc_NoMetaStructureAdapter_init)
        ;
        StructureProxyPickleSuite<NoMetaStructureAdapter>::bind(nometastructureadapter);

    typedef std::shared_ptr<NoSymmetryStructureAdapter>
        NoSymmetryStructureAdapterPtr;
    nb::class_<NoSymmetryStructureAdapter, StructureAdapter> nosymmetrystructureadapter(m,
            "NoSymmetryStructureAdapter", doc_NoSymmetryStructureAdapter);
    nosymmetrystructureadapter
        .def(nb::init<StructureAdapterPtr>(), nb::arg("adapter"),
                    doc_NoSymmetryStructureAdapter_init)
        ;
        StructureProxyPickleSuite<NoSymmetryStructureAdapter>::bind(nosymmetrystructureadapter);

    m.def("nometa", nometa<nb::object>, doc_nometa);
    m.def("nosymmetry", nosymmetry<nb::object>, doc_nosymmetry);
    m.def("_emptyStructureAdapter", emptyStructureAdapter,
            doc__emptyStructureAdapter);
    nb::object module_name = nb::str("diffpy.srreal.srreal_ext");
    PyObject* restore_function = PyCFunction_NewEx(
        &restoreStructureAdapterDef, nullptr, module_name.ptr());
    if (!restore_function)
        nb::raise_python_error();
    if (PyModule_AddObject(m.ptr(), "_restoreStructureAdapter",
            restore_function) < 0)
    {
        Py_DECREF(restore_function);
        nb::raise_python_error();
    }
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
