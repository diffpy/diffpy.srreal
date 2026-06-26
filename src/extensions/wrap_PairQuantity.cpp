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
* Bindings to the PairQuantity class.  The business protected methods
* can be overridden from Python to create custom calculator.
* The class provides bindings to the eval and value methods for all derived
* calculators and also the double attributes access that is inherited from
* the Attributes wrapper in wrap_Attributes.
*
* Exported classes in Python:
*
* class QuantityType -- wrapped std::vector<double>
*
* class BasePairQuantity -- base class to all calculators in Python

* class PairQuantity -- derived class with publicized protected methods
* _addPairContribution, _resetValue, etc.  Allows their override from Python.
*
*****************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/vector.h>

#include <cstdlib>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"
#include "srreal_validators.hpp"

#include <diffpy/srreal/PairQuantity.hpp>
#include <type_traits>

namespace nb = nanobind;
NB_MAKE_OPAQUE(diffpy::srreal::QuantityType);

namespace srrealmodule {
namespace nswrap_PairQuantity {

using namespace diffpy::srreal;

// docstrings ----------------------------------------------------------------

const char* doc_BasePairQuantity_eval = "\
Calculate a pair quantity for the specified structure.\n\
\n\
stru -- structure object that can be converted to StructureAdapter.\n\
        Use the last structure when None.\n\
\n\
Return a copy of the internal total contributions.\n\
May need to be further transformed to get the desired value.\n\
";

const char* doc_BasePairQuantity_value = "\
Internal vector of total contributions as numpy array.\n\
";

const char* doc_BasePairQuantity__mergeParallelData = "\
Process raw results string from a parallel job and add them to this instance.\n\
\n\
pdata    -- raw data string from the parallel _getParallelData function.\n\
            The actual processing of pdata happens in _executeParallelMerge.\n\
ncpu     -- number of parallel jobs.  The finishValue method is called after\n\
            merging ncpu parallel values.\n\
\n\
No return value.  For parallel calculation this method has to be executed\n\
exactly ncpu times in the master object after the resetValue call.\n\
Raise RuntimeError if called too many times.\n\
";

const char* doc_BasePairQuantity__getParallelData = "\
Return raw results string from a parallel job.\n\
";

const char* doc_BasePairQuantity_setStructure = "\
Assign structure to be evaluated without executing the calculation.\n\
This zeros the internal values array and updates the pair mask data.\n\
\n\
stru -- structure object that can be converted to StructureAdapter.\n\
\n\
No return value.\n\
";

const char* doc_BasePairQuantity_getStructure = "\
The StructureAdapter instance of the last evaluated structure.\n\
";

const char* doc_BasePairQuantity__setupParallelRun = "\
Configure this object for a partial calculation in a parallel run.\n\
\n\
cpuindex -- integer from 0 to ncpu-1 that identifies the partial\n\
            calculation to be evaluated.\n\
ncpu     -- number of parallel processes or the total number of\n\
            partial calculations.  Must be at least one.\n\
\n\
No return value.\n\
";

const char* doc_BasePairQuantity_evaluatortype = "\
String type of preferred evaluation procedure.\n\
\n\
Possible values are 'BASIC', 'OPTIMIZED', and 'CHECK'.  The value is\n\
always calculated from scratch when 'BASIC'.  The 'OPTIMIZED' evaluation\n\
updates the existing results by recalculating only the contributions\n\
from changed atoms.  The 'CHECK' evaluation verifies that 'OPTIMIZED'\n\
and 'BASIC' results are equal.  It is intended only for debugging\n\
purposes.\n\
\n\
See also evaluatortypeused.\n\
";

const char* doc_BasePairQuantity_evaluatortypeused = "\
String type of evaluation procedure used in the last calculation.\n\
\n\
Possible values are 'BASIC', 'OPTIMIZED', 'CHECK', and 'NONE'\n\
when calculator that has not been used yet.\n\
";

const char* doc_BasePairQuantity_maskAllPairs = "\
Set the calculation mask for all atom pairs in the structure.\n\
\n\
mask -- True if all pairs should be included, False if excluded.\n\
\n\
No return value.\n\
";

const char* doc_BasePairQuantity_invertMask = "\
Invert mask that controls which atom pairs should be included.\n\
";

const char* doc_BasePairQuantity_setPairMask = "\
Include or exclude specified atom pairs in the calculation.\n\
The pair masking is exclusively based either on site indices\n\
or atom types.  This function applies index-based masking and\n\
cancels any type-based masks.\n\
\n\
i    -- zero based index of the first site in the pair.\n\
        Can be also an iterable of indices or a string 'all' or 'ALL',\n\
        which select all sites in the structure.\n\
j    -- index of the second site in the pair.  Can be an iterable\n\
        or string 'all', just like argument i\n\
mask -- mask of the atom pair i, j,  True if included, False if excluded.\n\
others -- optional mask applied to all other pairs.  Ignored when None.\n\
\n\
No return value.\n\
";

const char* doc_BasePairQuantity_getPairMask = "\
Return calculation mask for a pair of atom indices.\n\
\n\
i    -- zero based index of the first site in the pair\n\
j    -- zero based index of the second site in the pair\n\
\n\
Return boolean mask.  Note the value may be incorrect, because type-based\n\
masking is applied with a delay.  The value is guaranteed correct after\n\
a call of setStructure or eval methods.\n\
";

const char* doc_BasePairQuantity_setTypeMask = "\
Include or exclude specified atom-type pairs in the calculation.\n\
The pair masking is exclusively based either on site indices\n\
or atom types.  This function applies type-based masking and\n\
cancels any previous index-based masks.\n\
\n\
tpi  -- first atom type in the pair, string or an iterable of strings.\n\
        When 'all' or 'ALL', tpi refers to all sites in the structure.\n\
tpj  -- second atom type in the pair, string or an iterable of strings.\n\
        When 'all' or 'ALL', tpj refers to all sites in the structure.\n\
mask -- mask for the atom types pair.\n\
        True if included, False if excluded.\n\
others -- optional mask applied to all other pairs.  Ignored when None.\n\
\n\
No return value.\n\
";

const char* doc_BasePairQuantity_getTypeMask = "\
Return calculation mask for a pair of atom types.\n\
\n\
tpi  -- element symbol of the first type in the pair.\n\
tpj  -- element symbol of the second type in the pair.\n\
\n\
Return boolean mask.  The value is meaningless for index-based\n\
masking.  Use getTypeMask('', '') to get the default pair mask.\n\
";

const char* doc_BasePairQuantity_ticker = "\
Return EventTicker object with the last configuration change time.\n\
\n\
The ticker should be clicked on every configuration change that\n\
requires reevaluation of the PairQuantity even for an unchanged\n\
structure.\n\
";

const char* doc_BasePairQuantity_copy = "\
Return a deep copy of this PairQuantity object.\n\
";

const char* doc_PairQuantity = "\
Base class for Python defined pair quantity calculators.\n\
No action by default.  Concrete calculators must override the\n\
_addPairContribution method to get some results.\n\
";

const char* doc_PairQuantity_ticker = "\
Return EventTicker object with the last configuration change time.\n\
\n\
The ticker should be clicked on every configuration change that\n\
requires reevaluation of the PairQuantity even for an unchanged\n\
structure.\n\
\n\
This method can be overridden in the derived class.\n\
";

const char* doc_PairQuantity__getParallelData = "\
Return raw results string from a parallel job.\n\
By default a serialized content of the internal values array.\n\
This can be added to the master object values by calling\n\
PairQuantity._executeParallelMerge.\n\
\n\
This method can be overridden in the derived class.\n\
";

const char* doc_PairQuantity__resizeValue = "\
Resize the internal contributions array to the specified size.\n\
\n\
sz   -- new length of the internal array.\n\
\n\
No return value.  This method can be overridden in the derived class.\n\
";

const char* doc_PairQuantity__resetValue = "\
Reset all contribution in the internal array to zero.\n\
\n\
No return value.  This method can be overridden in the derived class.\n\
For parallel calculations this resets the count of merged parallel\n\
results to zero.\n\
";

const char* doc_PairQuantity__configureBondGenerator = "\
Configure bond generator just before the start of summation.\n\
The default method sets the upper and lower limits for the pair\n\
distances.  A method override can be used to apply a different\n\
distance range.\n\
\n\
bnds -- instance of BaseBondGenerator to be configured\n\
\n\
No return value.  This method can be overridden in the derived class.\n\
";

const char* doc_PairQuantity__addPairContribution = "\
Process pair contribution at a unique bond generator state.\n\
No action by default, needs to be overridden to do something.\n\
\n\
bnds     -- instance of BaseBondGenerator holding data for\n\
            a particular pair of atoms during summation.\n\
sumscale -- integer scaling for this contribution passed from\n\
            PQEvaluator.  Equals 1 if bnds.site0() == bnds.site1(),\n\
            2 otherwise.  Can be negative when contribution is\n\
            removed for fast quantity updates.\n\
\n\
No return value.  This method is executed for every atom pair in the structure.\n\
";

const char* doc_PairQuantity__executeParallelMerge = "\
Process raw results string from a parallel job and add them to this instance.\n\
By default converts the string to an array an adds it to the internal values.\n\
This method should be never used directly, it is instead called by the\n\
_mergeParallelData method.\n\
\n\
pdata    -- raw data string from the parallel _getParallelData function.\n\
\n\
No return value.  This method can be overridden in the derived class.\n\
";

const char* doc_PairQuantity__finishValue = "\
Final processing of the results after iteration over all pairs.\n\
This is for operations that are not suitable in the _addPairContribution\n\
method, for example sorting.\n\
\n\
No return value.  This method can be overridden in the derived class.\n\
No action by default.\n\
";

const char* doc_PairQuantity__stashPartialValue = "\
Save results from unchanged part of the structure in OPTIMIZED evaluation.\n\
\n\
This method gets called in OPTIMIZED calculation just before assigning\n\
the new structure, which implicitly calls _resetValue.  The method must\n\
store internal partial results, for example as private class attributes.\n\
The accompanying function _restorePartialValue then recovers the stored\n\
values to undo the _resetValue effects.\n\
\n\
No return value.  This method must be overridden in the derived class to\n\
support OPTIMIZED evaluation.\n\
\n\
See also _restorePartialValue.\n\
";

const char* doc_PairQuantity__restorePartialValue = "\
Restore partial results from unchanged sub-structure in OPTIMIZED evaluation.\n\
\n\
This method is executed in OPTIMIZED calculation after assignment of the\n\
new Structure and the implicit call of _resetValue.  The method must\n\
restore internal results that were saved before by _stashPartialValue.\n\
\n\
No return value.  This method must be overridden in the derived class to\n\
support OPTIMIZED evaluation.\n\
\n\
See also _stashPartialValue.\n\
";

const char* doc_PairQuantity__value = "\
Reference to the internal vector of total contributions.\n\
";

// wrappers ------------------------------------------------------------------

inline nb::bytes to_bytes(const std::string &s)
{
    return nb::bytes(s.data(), s.size());
}


inline std::string from_bytes(nb::bytes b)
{
    return std::string(
        static_cast<const char *>(b.data()),
        b.size()
    );
}


// representation of QuantityType objects
nb::object repr_QuantityType(const QuantityType& v)
{
    nb::list values;

    for (size_t i = 0; i < v.size(); ++i) {
        values.append(v[i]);
    }

    nb::object t = nb::module_::import_("builtins").attr("tuple")(values);
    return nb::str("QuantityType{}").attr("format")(nb::repr(t));
}


// PairQuantity::eval is a template non-constant method and
// needs an explicit wrapper function.

nb::object eval_asarray(PairQuantity& obj, nb::object& a)
{
    QuantityType value = (a.is_none()) ? obj.eval() : obj.eval(a);
    nb::object rv = convertToNumPyArray(value);
    return rv;
}

// support for the evaluatortype property

const char* evtp_NONE = "NONE";
const char* evtp_BASIC = "BASIC";
const char* evtp_OPTIMIZED = "OPTIMIZED";
const char* evtp_CHECK = "CHECK";

std::string stringevaluatortype(PQEvaluatorType tp)
{
    switch (tp)
    {
        case NONE:
            return evtp_NONE;
        case BASIC:
            return evtp_BASIC;
        case OPTIMIZED:
            return evtp_OPTIMIZED;
        case CHECK:
            return evtp_CHECK;
    }
    PyErr_SetString(PyExc_NotImplementedError,
                    "Unknown internal value of PQEvaluatorType.");
    throw nb::python_error();
}


std::string getevaluatortype(const PairQuantity& obj)
{
    return stringevaluatortype(obj.getEvaluatorType());
}


void setevaluatortype(PairQuantity& pq, const std::string& tp)
{
    if (tp == evtp_BASIC)  return pq.setEvaluatorType(BASIC);
    if (tp == evtp_OPTIMIZED)  return pq.setEvaluatorType(OPTIMIZED);
    if (tp == evtp_CHECK)  return pq.setEvaluatorType(CHECK);

    throw nb::value_error(
        "evaluatortype must be one of ('BASIC', 'OPTIMIZED', 'CHECK')."
    );
}

// support for the evaluatortypeused read-only property

std::string getevaluatortypeused(const PairQuantity& obj)
{
    return stringevaluatortype(obj.getEvaluatorTypeUsed());
}

// support "all", "ALL" and integer iterables in setPairMask

std::vector<int> parsepairindex(nb::object i)
{
    std::vector<int> rv;
    // string equal "all" or "ALL"
    if (nb::isinstance<nb::str>(i)) 
    {
        std::string s = nb::cast<std::string>(i);

        if (s != PairQuantity::ALLATOMSSTR && s != "ALL") 
        {
            throw nb::value_error("String argument must be 'all' or 'ALL'.");
        }

        rv.push_back(PairQuantity::ALLATOMSINT);
        return rv;
    }
    // otherwise translate to a vector of integers
    rv = extractintvector(i);
    return rv;
}

// support string iterables in setTypeMask

std::vector<std::string> parsepairtypes(nb::object smbl)
{
    if (nb::isinstance<nb::str>(smbl)) 
    {
        return { nb::cast<std::string>(smbl) };
    }

    return nb::cast<std::vector<std::string>>(smbl);
}


void mask_all_pairs(PairQuantity& obj, nb::object msk)
{
    obj.maskAllPairs(nb::cast<bool>(msk));
}


void set_pair_mask(PairQuantity& obj,
        nb::object i, nb::object j, nb::object msk,
        nb::object others)
{
    if (!others.is_none())  mask_all_pairs(obj, others);
    bool mask = nb::cast<bool>(msk);
    // short circuit for normal call with scalar values
    if (!isiterable(i) && !isiterable(j))
    {
        const int i1 = extractint(i);
        const int j1 = extractint(j);
        obj.setPairMask(i1, j1, mask);
        return;
    }
    std::vector<int> iindices = parsepairindex(i);
    std::vector<int> jindices = parsepairindex(j);
    std::vector<int>::const_iterator ii, jj;
    for (ii = iindices.begin(); ii != iindices.end(); ++ii)
    {
        for (jj = jindices.begin(); jj != jindices.end(); ++jj)
        {
            obj.setPairMask(*ii, *jj, mask);
        }
    }
}


void set_type_mask(PairQuantity& obj,
        nb::object smbli, nb::object smblj, nb::object msk,
        nb::object others)
{
    using namespace std;
    if (!others.is_none())  mask_all_pairs(obj, others);

    bool mask = nb::cast<bool>(msk);

    std::vector<std::string> isymbols = parsepairtypes(smbli);
    std::vector<std::string> jsymbols = parsepairtypes(smblj);
    vector<string>::const_iterator tii, tjj;
    for (const std::string &ti : isymbols) 
    {
        for (const std::string &tj : jsymbols) 
        {
            obj.setTypeMask(ti, tj, mask);
        }
    }
}


// provide a copy method for convenient deepcopy of the object

nb::object pqcopy(nb::object pqobj)
{
    nb::object copy = nb::module_::import_("copy").attr("copy");
    return copy(pqobj);
}

// Helper C++ class for publicizing the protected methods.

class PairQuantityExposed : public PairQuantity
{
    public:

        // non-constant version suitable for exposing mvalue in Python
        QuantityType& value()
        {
            return mvalue;
        }


        void resizeValue(size_t sz)
        {
            this->PairQuantity::resizeValue(sz);
        }


        void resetValue()
        {
            this->PairQuantity::resetValue();
        }


        void configureBondGenerator(BaseBondGenerator& bnds) const
        {
            this->PairQuantity::configureBondGenerator(bnds);
        }


        void addPairContribution(const BaseBondGenerator& bnds, int sumscale)
        {
            this->PairQuantity::addPairContribution(bnds, sumscale);
        }


        void executeParallelMerge(const std::string& pdata)
        {
            this->PairQuantity::executeParallelMerge(pdata);
        }


        void finishValue()
        {
            this->PairQuantity::finishValue();
        }


        void stashPartialValue()
        {
            this->PairQuantity::stashPartialValue();
        }


        void restorePartialValue()
        {
            this->PairQuantity::restorePartialValue();
        }

};


// The second helper class allows override of the exposed PairQuantity
// methods from Python.

class PairQuantityWrap :
    public PairQuantityExposed,
    public PythonTrampolineTag
{
    public:

        NB_TRAMPOLINE(PairQuantityExposed, 10);

        // Make getParallelData overridable from Python.

        std::string getParallelData() const override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(
                nb_trampoline, "_getParallelData", false);

            if (ticket.key.is_valid())
            {
                nb::object pdata =
                    nb_trampoline.base().attr(ticket.key)();
                return from_bytes(nb::cast<nb::bytes>(pdata));
            }

            return this->PairQuantityExposed::getParallelData();
        }

        std::string default_getParallelData() const
        {
            return this->PairQuantityExposed::getParallelData();
        }

        // Make the ticker method overridable from Python

        diffpy::eventticker::EventTicker& ticker() const override
        {
            using diffpy::eventticker::EventTicker;
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(nb_trampoline, "ticker", false);

            if (ticket.key.is_valid()) 
            {
                nb::object ptic = nb_trampoline.base().attr(ticket.key)();
                return nb::cast<EventTicker &>(ptic);
            }

            return this->default_ticker();
        }

        diffpy::eventticker::EventTicker& default_ticker() const
        {
            return this->PairQuantityExposed::ticker();
        }

        // Make the protected virtual methods public so they
        // can be exported to Python and overridden as well.

        void resizeValue(size_t sz) override
        {
            NB_OVERRIDE_NAME("_resizeValue", resizeValue, sz);
        }

        void default_resizeValue(size_t sz)
        {
            this->PairQuantityExposed::resizeValue(sz);
        }


        void resetValue() override
        {
            NB_OVERRIDE_NAME("_resetValue", resetValue);
        }

        void default_resetValue()
        {
            this->PairQuantityExposed::resetValue();
        }


        void configureBondGenerator(BaseBondGenerator& bnds) const override
        {
            NB_OVERRIDE_NAME("_configureBondGenerator", configureBondGenerator, bnds);
        }

        void default_configureBondGenerator(BaseBondGenerator& bnds) const
        {
            this->PairQuantityExposed::configureBondGenerator(bnds);
        }


        void addPairContribution(const BaseBondGenerator& bnds,
                int summationscale) override
        {
            NB_OVERRIDE_NAME("_addPairContribution",
                            addPairContribution,
                            bnds,
                            summationscale);
        }

        void default_addPairContribution(const BaseBondGenerator& bnds,
                int summationscale)
        {
            this->PairQuantityExposed::addPairContribution(bnds, summationscale);
        }


        void executeParallelMerge(const std::string& pdata) override
        {
            nb::gil_scoped_acquire gil;
            nb::detail::ticket ticket(
                nb_trampoline, "_executeParallelMerge", false);

            if (ticket.key.is_valid())
            {
                nb_trampoline.base().attr(ticket.key)(to_bytes(pdata));
                return;
            }

            this->PairQuantityExposed::executeParallelMerge(pdata);
        }

        void default_executeParallelMerge(const std::string& pdata)
        {
            this->PairQuantityExposed::executeParallelMerge(pdata);
        }


        void finishValue() override
        {
        NB_OVERRIDE_NAME("_finishValue", finishValue);
        }

        void default_finishValue()
        {
            this->PairQuantityExposed::finishValue();
        }


        void stashPartialValue() override
        {
            NB_OVERRIDE_NAME("_stashPartialValue", stashPartialValue);
        }

        void default_stashPartialValue()
        {
            this->PairQuantityExposed::stashPartialValue();
        }


        void restorePartialValue() override
        {
            NB_OVERRIDE_NAME("_restorePartialValue", restorePartialValue);
        }

        void default_restorePartialValue()
        {
            this->PairQuantityExposed::restorePartialValue();
        }

};  // class PairQuantityWrap

}   // namespace nswrap_PairQuantity

// Wrapper definition --------------------------------------------------------

void wrap_PairQuantity(nb::module_& m)
{
    using namespace nswrap_PairQuantity;
    using diffpy::Attributes;

    typedef StructureAdapterPtr&(PairQuantity::*getstru)();

    nb::bind_vector<QuantityType>(m, "QuantityType")
        .def("__repr__", &repr_QuantityType);

    nb::class_<PairQuantity, Attributes>
        basepq(m, "BasePairQuantity");
    basepq
        .def(nb::init<>())
        .def("eval", eval_asarray, nb::arg("stru")=nb::none(),
                doc_BasePairQuantity_eval)
        .def_prop_ro("value", value_asarray<PairQuantity>,
                doc_BasePairQuantity_value)
        .def("_mergeParallelData",
                [](PairQuantity &pq, nb::bytes pdata, int ncpu)
                {
                    pq.mergeParallelData(from_bytes(pdata), ncpu);
                },
                nb::arg("pdata"), nb::arg("ncpu"),
                doc_BasePairQuantity__mergeParallelData)
        .def("_getParallelData",
                [](const PairQuantity &pq)
                {
                    return to_bytes(pq.getParallelData());
                },
                doc_BasePairQuantity__getParallelData)
        .def("setStructure", [](PairQuantity &pq, nb::object stru) 
                {
                    pq.setStructure(stru);
                },
                nb::arg("stru"),
                doc_BasePairQuantity_setStructure)
        .def("getStructure", getstru(&PairQuantity::getStructure),
                doc_BasePairQuantity_getStructure)
        .def("_setupParallelRun", &PairQuantity::setupParallelRun,
                nb::arg("cpuindex"), nb::arg("ncpu"),
                doc_BasePairQuantity__setupParallelRun)
        .def_prop_rw("evaluatortype",
                getevaluatortype, setevaluatortype,
                doc_BasePairQuantity_evaluatortype)
        .def_prop_ro("evaluatortypeused",
                getevaluatortypeused,
                doc_BasePairQuantity_evaluatortypeused)
        .def("maskAllPairs", mask_all_pairs,
                nb::arg("mask"),
                doc_BasePairQuantity_maskAllPairs)
        .def("invertMask", &PairQuantity::invertMask,
                doc_BasePairQuantity_invertMask)
        .def("setPairMask", set_pair_mask,
                nb::arg("i"), nb::arg("j"), nb::arg("mask"),
                 nb::arg("others")=nb::none(),
                doc_BasePairQuantity_setPairMask)
        .def("getPairMask", &PairQuantity::getPairMask,
                nb::arg("i"), nb::arg("j"),
                doc_BasePairQuantity_getPairMask)
        .def("setTypeMask", set_type_mask,
                nb::arg("tpi"), nb::arg("tpj"), nb::arg("mask"),
                 nb::arg("others")=nb::none(),
                doc_BasePairQuantity_setTypeMask)
        .def("getTypeMask", &PairQuantity::getTypeMask,
                nb::arg("tpi"), nb::arg("tpj"),
                doc_BasePairQuantity_getTypeMask)
        .def("ticker", &PairQuantity::ticker,
                nb::rv_policy::reference_internal,
                doc_BasePairQuantity_ticker)
        .def("copy", pqcopy,
                doc_BasePairQuantity_copy)
        ;

    nb::class_<PairQuantityExposed, PairQuantity, PairQuantityWrap> pq(m, "PairQuantity", doc_PairQuantity);
    pq
        .def(nb::init<>())
        .def("ticker",
                &PairQuantityExposed::ticker,
                nb::rv_policy::reference_internal,
                doc_PairQuantity_ticker)
        .def("_getParallelData", [](const PairQuantityExposed &pq) 
                {
                    return to_bytes(pq.getParallelData());
                },
                doc_PairQuantity__getParallelData)
        .def("_resizeValue",
                &PairQuantityExposed::resizeValue,
                nb::arg("sz"),
                doc_PairQuantity__resizeValue)
        .def("_resetValue",
                &PairQuantityExposed::resetValue,
                doc_PairQuantity__resetValue)
        .def("_configureBondGenerator",
                &PairQuantityExposed::configureBondGenerator,
                nb::arg("bnds"),
                doc_PairQuantity__configureBondGenerator)
        .def("_addPairContribution",
                &PairQuantityExposed::addPairContribution,
                nb::arg("bnds"), nb::arg("sumscale"),
                doc_PairQuantity__addPairContribution)
        .def("_executeParallelMerge",
                [](PairQuantityExposed &pq, nb::bytes pdata)
                {
                    pq.executeParallelMerge(from_bytes(pdata));
                },
                nb::arg("pdata"),
                doc_PairQuantity__executeParallelMerge)
        .def("_finishValue",
                &PairQuantityExposed::finishValue,
                doc_PairQuantity__finishValue)
        .def("_stashPartialValue",
                &PairQuantityExposed::stashPartialValue,
                doc_PairQuantity__stashPartialValue)
        .def("_restorePartialValue",
                &PairQuantityExposed::restorePartialValue,
                doc_PairQuantity__restorePartialValue)
        .def_prop_ro("_value", [](PairQuantityExposed &pq) -> QuantityType & 
                {
                    return pq.value();
                },
                nb::rv_policy::reference_internal,
                doc_PairQuantity__value)
        ;
        // classes PairQuantityExposed, PairQuantityWrap add no members,
        // therefore we can create pickle suite from C++ base class.
        PairQuantityPickleSuite<
            PairQuantity,
            DICT_PICKLE,
            PairQuantityExposed>::bind(pq);

}

}   //  namespace srrealmodule

// End of file
