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

#include <boost/python/class.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/python/copy_non_const_reference.hpp>
#include <boost/python/return_internal_reference.hpp>

#include "srreal_converters.hpp"
#include "srreal_pickling.hpp"

#include <diffpy/srreal/PairQuantity.hpp>

namespace srrealmodule {
namespace nswrap_PairQuantity {

using namespace boost;
using namespace boost::python;
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
Possible values are 'BASIC' and 'OPTIMIZED'.  The value is always\n\
calculated from scratch when 'BASIC'.  The 'OPTIMIZED' evaluation\n\
updates the existing results by recalculating only the contributions\n\
from changed atoms.\n\
\n\
See also evaluatortypeused.\n\
";

const char* doc_BasePairQuantity_evaluatortypeused = "\
String type of evaluation procedure used in the last calculation.\n\
\n\
Possible values are 'BASIC', 'OPTIMIZED', and 'NONE' for calculator\n\
that has not been used yet.\n\
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

// representation of QuantityType objects
python::object repr_QuantityType(const QuantityType& v)
{
    python::object rv = ("QuantityType%r" %
        python::make_tuple(python::tuple(v)));
    return rv;
}


// PairQuantity::eval is a template non-constant method and
// needs an explicit wrapper function.

python::object eval_asarray(PairQuantity& obj, python::object& a)
{
    QuantityType value = (Py_None == a.ptr()) ? obj.eval() : obj.eval(a);
    python::object rv = convertToNumPyArray(value);
    return rv;
}

// support for the evaluatortype property

const char* evtp_NONE = "NONE";
const char* evtp_BASIC = "BASIC";
const char* evtp_OPTIMIZED = "OPTIMIZED";

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
    }
    std::string emsg = "Unknown internal value of PQEvaluatorType.";
    throw std::out_of_range(emsg);
}


std::string getevaluatortype(const PairQuantity& obj)
{
    return stringevaluatortype(obj.getEvaluatorType());
}


void setevaluatortype(PairQuantity& pq, const std::string& tp)
{
    if (tp == evtp_BASIC)  return pq.setEvaluatorType(BASIC);
    if (tp == evtp_OPTIMIZED)  return pq.setEvaluatorType(OPTIMIZED);
    python::object emsg = ("evaluatortype must be either %r or %r." %
            python::make_tuple(evtp_BASIC, evtp_OPTIMIZED));
    PyErr_SetObject(PyExc_ValueError, emsg.ptr());
    throw_error_already_set();
}

// support for the evaluatortypeused read-only property

std::string getevaluatortypeused(const PairQuantity& obj)
{
    return stringevaluatortype(obj.getEvaluatorTypeUsed());
}

// support "all", "ALL" and integer iterables in setPairMask

std::vector<int> parsepairindex(python::object i)
{
    std::vector<int> rv;
    // string equal "all" or "ALL"
    python::extract<std::string> gets(i);
    if (gets.check())
    {
        python::str lc_all(PairQuantity::ALLATOMSSTR);
        python::str uc_all = lc_all.upper();
        if (i != lc_all && i != uc_all)
        {
            python::object emsg = ("String argument must be %r or %r." %
                 python::make_tuple(lc_all, uc_all));
            PyErr_SetObject(PyExc_ValueError, emsg.ptr());
            throw_error_already_set();
        }
        rv.push_back(PairQuantity::ALLATOMSINT);
        return rv;
    }
    // otherwise translate to a vector of integers
    rv = extractintvector(i);
    return rv;
}

// support string iterables in setTypeMask

std::vector<std::string> parsepairtypes(
        python::extract<std::string>& getsmbli, python::object smbli)
{
    std::vector<std::string> rv;
    if (getsmbli.check())
    {
        rv.push_back(getsmbli());
    }
    else
    {
        python::stl_input_iterator<std::string> first(smbli), last;
        rv.assign(first, last);
    }
    return rv;
}


void mask_all_pairs(PairQuantity& obj, python::object msk)
{
    bool mask = msk;
    obj.maskAllPairs(mask);
}


void set_pair_mask(PairQuantity& obj,
        python::object i, python::object j, python::object msk,
        python::object others)
{
    if (Py_None != others.ptr())  mask_all_pairs(obj, others);
    python::extract<int> geti(i);
    python::extract<int> getj(j);
    bool mask = msk;
    // short circuit for normal call
    if (geti.check() && getj.check())
    {
        obj.setPairMask(geti(), getj(), mask);
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
        python::object smbli, python::object smblj, python::object msk,
        python::object others)
{
    using namespace std;
    if (Py_None != others.ptr())  mask_all_pairs(obj, others);
    python::extract<string> getsmbli(smbli);
    python::extract<string> getsmblj(smblj);
    bool mask = msk;
    // short circuit for normal call
    if (getsmbli.check() && getsmblj.check())
    {
        obj.setTypeMask(getsmbli(), getsmblj(), mask);
        return;
    }
    vector<string> isymbols = parsepairtypes(getsmbli, smbli);
    vector<string> jsymbols = parsepairtypes(getsmblj, smblj);
    vector<string>::const_iterator tii, tjj;
    for (tii = isymbols.begin(); tii != isymbols.end(); ++tii)
    {
        for (tjj = jsymbols.begin(); tjj != jsymbols.end(); ++tjj)
        {
            obj.setTypeMask(*tii, *tjj, mask);
        }
    }
}


// provide a copy method for convenient deepcopy of the object

python::object pqcopy(python::object pqobj)
{
    python::object copy = python::import("copy").attr("copy");
    python::object rv = copy(pqobj);
    return rv;
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
    public wrapper<PairQuantityExposed>
{
    public:

        // Make getParallelData overridable from Python.

        std::string getParallelData() const
        {
            override f = this->get_override("_getParallelData");
            if (f)  return f();
            return this->default_getParallelData();
        }

        std::string default_getParallelData() const
        {
            return this->PairQuantityExposed::getParallelData();
        }

        // Make the ticker method overridable from Python

        diffpy::eventticker::EventTicker& ticker() const
        {
            using diffpy::eventticker::EventTicker;
            override f = this->get_override("ticker");
            if (f)
            {
                // avoid "dangling reference error" when used from C++
                object ptic = f();
                return extract<EventTicker&>(ptic);
            }
            return this->default_ticker();
        }

        diffpy::eventticker::EventTicker& default_ticker() const
        {
            return this->PairQuantityExposed::ticker();
        }

        // Make the protected virtual methods public so they
        // can be exported to Python and overridden as well.

        void resizeValue(size_t sz)
        {
            override f = this->get_override("_resizeValue");
            if (f)  f(sz);
            else    this->default_resizeValue(sz);
        }

        void default_resizeValue(size_t sz)
        {
            this->PairQuantityExposed::resizeValue(sz);
        }


        void resetValue()
        {
            override f = this->get_override("_resetValue");
            if (f)  f();
            else    this->default_resetValue();
        }

        void default_resetValue()
        {
            this->PairQuantityExposed::resetValue();
        }


        void configureBondGenerator(BaseBondGenerator& bnds) const
        {
            override f = this->get_override("_configureBondGenerator");
            if (f)  f(ptr(&bnds));
            else    this->default_configureBondGenerator(bnds);
        }

        void default_configureBondGenerator(BaseBondGenerator& bnds) const
        {
            this->PairQuantityExposed::configureBondGenerator(bnds);
        }


        void addPairContribution(const BaseBondGenerator& bnds,
                int summationscale)
        {
            override f = this->get_override("_addPairContribution");
            if (f)  f(ptr(&bnds), summationscale);
            else    this->default_addPairContribution(bnds, summationscale);
        }

        void default_addPairContribution(const BaseBondGenerator& bnds,
                int summationscale)
        {
            this->PairQuantityExposed::addPairContribution(bnds, summationscale);
        }


        void executeParallelMerge(const std::string& pdata)
        {
            override f = this->get_override("_executeParallelMerge");
            if (f)  f(pdata);
            else    this->default_executeParallelMerge(pdata);
        }

        void default_executeParallelMerge(const std::string& pdata)
        {
            this->PairQuantityExposed::executeParallelMerge(pdata);
        }


        void finishValue()
        {
            override f = this->get_override("_finishValue");
            if (f)  f();
            else    this->default_finishValue();
        }

        void default_finishValue()
        {
            this->PairQuantityExposed::finishValue();
        }


        void stashPartialValue()
        {
            override f = this->get_override("_stashPartialValue");
            if (f)  f();
            else    this->default_stashPartialValue();
        }

        void default_stashPartialValue()
        {
            this->PairQuantityExposed::stashPartialValue();
        }


        void restorePartialValue()
        {
            override f = this->get_override("_restorePartialValue");
            if (f)  f();
            else    this->default_restorePartialValue();
        }

        void default_restorePartialValue()
        {
            this->PairQuantityExposed::restorePartialValue();
        }

};  // class PairQuantityWrap

}   // namespace nswrap_PairQuantity

// Wrapper definition --------------------------------------------------------

void wrap_PairQuantity()
{
    using namespace nswrap_PairQuantity;
    using diffpy::Attributes;
    const python::object None;

    typedef StructureAdapterPtr&(PairQuantity::*getstru)();

    class_<QuantityType>("QuantityType")
        .def(vector_indexing_suite<QuantityType>())
        .def("__repr__", repr_QuantityType)
        ;

    class_<PairQuantity, bases<Attributes> >("BasePairQuantity")
        .def("eval", eval_asarray, python::arg("stru")=None,
                doc_BasePairQuantity_eval)
        .add_property("value", value_asarray<PairQuantity>,
                doc_BasePairQuantity_value)
        .def("_mergeParallelData", &PairQuantity::mergeParallelData,
                (python::arg("pdata"), python::arg("ncpu")),
                doc_BasePairQuantity__mergeParallelData)
        .def("_getParallelData", &PairQuantity::getParallelData,
                doc_BasePairQuantity__getParallelData)
        .def("setStructure", &PairQuantity::setStructure<object>,
                python::arg("stru"),
                doc_BasePairQuantity_setStructure)
        .def("getStructure", getstru(&PairQuantity::getStructure),
                return_value_policy<copy_non_const_reference>(),
                doc_BasePairQuantity_getStructure)
        .def("_setupParallelRun", &PairQuantity::setupParallelRun,
                (python::arg("cpuindex"), python::arg("ncpu")),
                doc_BasePairQuantity__setupParallelRun)
        .add_property("evaluatortype",
                getevaluatortype, setevaluatortype,
                doc_BasePairQuantity_evaluatortype)
        .add_property("evaluatortypeused",
                getevaluatortypeused,
                doc_BasePairQuantity_evaluatortypeused)
        .def("maskAllPairs", mask_all_pairs,
                python::arg("mask"),
                doc_BasePairQuantity_maskAllPairs)
        .def("invertMask", &PairQuantity::invertMask,
                doc_BasePairQuantity_invertMask)
        .def("setPairMask", set_pair_mask,
                (python::arg("i"), python::arg("j"), python::arg("mask"),
                 python::arg("others")=None),
                doc_BasePairQuantity_setPairMask)
        .def("getPairMask", &PairQuantity::getPairMask,
                (python::arg("i"), python::arg("j")),
                doc_BasePairQuantity_getPairMask)
        .def("setTypeMask", set_type_mask,
                (python::arg("tpi"), python::arg("tpj"), python::arg("mask"),
                 python::arg("others")=None),
                doc_BasePairQuantity_setTypeMask)
        .def("getTypeMask", &PairQuantity::getTypeMask,
                (python::arg("tpi"), python::arg("tpj")),
                doc_BasePairQuantity_getTypeMask)
        .def("ticker", &PairQuantity::ticker,
                return_internal_reference<>(),
                doc_BasePairQuantity_ticker)
        .def("copy", pqcopy,
                doc_BasePairQuantity_copy)
        .def_pickle(PairQuantityPickleSuite<PairQuantity>())
        ;

    class_<PairQuantityWrap, bases<PairQuantity>,
        noncopyable>("PairQuantity", doc_PairQuantity)
        .def("ticker",
                &PairQuantityExposed::ticker,
                &PairQuantityWrap::default_ticker,
                return_internal_reference<>(),
                doc_PairQuantity_ticker)
        .def("_getParallelData",
                &PairQuantityExposed::getParallelData,
                &PairQuantityWrap::default_getParallelData,
                doc_PairQuantity__getParallelData)
        .def("_resizeValue",
                &PairQuantityExposed::resizeValue,
                &PairQuantityWrap::default_resizeValue,
                python::arg("sz"),
                doc_PairQuantity__resizeValue)
        .def("_resetValue",
                &PairQuantityExposed::resetValue,
                &PairQuantityWrap::default_resetValue,
                doc_PairQuantity__resetValue)
        .def("_configureBondGenerator",
                &PairQuantityExposed::configureBondGenerator,
                &PairQuantityWrap::default_configureBondGenerator,
                python::arg("bnds"),
                doc_PairQuantity__configureBondGenerator)
        .def("_addPairContribution",
                &PairQuantityExposed::addPairContribution,
                &PairQuantityWrap::default_addPairContribution,
                (python::arg("bnds"), python::arg("sumscale")),
                doc_PairQuantity__addPairContribution)
        .def("_executeParallelMerge",
                &PairQuantityExposed::executeParallelMerge,
                &PairQuantityWrap::default_executeParallelMerge,
                python::arg("pdata"),
                doc_PairQuantity__executeParallelMerge)
        .def("_finishValue",
                &PairQuantityExposed::finishValue,
                &PairQuantityWrap::default_finishValue,
                doc_PairQuantity__finishValue)
        .def("_stashPartialValue",
                &PairQuantityExposed::stashPartialValue,
                &PairQuantityWrap::default_stashPartialValue,
                doc_PairQuantity__stashPartialValue)
        .def("_restorePartialValue",
                &PairQuantityExposed::restorePartialValue,
                &PairQuantityWrap::default_restorePartialValue,
                doc_PairQuantity__restorePartialValue)
        .add_property("_value", make_function(&PairQuantityWrap::value,
                    return_internal_reference<>()),
                doc_PairQuantity__value)
        ;

}

}   //  namespace srrealmodule

// End of file
