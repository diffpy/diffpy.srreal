/***********************************************************************
* $Id$
*
* Boost.python bindings to ProfileCalculator. 
***********************************************************************/
#include "profilecalculator.h"
#include "converters.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <string>
#include <iostream>

using namespace boost::python;
using namespace SrReal;

namespace {

class ProfileCalculatorWrap 
    : public ProfileCalculator, public wrapper<ProfileCalculator>
{

    public: 

    ProfileCalculatorWrap(
        BondIterator& _bonditer,
        BondWidthCalculator& _bwcalc) 
        : ProfileCalculator(_bonditer, _bwcalc) 
        {}

    // pure virtual functions
    float* getPDF()
    {
        return this->get_override("getPDF")();
    }

    PyObject* getPDFNdArray()
    {
        float* pdf = this->getPDF();
        size_t numpoints = this->getNumPoints();
        std::vector<int> dims(1, numpoints);
        return makeNdArray(pdf, dims);
    }

    float* getRDF()
    {
        return this->get_override("getRDF")();
    }

    PyObject* getRDFNdArray()
    {
        float* rdf = this->getRDF();
        size_t numpoints = this->getNumPoints();
        std::vector<int> dims(1, numpoints);
        return makeNdArray(rdf, dims);
    }

    // virtual functions
    
    BondIterator& default_getBondIterator()
    { return ProfileCalculator::getBondIterator(); }

    BondIterator& getBondIterator()
    {
        if (override getBondIterator 
                = this->get_override("getBondIterator")) 
        {
            return getBondIterator();
        }
        return default_getBondIterator();
    }
    
    BondWidthCalculator& default_getBondWidthCalculator()
    { return ProfileCalculator::getBondWidthCalculator(); }

    BondWidthCalculator& getBondWidthCalculator()
    {
        if (override getBondWidthCalculator 
                = this->get_override("getBondWidthCalculator")) 
        {
            return getBondWidthCalculator();
        }
        return default_getBondWidthCalculator();
    }

    void default_setScatType(const ObjCryst::RadiationType _radtype)
    { ProfileCalculator::setScatType(_radtype); }

    void setScatType(const ObjCryst::RadiationType _radtype)
    {
        if (override setScatType 
                = this->get_override("setScatType")) 
        {
            setScatType(_radtype);
            return;
        }
        default_setScatType(_radtype);
    }

    ObjCryst::RadiationType default_getScatType()
    { return ProfileCalculator::getScatType(); }

    ObjCryst::RadiationType getScatType()
    {
        if (override getScatType 
                = this->get_override("getScatType")) 
        {
            return getScatType();
        }
        return default_getScatType();
    }

    void default_setCalculationPoints(
            const float* _rvals, const size_t numpoints)
    { ProfileCalculator::setCalculationPoints(_rvals, numpoints); }

    void setCalculationPoints(
            const float* _rvals, const size_t numpoints)
    {
        if (override setCalculationPoints 
                = this->get_override("setCalculationPoints")) 
        {
            setCalculationPoints(_rvals, numpoints);
            return;
        }
        default_setCalculationPoints(_rvals, numpoints);
    }

    const float* default_getCalculationPoints()
    { return ProfileCalculator::getCalculationPoints(); }

    const float* getCalculationPoints()
    {
        if (override getCalculationPoints 
                = this->get_override("getCalculationPoints")) 
        {
            return getCalculationPoints();
        }
        return default_getCalculationPoints();
    }

    PyObject* getCalculationPointsNdArray()
    {
        const float* rvals = this->getCalculationPoints();
        size_t numpoints = this->getNumPoints();
        std::vector<int> dims(1, numpoints);
        return makeNdArray(const_cast<float*>(rvals), dims);
    }

    size_t default_getNumPoints()
    { return ProfileCalculator::getNumPoints(); }

    size_t getNumPoints()
    {
        if (override getNumPoints 
                = this->get_override("getNumPoints")) 
        {
            return getNumPoints();
        }
        return default_getNumPoints();
    }

    void default_setQmax(float val)
    { ProfileCalculator::setQmax(val); }

    void setQmax(float val)
    {
        if (override setQmax 
                = this->get_override("setQmax")) 
        {
            setQmax(val);
            return;
        }
        default_setQmax(val);
    }

    float default_getQmax()
    { return ProfileCalculator::getQmax(); }

    float getQmax()
    {
        if (override getQmax 
                = this->get_override("getQmax")) 
        {
            return getQmax();
        }
        return default_getQmax();
    }
   
    void default_setQmin(float val)
    { ProfileCalculator::setQmin(val); }

    void setQmin(float val)
    {
        if (override setQmin 
                = this->get_override("setQmin")) 
        {
            setQmin(val);
            return;
        }
        default_setQmin(val);
    }

    float default_getQmin()
    { return ProfileCalculator::getQmin(); }

    float getQmin()
    {
        if (override getQmin 
                = this->get_override("getQmin")) 
        {
            return getQmin();
        }
        return default_getQmin();
    }
   
}; // ProfileCalculatorWrap

} // anonymous namespace


BOOST_PYTHON_MODULE(_profilecalculator)
{

    class_<ProfileCalculatorWrap, boost::noncopyable, 
        bases<ObjCryst::RefinableObj> >
        ("ProfileCalculator", init<BondIterator&,BondWidthCalculator&>())
        .def("getBondIterator", &ProfileCalculator::getBondIterator, 
            &ProfileCalculatorWrap::default_getBondIterator,
            return_internal_reference<>())
        .def("getBondWidthCalculator", &ProfileCalculator::getBondWidthCalculator, 
            &ProfileCalculatorWrap::default_getBondWidthCalculator,
            return_internal_reference<>())
        .def("setScatType", &ProfileCalculator::setScatType, 
            &ProfileCalculatorWrap::default_setScatType)
        .def("getScatType", &ProfileCalculator::getScatType, 
            &ProfileCalculatorWrap::default_getScatType)
        .def("getCalculationPoints", &ProfileCalculatorWrap::getCalculationPointsNdArray)
        .def("setCalculationPoints", &ProfileCalculator::setCalculationPoints, 
            &ProfileCalculatorWrap::default_setCalculationPoints)
        .def("getNumPoints", &ProfileCalculator::getNumPoints, 
            &ProfileCalculatorWrap::default_getNumPoints)
        .def("setQmax", &ProfileCalculator::setQmax, 
            &ProfileCalculatorWrap::default_setQmax)
        .def("getQmax", &ProfileCalculator::getQmax, 
            &ProfileCalculatorWrap::default_getQmax)
        .def("setQmin", &ProfileCalculator::setQmin, 
            &ProfileCalculatorWrap::default_setQmin)
        .def("getQmin", &ProfileCalculator::getQmin, 
            &ProfileCalculatorWrap::default_getQmin)
        .def("getRDF", &ProfileCalculatorWrap::getRDFNdArray)
        .def("getPDF", &ProfileCalculatorWrap::getPDFNdArray)
        ;
}
