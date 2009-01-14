/***********************************************************************
* $Id$
*
* Boost.python bindings to PDFCalculator. 
***********************************************************************/
#include "pdfcalculator.h"
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

class PDFCalculatorWrap 
    : public PDFCalculator, public wrapper<PDFCalculator>
{

    public: 

    PDFCalculatorWrap(
        BondIterator& _bonditer,
        BondWidthCalculator& _bwcalc) 
        : PDFCalculator(_bonditer, _bwcalc) 
        {}

    // virtual functions
    
    void default_setCalculationPoints(
            const float* _rvals, const size_t numpoints)
    { PDFCalculator::setCalculationPoints(_rvals, numpoints); }

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

    void default_setQmax(float val)
    { PDFCalculator::setQmax(val); }

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

    void default_setQmin(float val)
    { PDFCalculator::setQmin(val); }

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

    float* default_getRDF()
    { return PDFCalculator::getRDF(); }

    float* getRDF()
    {
        if (override getRDF 
                = this->get_override("getRDF")) 
        {
            return getRDF();
        }
        return default_getRDF();
    }

    PyObject* getRDFNdArray()
    {
        float* rdf = this->getRDF();
        size_t numpoints = this->getNumPoints();
        std::vector<int> dims(1, numpoints);
        return makeNdArray(rdf, dims);
    }

    float* default_getPDF()
    { return PDFCalculator::getPDF(); }

    float* getPDF()
    {
        if (override getPDF 
                = this->get_override("getPDF")) 
        {
            return getPDF();
        }
        return default_getPDF();
    }

    PyObject* getPDFNdArray()
    {
        float* pdf = this->getPDF();
        size_t numpoints = this->getNumPoints();
        std::vector<int> dims(1, numpoints);
        return makeNdArray(pdf, dims);
    }

   
}; // PDFCalculatorWrap

} // anonymous namespace


BOOST_PYTHON_MODULE(_pdfcalculator)
{

    class_<PDFCalculatorWrap, boost::noncopyable, 
        bases<ProfileCalculator> >
        ("PDFCalculator", init<BondIterator&,BondWidthCalculator&>())
        .def("setCalculationPoints", &PDFCalculator::setCalculationPoints, 
            &PDFCalculatorWrap::default_setCalculationPoints)
        .def("setQmax", &PDFCalculator::setQmax, 
            &PDFCalculatorWrap::default_setQmax)
        .def("setQmin", &PDFCalculator::setQmin, 
            &PDFCalculatorWrap::default_setQmin)
        .def("getRDF", &PDFCalculatorWrap::getRDFNdArray)
        .def("getPDF", &PDFCalculatorWrap::getPDFNdArray)
        ;
}
