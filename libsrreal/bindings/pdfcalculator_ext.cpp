/***********************************************************************
* $Id$
*
* Boost.python bindings to PDFCalculator. 
***********************************************************************/
#include "pdfcalculator.h"

#include <boost/utility.hpp>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

using namespace boost::python;
using namespace SrReal;


BOOST_PYTHON_MODULE(_pdfcalculator)
{

    class_<PDFCalculator, bases<ProfileCalculator> >
        ("PDFCalculator", init<BondIterator&,BondWidthCalculator&>())
        ;
}
