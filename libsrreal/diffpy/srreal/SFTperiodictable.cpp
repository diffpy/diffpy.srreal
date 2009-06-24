/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* class SFTperiodictableXray
* class SFTperiodictableNeutron
*
* Implementations of x-ray and neutron ScatteringFactorTable using Paul
* Kienzle periodictable library for Python.  The instances can be created
* using the createScatteringFactorTable factory, see the end of this file for
* available type strings.
*
* $Id$
*
*****************************************************************************/

#include <stdexcept>
#include <string>
#include <boost/python.hpp>

#include <diffpy/srreal/ScatteringFactorTable.hpp>
#include <diffpy/PythonInterface.hpp>

using namespace std;
namespace python = boost::python;

// Local helpers -------------------------------------------------------------

namespace {

// reference to the symbol method of periodictable.elements
python::object periodictable_elements_symbol()
{
    static bool did_import = false;
    static python::object symbol;
    // short-circuit return
    if (did_import)  return symbol;
    // first pass requires actual import
    diffpy::initializePython();
    python::object mod = python::import("periodictable");
    python::object elements = mod.attr("elements");
    symbol = elements.attr("symbol");
    did_import = true;
    return symbol;
}

}   // namespace

namespace diffpy {
namespace srreal {

//////////////////////////////////////////////////////////////////////////////
// class SFTperiodictableXray
//////////////////////////////////////////////////////////////////////////////

class SFTperiodictableXray : public ScatteringFactorTable
{
    public:

        // constructors

        SFTperiodictableXray()  { }


        ScatteringFactorTable* create() const
        {
            ScatteringFactorTable* rv = new SFTperiodictableXray();
            return rv;
        }


        ScatteringFactorTable* copy() const
        {
            ScatteringFactorTable* rv = new SFTperiodictableXray(*this);
            return rv;
        }

        // methods

        const string& type() const
        {
            static string rv = "SFTperiodictableXray";
            return rv;
        }


        const string& radiationType() const
        {
            static string rv = "X";
            return rv;
        }

    protected:

        // methods

        double fetch(const string& smbl) const
        {
            double rv;
            python::object symbol = periodictable_elements_symbol();
            try {
                python::object el = symbol(smbl);
                rv = python::extract<int>(el.attr("number"));
            }
            catch (python::error_already_set e) {
                string emsg = diffpy::getPythonErrorString();
                PyErr_Clear();
                throw invalid_argument(emsg);
            }
            return rv;
        }

};  // class SFTperiodictableXray

//////////////////////////////////////////////////////////////////////////////
// class SFTperiodictableNeutron
//////////////////////////////////////////////////////////////////////////////

class SFTperiodictableNeutron : public ScatteringFactorTable
{
    public:

        // constructors

        SFTperiodictableNeutron()  { }


        ScatteringFactorTable* create() const
        {
            ScatteringFactorTable* rv = new SFTperiodictableNeutron();
            return rv;
        }


        ScatteringFactorTable* copy() const
        {
            ScatteringFactorTable* rv = new SFTperiodictableNeutron(*this);
            return rv;
        }

        // methods

        const string& type() const
        {
            static string rv = "SFTperiodictableNeutron";
            return rv;
        }


        const string& radiationType() const
        {
            static string rv = "N";
            return rv;
        }

    protected:

        // methods

        double fetch(const string& smbl) const
        {
            double rv;
            python::object symbol = periodictable_elements_symbol();
            try {
                python::object el = symbol(smbl);
                python::object b_c = el.attr("neutron").attr("b_c");
                rv = python::extract<double>(b_c);
            }
            catch (python::error_already_set) {
                string emsg = diffpy::getPythonErrorString();
                PyErr_Clear();
                throw invalid_argument(emsg);
            }
            return rv;
        }

};  // class SFTperiodictableNeutron

// Registration --------------------------------------------------------------

bool reg_SFTperiodictableXray = (
        registerScatteringFactorTable(SFTperiodictableXray()) &&
        aliasScatteringFactorTable("SFTperiodictableXray", "X")
        );

bool reg_SFTperiodictableNeutron = (
        registerScatteringFactorTable(SFTperiodictableNeutron()) &&
        aliasScatteringFactorTable("SFTperiodictableNeutron", "N")
        );

}   // namespace srreal
}   // namespace diffpy

// End of file
