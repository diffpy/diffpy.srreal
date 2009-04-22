/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Christopher Farrow, Pavol Juhas
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
#include <boost/python.hpp>

#include <diffpy/srreal/ScatteringFactorTable.hpp>
#include <diffpy/PythonInterface.hpp>

using namespace std;
using namespace diffpy::srreal;
namespace python = boost::python;

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

    protected:

        // methods

        double fetch(const string& smbl) const
        {
            double rv;
            diffpy::initializePython();
            python::object mod = python::import("periodictable");
            python::object elements = mod.attr("elements");
            try {
                python::object el = elements.attr("symbol")(smbl);
                rv = python::extract<int>(el.attr("number"));
            }
            catch (python::error_already_set e) {
                if (PyErr_Occurred())   PyErr_Clear();
                const char* emsg = "Invalid atom type.";
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

    protected:

        // methods

        double fetch(const string& smbl) const
        {
            double rv;
            diffpy::initializePython();
            python::object mod = python::import("periodictable");
            python::object elements = mod.attr("elements");
            try {
                python::object el = elements.attr("symbol")(smbl);
                python::object b_c = el.attr("neutron").attr("b_c");
                rv = python::extract<double>(b_c);
            }
            catch (python::error_already_set) {
                if (PyErr_Occurred())   PyErr_Clear();
                const char* emsg = "Invalid atom type.";
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

// End of file
