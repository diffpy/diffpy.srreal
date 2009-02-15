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
* class TestVR3Structure -- unit tests for using simple structure class
*     VR3Structure with pair quantity calculators
*
* $Id$
*
*****************************************************************************/

#include <signal.h>
#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

#include <boost/filesystem/path.hpp>
#include <boost/python.hpp>

#include <diffpy/srreal/DiffPyStructureAdapter.hpp>
#include "globals.hpp"

using namespace std;
using namespace boost;
using namespace diffpy::srreal;


class TestDiffPyStructureAdapter : public CppUnit::TestFixture
{

    CPPUNIT_TEST_SUITE(TestDiffPyStructureAdapter);
    CPPUNIT_TEST(test_countSites);
    CPPUNIT_TEST_SUITE_END();

private:

    auto_ptr<DiffPyStructureAdapter> m_ni;

public:

    void setUp()
    {
        try {
            this->initializePython();
            if (!m_ni.get())
            {
                python::object stru;
                stru = this->loadTestStructure("Ni.cif");
                m_ni.reset(new DiffPyStructureAdapter(stru));
            }
        }
        catch (boost::python::error_already_set) {
            if (PyErr_Occurred())   PyErr_Print();
            exit(2);
        }
    }


    void test_countSites()
    {
        try {
            CPPUNIT_ASSERT_EQUAL(4, m_ni->countSites());
        }
        catch (python::error_already_set) {
            if (PyErr_Occurred())   PyErr_Print();
        }
    }

private:

    void initializePython() const
    {  
        if (Py_IsInitialized())  return;
        Py_Initialize();
        static int py_argc = 1;
        static char py_arg0[7] = "python";
        static char* py_argv[] = {py_arg0};
        PySys_SetArgv(py_argc, py_argv);
        // Make sure Python does not eat SIGINT.
        signal(SIGINT, SIG_DFL);
    }


    python::object loadTestStructure(const string& tailname) const
    {
        using boost::filesystem::path;
        path fp = path(testdata_dir()) /= tailname;
        python::object stru = this->newDiffPyStructure();
        stru.attr("read")(fp.string());
        // python::call_method<void>(stru.ptr(), "read", filename);
        return stru;
    }


    python::object newDiffPyStructure() const
    {
        python::object mod = python::import("diffpy.Structure");
        python::object stru = mod.attr("Structure")();
        return stru;
    }

};

// Registers the fixture into the 'registry'
CPPUNIT_TEST_SUITE_REGISTRATION(TestDiffPyStructureAdapter);

// End of file
