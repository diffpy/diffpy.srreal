/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2006 Trustees of the Michigan State University.
*                   All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Common global variables, such as useful paths.
*
* $Id$
*
*****************************************************************************/

#ifndef GLOBALS_HPP_INCLUDED
#define GLOBALS_HPP_INCLUDED

#include <string>
#include <boost/filesystem/path.hpp>

const std::string& thisfile(const char* argzero=NULL);
const std::string& tests_dir();
const std::string& testdata_dir();

// Implementation ------------------------------------------------------------

inline
const std::string& thisfile(const char* argzero)
{
    using namespace std;
    static string rv(__FILE__);
    if (argzero)
    {
        using boost::filesystem::path;
        path p0(argzero);
        path ptf = p0.branch_path() /= path(__FILE__).leaf();
        rv = ptf.string();
    }
    return rv;
}


inline
const std::string& tests_dir()
{
    using namespace std;
    using boost::filesystem::path;
    static string rv;
    path ptd = path(thisfile()).branch_path();
    rv = ptd.string();
    return rv;
}


inline
const std::string& testdata_dir()
{
    using namespace std;
    using boost::filesystem::path;
    static string rv;
    path ptdd = path(tests_dir()) /= "testdata";
    rv = ptdd.string();
    return rv;
}

#endif  // GLOBALS_HPP_INCLUDED
