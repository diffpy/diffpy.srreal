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
* Various common mathematical constants and functions.
*
* $Id$
*
*****************************************************************************/

#ifndef MATHUTILS_IPP_INCLUDED
#define MATHUTILS_IPP_INCLUDED

#include <cmath>

namespace diffpy {
namespace mathutils {

inline
double cosd(double x)
{
    double xp = fmod(fabs(x), 360.0);
    if (remainder(xp, 60.0) == 0.0 || remainder(xp, 90.0) == 0.0)
    {
	switch(int(round(xp)))
	{
	    case 0: return 1.0;
	    case 60:
	    case 300: return 0.5;
	    case 90:
	    case 270: return 0.0;
	    case 120:
	    case 240: return -0.5;
	    case 180: return -1.0;
	};
    }
    return cos(x/180.0*M_PI);
}


inline
double sind(double x)
{
    return cosd(90.0 - x);
}


inline
double acosd(double x)
{
    if (remainder(x, 0.5) == 0.0)
    {
	switch(int(round(x/0.5)))
	{
	    case 0: return 90.0;
	    case 1: return 60.0;
	    case -1: return 120.0;
	    case 2: return 0.0;
	    case -2: return 180.0;
	};
    }
    return acos(x)/M_PI*180.0;
}


inline
double asind(double x)
{
    if (remainder(x, 0.5) == 0.0)
    {
	switch(int(round(x/0.5)))
	{
	    case 0: return 0.0;
	    case 1: return 30.0;
	    case -1: return -30.0;
	    case 2: return 90.0;
	    case -2: return -90.0;
	};
    }
    return acos(x)/M_PI*180.0;
}


inline
bool eps_eq(const double& x, const double& y, double eps)
{
    return fabs(x-y) < eps;
}


inline
bool eps_gt(const double& x, const double& y, double eps)
{
    return x > y + eps;
}


inline
bool eps_lt(const double& x, const double& y, double eps)
{
    return x < y - eps;
}


} // namespace mathutils
} // namespace diffpy

// vim:ft=cpp:

#endif  // MATHUTILS_IPP_INCLUDED
