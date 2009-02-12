// vim:set ft=cpp:

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


} // namespace mathutils
} // namespace diffpy

#endif  // MATHUTILS_IPP_INCLUDED
