#ifndef MATHUTILS_HPP_INCLUDED
#define MATHUTILS_HPP_INCLUDED

#include <limits>

namespace diffpy {
namespace mathutils {

// constants

const double DOUBLE_MAX = std::numeric_limits<double>().max();
const double DOUBLE_EPS = std::numeric_limits<double>().epsilon();

// trigonometric functions with more exact values at n*30 degrees

double cosd(double);
double sind(double x);
double acosd(double x);
double asind(double x);

}   // namespace mathutils
}   // namespace diffpy

// Implementation ------------------------------------------------------------

#include <diffpy/mathutils.ipp>

#endif  // MATHUTILS_HPP_INCLUDED
