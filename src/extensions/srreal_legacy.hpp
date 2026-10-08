// Archive-only stand-ins for the Boost.Python component wrapper types.
// See LICENSE.rst for license information.
#ifndef SRREAL_LEGACY_HPP_INCLUDED
#define SRREAL_LEGACY_HPP_INCLUDED

#include <diffpy/srreal/forwardtypes.hpp>
#include <boost/serialization/base_object.hpp>
#include <stdexcept>
#include <string>

namespace srrealmodule {

// Old calculator archives contain both a serialized C++ wrapper and the
// actual Python component in a separate pickle state item. Boost constructs
// the former without a Python instance, which is invalid for an NB_TRAMPOLINE.
// Read its base data into this temporary, then replace it with the separately
// unpickled Python object in the calculator's __setstate__.
template <class T>
class LegacyComponent : public T
{
    public:
        typename T::SharedPtr create() const { unavailable(); }
        typename T::SharedPtr clone() const { unavailable(); }
        const std::string& type() const { unavailable(); }
        const std::string& radiationType() const { unavailable(); }
        double operator()(const double&) const { unavailable(); }
        double operator()(double, double) const { unavailable(); }
        double xboundlo(double) const { unavailable(); }
        double xboundhi(double) const { unavailable(); }
        double standardLookup(const std::string&) const { unavailable(); }
        double standardLookup(const std::string&, double) const { unavailable(); }
        double calculate(const diffpy::srreal::BaseBondGenerator&) const { unavailable(); }
        double maxWidth(diffpy::srreal::StructureAdapterPtr, double, double) const
        {
            unavailable();
        }

    private:
        [[noreturn]] static void unavailable()
        {
            throw std::runtime_error("legacy Python component was not restored");
        }

        friend class boost::serialization::access;
        template <class Archive>
        void serialize(Archive& ar, const unsigned int)
        {
            ar & boost::serialization::base_object<T>(*this);
        }
};

}  // namespace srrealmodule

#endif
