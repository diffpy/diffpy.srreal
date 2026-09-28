// Custom virtual dispatch for cached reference returns and argument conversion.
// See LICENSE.rst for license information.
#ifndef SRREAL_TRAMPOLINE_HPP_INCLUDED
#define SRREAL_TRAMPOLINE_HPP_INCLUDED

#include <nanobind/trampoline.h>

namespace srrealmodule {

// Most overrides use NB_OVERRIDE directly. The overrides that need custom
// conversion must retain a ticket for the entire Python call (including
// recursion detection). Isolate the nanobind 2/3 ABI difference here.
class OverrideTicket : public nanobind::detail::ticket
{
    public:
        template <class Trampoline>
        OverrideTicket(const Trampoline& trampoline, const char* name, bool pure)
            : nanobind::detail::ticket(trampoline, name,
#if NB_VERSION_MAJOR >= 3
                    nanobind::detail::str_hash(name),
#endif
                    pure)
        { }

        OverrideTicket(const OverrideTicket&) = delete;
        OverrideTicket& operator=(const OverrideTicket&) = delete;
};

}  // namespace srrealmodule

#endif
