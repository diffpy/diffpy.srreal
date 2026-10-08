// Custom virtual dispatch for cached reference returns and argument conversion.
// See LICENSE.rst for license information.
#ifndef SRREAL_TRAMPOLINE_HPP_INCLUDED
#define SRREAL_TRAMPOLINE_HPP_INCLUDED

#include <nanobind/trampoline.h>

namespace srrealmodule {

// Most overrides use NB_OVERRIDE directly. The overrides that need custom
// conversion must retain a ticket for the entire Python call (including
// recursion detection). Keep the low-level dispatch API in one place.
class OverrideTicket : public nanobind::detail::ticket
{
    public:
        OverrideTicket(const nanobind::detail::trampoline& trampoline,
                const char* name, bool pure)
            : nanobind::detail::ticket(trampoline, name,
                    nanobind::detail::str_hash(name),
                    pure)
        { }

        OverrideTicket(const OverrideTicket&) = delete;
        OverrideTicket& operator=(const OverrideTicket&) = delete;
};

}  // namespace srrealmodule

#endif
