/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   (c) 2013 Brookhaven Science Associates,
*                   Brookhaven National Laboratory.
*                   All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Bindings to the EventTicker class.
*
*****************************************************************************/

#include <boost/python/class.hpp>

#include <diffpy/EventTicker.hpp>

#include "srreal_pickling.hpp"

namespace srrealmodule {
namespace nswrap_EventTicker {

using namespace boost;
using diffpy::eventticker::EventTicker;

// docstrings ----------------------------------------------------------------

const char* doc_EventTicker = "\
Class for storing modification 'times' of dependent objects\n\
The default EventTicker object is initialized to a zero time.\n\
";

const char* doc_EventTicker_cp = "\
Constructor EventTicker at the same modification time as the source.\n\
";

const char* doc_EventTicker___repr__ = "\
String representation of the EventTicker object\n\
";

const char* doc_EventTicker_click = "\
Increment ticker value to the latest unique time.\n\
";

const char* doc_EventTicker_updateFrom = "\
Update ticker time to the value of other, newer ticker.\n\
Keep original value if the other ticker is older.\n\
\n\
other    -- instance of another EventTicker object\n\
\n\
No return value.\n\
";

const char* doc_EventTicker__value = "\
Return the internal time for this ticker.  This is a tuple of 2 integers,\n\
where the latter is the total number of click calls at the last update\n\
and the first one is zero unless there was an integer overflow.\n\
";

// wrappers ------------------------------------------------------------------

// getter for the _value property

python::tuple gettickervalue(const EventTicker& tc)
{
    EventTicker::value_type v = tc.value();
    return python::make_tuple(v.first, v.second);
}

// representation of EventTicker objects

python::object repr_EventTicker(const EventTicker& tc)
{
    python::object rv = ("EventTicker(%i, %i)" % gettickervalue(tc));
    return rv;
}

}   // namespace nswrap_EventTicker

// Wrapper definition --------------------------------------------------------

void wrap_EventTicker()
{
    using namespace nswrap_EventTicker;
    using namespace boost::python;

    class_<EventTicker>("EventTicker", doc_EventTicker)
        .def(init<const EventTicker&>(doc_EventTicker_cp))
        .def("__repr__", repr_EventTicker, doc_EventTicker___repr__)
        .def(self < self)
        .def(self <= self)
        .def(self > self)
        .def(self >= self)
        .def(self == self)
        .def(self != self)
        .def("click", &EventTicker::click,
                doc_EventTicker_click)
        .def("updateFrom", &EventTicker::updateFrom,
                python::arg("other"),
                doc_EventTicker_updateFrom)
        .add_property("_value", gettickervalue,
                doc_EventTicker__value)
        .def_pickle(SerializationPickleSuite<EventTicker,DICT_IGNORE>())
        ;

}

}   // namespace srrealmodule

// End of file
