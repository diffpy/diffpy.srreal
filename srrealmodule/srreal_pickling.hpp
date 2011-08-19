/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Pickling support that uses serialization of the libdiffpy classes.
*
* $Id$
*
*****************************************************************************/

#ifndef SRREAL_PICKLING_HPP_INCLUDED
#define SRREAL_PICKLING_HPP_INCLUDED

#include <boost/python.hpp>
#include <string>
#include <sstream>

#include <diffpy/serialization.hpp>

namespace srrealmodule {

template <class T>
std::string serialization_tostring(const T& tobj)
{
    using namespace std;
    ostringstream storage(ios::binary);
    diffpy::serialization::oarchive oa(storage, ios::binary);
    oa << tobj;
    return storage.str();
}


template <class T>
void serialization_fromstring(T& tobj, const std::string& s)
{
    using namespace std;
    istringstream storage(s, ios::binary);
    diffpy::serialization::iarchive ia(storage, ios::binary);
    ia >> tobj;
}


inline
void ensure_tuple_length(boost::python::tuple state, const int statelen)
{
    using namespace boost::python;
    if (len(state) == statelen)  return;
    object emsg = ("expected %i-item tuple in call to "
            "__setstate__; got %s" % make_tuple(statelen, state));
    PyErr_SetObject(PyExc_ValueError, emsg.ptr());
    throw_error_already_set();
}

enum {DICT_IGNORE=false, DICT_PICKLE=true};

template <class T, bool pickledict=DICT_PICKLE>
class SerializationPickleSuite : public boost::python::pickle_suite
{
    public:

        static boost::python::tuple getstate(boost::python::object obj)
        {
            using namespace std;
            const T& tobj = boost::python::extract<const T&>(obj);
            string content = serialization_tostring(tobj);
            boost::python::tuple rv = pickledict ?
                boost::python::make_tuple(content, obj.attr("__dict__")) :
                boost::python::make_tuple(content);
            return rv;
        }


        static void setstate(
                boost::python::object obj, boost::python::tuple state)
        {
            using namespace std;
            using namespace boost::python;
            T& tobj = extract<T&>(obj);
            int statelen = pickledict ? 2 : 1;
            ensure_tuple_length(state, statelen);
            // load the C++ object
            string content = extract<string>(state[0]);
            serialization_fromstring(tobj, content);
            // restore the object's __dict__
            if (pickledict)
            {
                dict d = extract<dict>(obj.attr("__dict__"));
                d.update(state[1]);
            }
        }

        static bool getstate_manages_dict()  { return pickledict; }

};  // class SerializationPickleSuite

}   // namespace srrealmodule

#endif  // SRREAL_PICKLING_HPP_INCLUDED
