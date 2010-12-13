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

#include <string>
#include <sstream>
#include <boost/python.hpp>
#include <diffpy/serialization.hpp>

namespace srrealmodule {

template <class T>
class SerializationPickleSuite : public boost::python::pickle_suite
{
    public:

        static boost::python::tuple getstate(boost::python::object pqobj)
        {
            using namespace std;
            const T& pq = boost::python::extract<const T&>(pqobj);
            ostringstream storage(ios::binary);
            diffpy::serialization::oarchive oa(storage, ios::binary);
            oa << pq;
            boost::python::tuple rv = boost::python::make_tuple(
                    storage.str(), pqobj.attr("__dict__"));
            return rv;
        }


        static void setstate(
                boost::python::object pqobj, boost::python::tuple state)
        {
            using namespace std;
            using namespace boost::python;
            T& pq = extract<T&>(pqobj);
            if (len(state) != 2)
            {
                object emsg = ("expected 2-item tuple in call "
                        "to __setstate__; got %s" % state);
                PyErr_SetObject(PyExc_ValueError, emsg.ptr());
                throw_error_already_set();
            }
            // load the C++ object
            string content = extract<string>(state[0]);
            istringstream storage(content, ios::binary);
            diffpy::serialization::iarchive ia(storage, ios::binary);
            ia >> pq;
            // restore the object's __dict__
            dict d = extract<dict>(pqobj.attr("__dict__"));
            d.update(state[1]);
        }

        static bool getstate_manages_dict()  { return true; }

};  // class SerializationPickleSuite

}   // namespace srrealmodule

#endif  // SRREAL_PICKLING_HPP_INCLUDED
