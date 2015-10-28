/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE_DANSE.txt for license information.
*
******************************************************************************
*
* Pickling support that uses serialization of the libdiffpy classes.
*
*****************************************************************************/

#ifndef SRREAL_PICKLING_HPP_INCLUDED
#define SRREAL_PICKLING_HPP_INCLUDED

#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/str.hpp>
#include <boost/python/operators.hpp>

#include <string>
#include <sstream>

#include <diffpy/serialization.hpp>
#include <diffpy/srreal/forwardtypes.hpp>

namespace srrealmodule {

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
            string content = diffpy::serialization_tostring(tobj);
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
            diffpy::serialization_fromstring(tobj, content);
            // restore the object's __dict__
            if (pickledict)
            {
                dict d = extract<dict>(obj.attr("__dict__"));
                d.update(state[1]);
            }
        }

        static bool getstate_manages_dict()  { return pickledict; }

};  // class SerializationPickleSuite


template <class T, bool pickledict=DICT_PICKLE>
class PairQuantityPickleSuite :
    public SerializationPickleSuite<T, pickledict>
{
    private:

        typedef SerializationPickleSuite<T, pickledict> Super;

    public:

        static boost::python::tuple getstate(boost::python::object obj)
        {
            using namespace boost::python;
            using namespace diffpy::srreal;
            // store the original structure object
            object stru = obj.attr("getStructure")();
            // temporarily remove structure from the pair quantity
            T& pq = extract<T&>(obj);
            StructureAdapterPtr pstru =
                replacePairQuantityStructure(pq, StructureAdapterPtr());
            object state0 = Super::getstate(obj);
            // restore the original structure
            replacePairQuantityStructure(pq, pstru);
            tuple rv = make_tuple(state0, stru);
            return rv;
        }


        static void setstate(
                boost::python::object obj, boost::python::tuple state)
        {
            using namespace boost::python;
            using namespace diffpy::srreal;
            ensure_tuple_length(state, 2);
            // restore the state using boost serialization
            tuple st0 = extract<tuple>(state[0]);
            Super::setstate(obj, st0);
            // restore the structure object
            StructureAdapterPtr pstru = extract<StructureAdapterPtr>(state[1]);
            T& pq = extract<T&>(obj);
            replacePairQuantityStructure(pq, pstru);
        }

};  // class PairQuantityPickleSuite


template <class T>
class StructureAdapterPickleSuite : public boost::python::pickle_suite
{
    public:

        static boost::python::tuple getinitargs(
                diffpy::srreal::StructureAdapterPtr adpt)
        {
            using namespace boost;
            python::tuple rv;
            // if adapter has been created from Python, we can use the default
            // Python constructor, i.e., __init__ with no arguments.
            if (frompython(adpt))  return rv;
            // otherwise the instance is from a non-wrapped C++ adapter,
            // and we need to reconstruct it using boost::serialization
            std::string content = diffpy::serialization_tostring(adpt);
            rv = python::make_tuple(content);
            return rv;
        }


        static boost::python::tuple getstate(boost::python::object obj)
        {
            using namespace boost;
            using namespace std;
            using diffpy::srreal::StructureAdapterPtr;
            StructureAdapterPtr adpt =
                python::extract<StructureAdapterPtr>(obj);
            python::object content;
            // Store serialization data for a Python-built object
            if (frompython(adpt))
            {
                const T& tobj = boost::python::extract<const T&>(obj);
                content = python::str(diffpy::serialization_tostring(tobj));
            }
            python::tuple rv =
                python::make_tuple(content, obj.attr("__dict__"));
            return rv;
        }


        static void setstate(
                boost::python::object obj, boost::python::tuple state)
        {
            using namespace std;
            using namespace boost::python;
            ensure_tuple_length(state, 2);
            // Restore the C++ data from state[0] for Python built-objects.
            // state[0] is None for C++ objects and there is no need to do
            // anything as those were already restored by string constructor.
            object st0 = state[0];
            if (st0.ptr() != Py_None)
            {
                T& tobj = extract<T&>(obj);
                string content = extract<string>(state[0]);
                diffpy::serialization_fromstring(tobj, content);
            }
            // restore the object's __dict__
            dict d = extract<dict>(obj.attr("__dict__"));
            d.update(state[1]);
        }


        static bool getstate_manages_dict()  { return true; }


    private:

        static bool frompython(
                diffpy::srreal::StructureAdapterPtr adpt)
        {
            return bool(boost::dynamic_pointer_cast<T>(adpt));
        }

};  // class StructureAdapterPickleSuite

/// Helper function for creating Python constructor from string
/// that restores c++ non-wrapped classes.
boost::python::object StructureAdapter_constructor();

}   // namespace srrealmodule

#endif  // SRREAL_PICKLING_HPP_INCLUDED
