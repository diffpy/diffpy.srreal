/***********************************************************************
* $Id$
***********************************************************************/
#ifndef _CONVERTERS_H
#define _CONVERTERS_H

#include <vector>
#include <boost/python.hpp>

#include <numpy/arrayobject.h>

// 
//
using namespace boost::python;

// Make an array out of a data pointer and a dimension vector
PyObject* makeNdArray(float* data, std::vector<int>& dims)
{
    PyObject* pyarray = PyArray_SimpleNewFromData
                (dims.size(), &dims[0], PyArray_FLOAT, (void *) data);
    return incref(PyArray_Copy( (PyArrayObject*) pyarray ));
}

#endif
