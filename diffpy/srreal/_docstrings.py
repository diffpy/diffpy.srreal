#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     Complex Modeling Initiative
#                   (c) 2016 Brookhaven Science Associates,
#                   Brookhaven National Laboratory.
#                   All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################


"""\
Docstrings for classes and functions in srreal_ext module.
"""

# Shared docstrings for classes derived from HasClassRegistry ----------------

def get_registry_docstrings(cls):
    """Build a dictionary of docstrings per each HasClassRegistry method.

    Parameters
    ----------
    cls : class type that is wrapped
        This parameter is used to extract the class name and substitute
        it in the docstrings template.

    Returns a dictionary mapping Python method names to their docstrins.
    """
    n = cls.__name__
    rv = {k : v.replace('@NAME@', n) for k, v in (
        ("create", doc_HasClassRegistry_create),
        ("clone", doc_HasClassRegistry_clone),
        ("type", doc_HasClassRegistry_type),
        ("_registerThisType", doc_HasClassRegistry__registerThisType),
        ("createByType", doc_HasClassRegistry_createByType),
        ("getRegisteredTypes", doc_HasClassRegistry_getRegisteredTypes),
    )}
    return rv


doc_HasClassRegistry_create = """\
Return a new instance of the same type as self.

This method must be overloaded in a derived class.
"""


doc_HasClassRegistry_clone = """\
Return a new instance that is a copy of self.

This method must be overloaded in a derived class.
"""


doc_HasClassRegistry_type = """\
Return a unique string type that identifies a @NAME@-derived class.
The string type is used for class registration and in the `createByType`
function.

This method must be overloaded in a derived class.
"""


doc_HasClassRegistry__registerThisType = """\
Add this class to the global registry of @NAME@ types.

This method must be called once after definition of the derived
class to support pickling and the `createByType` factory.
"""


doc_HasClassRegistry_createByType = """\
Return a new @NAME@ instance of the specified string type.

Parameters
----------
tp : str
    string type identifying a registered @NAME@ class.

Returns a new instance of the @NAME@-derived class.

See Also
--------
getRegisteredTypes : Return set of the recognized type strings.
"""


doc_HasClassRegistry_getRegisteredTypes = """\
Return a set of string types of the registered @NAME@ classes.

These are the allowed arguments for the `createByType` factory.
"""
