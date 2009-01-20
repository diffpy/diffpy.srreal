#!/usr/bin/env python
"""A modified park.Model class with some utility functions that interfere with
the normal park.Model behavior. In particular, parameter accessors give the
parameter itself, rather than its value.
"""

__id__ = "$Id:"

import park

class DynamicModel(park.Model):

    name = property(fget = lambda self: self.parameterset.name)

    def __getitem__(self, p):
        """
        Return a parameter or parameterset by name.
        """
        return self.parameterset[p]

    def __getattr__(self, p):
        """
        Return a parameter or parameterset by name.
        """
        try:
            return self.parameterset[p]
        except KeyError:
            raise AttributeError(
                "'%s' object has no attribute '%s'"\
                %(self.__class__.__name__, p))

if __name__ == "__main__":

    pass
