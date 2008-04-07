#!/usr/bin/env python
"""This module contains the PDFData class for containing a PDF signal.

See the class documentation for more information.
"""

__id__ = "$Id$"

from RefinementAPI.Data import Data
from RefinementAPI.Refinable import Refinable
from RefinementAPI.exceptiontools import verifyType
from numpy import array
from RefinementAPI.exceptiontools import isIterable, isFloatable

class PDFData(Data, Refinable):
    """Class for holding a diffraction pattern.

    Attributes
    _xobs       --  A numpy array of the observed scattering variable
    _yobs       --  A numpy array of the observed pattern
    _uobs       --  A numpy array of the uncertainty of the observed pattern
    _meta       --  MetaData instance (default None) set when 'loadData'
                    is called. The instance contains the entries from the
                    _meta dictionary from the PatternParser used with
                    'loadData' in addition to the parser format under the
                    "parser" key.
    _stype      --  Scattering type (string, "X" or "N", default "X")
    _qmax       --  Maximum Q-value in scattering (nonnegative float,
                    1/Angstrom, default 0). A _qmax value of zero implies
                    infinite qmax.

    Refinable parameters
    scale       --  Scale factor (float, unitless, default 1.0).
    qbroad      --  Peak broadening factor (float, 1/Angstroms^2, default 0.0).
    qdamp       --  Peak damping factor (float, 1/Angstroms^2, default 0.0).
    spdiameter  --  Spherical nanoparticle diameter (float, Angstroms, default
                    0.)

    Common Exceptions
    TypeError is thrown when method arguments are of inappropriate type.
    ValueError when method argument does not fall within specified bounds.
    """

    # The following methods do not need to be overloaded

    def __init__(self):
        """Initialize the attributes."""
        Data.__init__(self)
        Refinable.__init__(self)
        self._stype = "X"
        self._qmax = 0
        self.addParameter("scale", 1.0)
        self.addParameter("qbroad", 0.0)
        self.addParameter("qdamp", 0.0)
        self.addParameter("spdiameter", 0.0)
        
    def setQmax(self, qmax):
        """Set qmax.

        Arguments:
        qmax        --  Qmax value (nonnegative float, 1/Angstroms).
        """
        if not isFloatable(qmax):
            raise TypeError("Qmax must be a float")
        qmax = float(qmax)
        if qmax < 0:
            raise ValueError("Qmax cannot be negative")
        self._qmax = qmax
        return

    def getQmax(self):
        """Get qmax."""
        return self._qmax

    def setScatteringType(self, stype):
        """Set the scattering radiation type.

        Arguments:
        stype       --  The scattering type, "X" for xray, "N" for neutron.
        """
        verifyType(stype, str)
        stype = stype.upper()
        if stype not in ["X", "N"]:
            raise ValueError("Scattering type must be 'X' or 'N'")
        self._stype = stype
        return

    def getScatteringType(self):
        """Get the scattering radiation type."""
        return self._stype

    def loadData(self, filename, parser):
        """Load a diffraction pattern from file.

        This sets the name of this pattern to filename and will overwrite the
        local attributes and variables based on the metadata from the parser.

        Arguments
        filename    --  The name of the file from which to load the data
                        (string).
        parser      --  A PDFParser instance. This will call the parser's
                        parseFile method to read the data. If data has already
                        been parsed from file, then it will not be parsed again.
                        This means that the parser instance is reusable.  See
                        the PDFParser class for more information.

        Raises
        IOError if the filename cannot be read
        ParserError if the data cannot be parsed
        """
        from RefinementAPI.MetaData import MetaData
        # make a new metadata instance

        x, y, u = parser.getParsedData()
        if len(x) == 0:
            parser.parseFile(filename)
            x, y, u = parser.getParsedData()

        self._meta.update(parser.getMetaData())
        self.setName(filename)
        self.setDataArrays(x, y, u)

        # Set attributes based on metadata
        self.setQmax( self._meta.get("qmax", self._qmax) )
        self.setScatteringType( self._meta.get("stype", self._stype) )

        # Set variables based on metadata
        self.scale = self._meta.get("scale", self.scale)
        self.qdamp = self._meta.get("qdamp", self.qdamp)
        self.qbroad = self._meta.get("qbroad", self.qbroad)
        self.spdiameter = self._meta.get("spdiameter", self.spdiameter)
        return


# End of PDFData

if __name__ == "__main__":
    # Check to see if everything imports correctly
    pass
