#!/usr/bin/env python
"""This module contains the PDFParser class for parsing pdf data."""

__id__ = "$Id$"

from numpy import array, sqrt

from RefinementAPI.DataParser import DataParser
from RefinementAPI.errors import ParseError
from RefinementAPI.MetaData import MetaData

class PDFParser(DataParser):
    """Class for parsing pdf data.

    Attributes

    _format     --  Name of the data format that this parses (string, default
                    ""). The format string is a unique identifier for the data
                    format handled by the parser.
    _xdata      --  r-values.
    _ydata      --  G(r) values.
    _udata      --  Uncertainty in G(r) values.
    _meta       --  A MetaData instance containing metadata read from the file.

    Metadata
    filename    --  The name of the file from which data was parsed. This key
                    will not exist if data was not read from file.
    qmax        --  Maximum q-value used in data.
    qdamp       --  Peak dampening factor
    qbroad      --  Peak broadening factor
    stype       --  Scattering type.
    scale       --  Scale factor.
    spdiameter  --  Spherical nanoparticle diameter.
    temperature --  Temperature at which the data was taken.
    doping      --  Doping value
    other metadata defined by the programs

    Common Exceptions
    TypeError is thrown when method arguments are of inappropriate type.
    ValueError when method argument does not fall within specified bounds.
    """

    def __init__(self):
        """Initialize the attributes."""
        DataParser.__init__(self)
        from RefinementAPI.MetaData import MetaData
        self._meta = MetaData()
        return

    def parseFile(self, filename):
        """Parse a file and set the _xobs, _yobs, _uobs and _meta variables.

        This wipes out the currently loaded data and selected bank number.

        Arguments
        filename    --  The name of the file to parse

        Raises: 
        IOError if the file cannot be read
        ParseError if the file cannot be parsed
        """
        infile = file(filename, 'r')
        self._meta = MetaData()
        filestring = infile.read()
        self.parseString(filestring)
        infile.close()
        self._meta["filename"] = filename
        return

    def parseString(self, pdfstring):
        """Parse a string and set the _xobs, _yobs, _uobs and _meta variables.

        This reads the format specified by PDFgetN and PDFgetX2.  This wipes out
        the currently loaded data.
        
        Arguments
        pdfstring   --  A string containing the pdf signal.

        Raises: 
        ParseError if the string cannot be parsed
        """
        import re
        res = re.search(r'^#+ start data\s*(?:#.*\s+)*', pdfstring, re.M)
        # start_data is position where the first data line starts
        if res:
            start_data = res.end()
        else:
            res = re.search(r'^[^#]', pdfstring, re.M)
            if res:
                start_data = res.start()
            else:
                start_data = 0
        header = pdfstring[:start_data]
        databody = pdfstring[start_data:].strip()
        
        # find where the metadata starts
        metadata = ''
        res = re.search(r'^#+\ +metadata\b\n', header, re.M)
        if res:
            metadata = header[res.end():]
            header = header[:res.start()]   
            
        # parse header
        rx = { 'f' : r'[-+]?(\d+(\.\d*)?|\d*\.\d+)([eE][-+]?\d+)?' }
        # stype
        if re.search('(x-?ray|PDFgetX)', header, re.I):
            self._meta["stype"] = 'X'
        elif re.search('(neutron|PDFgetN)', header, re.I):
            self._meta["stype"] = 'N'
        # qmax
        regexp = r"\bqmax *= *(%(f)s)\b" % rx
        res = re.search(regexp, header, re.I)
        if res:
            self._meta["qmax"] = float(res.groups()[0])
        # qdamp
        regexp = r"\b(?:qdamp|qsig) *= *(%(f)s)\b" % rx
        res = re.search(regexp, header, re.I)
        if res:
            self._meta["qdamp"] = float(res.groups()[0])
        # qbroad
        regexp = r"\b(?:qbroad|qalp) *= *(%(f)s)\b" % rx
        res = re.search(regexp, header, re.I)
        if res:
            self._meta["qbroad"] = float(res.groups()[0])
        # spdiameter
        regexp = r"\bspdiameter *= *(%(f)s)\b" % rx
        res = re.search(regexp, header, re.I)
        if res:
            self._meta["spdiameter"] = float(res.groups()[0])
        # dscale
        regexp = r"\bdscale *= *(%(f)s)\b" % rx
        res = re.search(regexp, header, re.I)
        if res:
            self._meta["scale"] = float(res.groups()[0])
        # temperature
        regexp = r"\b(?:temp|temperature|T)\ *=\ *(%(f)s)\b" % rx
        res = re.search(regexp, header)
        if res:
            self._meta['temperature'] = float(res.groups()[0])
        # doping
        regexp = r"\b(?:x|doping)\ *=\ *(%(f)s)\b" % rx
        res = re.search(regexp, header)
        if res:
            self._meta['doping'] = float(res.groups()[0])
            
        # parsing gerneral metadata
        if metadata:
            regexp = r"\b(\w+)\ *=\ *(%(f)s)\b" % rx
            while True:
                res = re.search(regexp, metadata, re.M)
                if res:
                    self._meta[res.groups()[0]] = float(res.groups()[1])
                    metadata = metadata[res.end():]
                else:
                    break

        # read actual data - robs, Gobs, drobs, dGobs
        has_drobs = True
        has_dGobs = True
        # raise InvalidDataFormat if something goes wrong
        robs = []
        Gobs = []
        dGobs = []
        drobs = []
        try:
            for line in databody.split("\n"):
                v = line.split()
                # there should be at least 2 value in the line
                robs.append(float(v[0]))
                Gobs.append(float(v[1]))
                # drobs is valid if all values are defined and positive
                has_drobs = has_drobs and len(v) > 2
                if has_drobs:
                    v2 = float(v[2])
                    has_drobs = v2 > 0.0
                    drobs.append(v2)
                # dGobs is valid if all values are defined and positive
                has_dGobs = has_dGobs and len(v) > 3
                if has_dGobs:
                    v3 = float(v[3])
                    has_dGobs = v3 > 0.0
                    dGobs.append(v3)
            if not has_drobs:
                drobs = len(robs) * [0.0]
            if not has_dGobs:
                dGobs = len(robs) * [0.0]
        except (ValueError, IndexError):
            raise ParseError('Cannot read Gobs')
        if not has_drobs:   drobs = len(robs) * [0.0]
        if not has_dGobs:   dGobs = len(robs) * [0.0]
        self._xdata = array(robs)
        self._ydata = array(Gobs)
        self._udata = array(dGobs)
        return

    def getMetaData(self):
        """Get the parsed metadata.

        Returns a MetaData instance containing the metadata.
        """
        return self._meta

# End of PDFParser

if __name__ == "__main__":
    # Check to see if everything imports correctly
    pass

