#!/usr/bin/env python

################################################################

import numpy
import os, sys
import traceback
################################################################

from park.xmlUtil.xmlAttribute import XmlAttribute
from park.fit.xmlData import XmlData, DATA_SOURCE_TAG
from park.fit.xmlDataMeta import XmlMetaData, XmlInstrumentData

from park.fit.xmlDataArray import XmlDataArray
from park.fit.xmlDataReduction import XmlReductionData

from park.xmlUtil.xmlHelper import setDefault as SetDefault
from park.theory.utilIO import readAsciiData, writeAsciiData, joinData

################################################################
from park.theory.chisq import getScaleMatch, getScaleMatchErrorbar
################################################################

## file suffix for the combined data.
PDF_FILE_SUBFFIX = '.dat'
PDF_ARRAY_LABELS = ('X', 'Y', 'dY')
#################################################################

""" Data structure for pdf data. """
#################################################################

PDF_INSTRUMENT_DATA_TAG = 'pdf'
 
PDF_DATA_CLASSNAME = 'pdfData.PDFData' 

#################################################################        
class PDFData(XmlData):  
    """ PDF data: three columns: X - Y - dY  """

#################################################################

    def __init__(self):
        """ Constructor."""
        super(PDFData, self).__init__()  
        self.setClassName(PDF_DATA_CLASSNAME)
        ## the raw data
        self._xx0 = None
        self._xy0 = None
        self._xu0 = None
#################################################################

    def _postParse(self):
        """ Constructor."""
        if not hasattr(self, '_xx0'):
            self._xx0 = None
        if not hasattr(self, '_xy0'):
            self._xy0 = None
        if not hasattr(self, '_xu0'):
            self._xu0 = None
#################################################################
             
    def _getNodeObject(self, nodename):
        if nodename == PDF_INSTRUMENT_DATA_TAG:
            return PDFInstrumentData() 
        else:
            return super(PDFData, 
                self)._getNodeObject(nodename)
#################################################################

    def _getSelfObject(self):
        """ return a copy of current object.  """
        return PDFData()
#################################################################

    def _checkXmlReductionData(self, reductionData):
        """ Check the validity of reduction portion."""
        #reduction = super(PDFData, self).checkReductionData()

        cnt = len(reductionData.getXmlDataArray())
        if cnt <= 0:
            reductionData.add(XmlDataArray())
            reductionData.add(XmlDataArray())            
            reductionData.add(XmlDataArray())            
        elif cnt == 1:
            reductionData.add(XmlDataArray())
            reductionData.add(XmlDataArray())
        elif cnt == 2:
            reductionData.add(XmlDataArray())
#################################################################

    def hasData(self):
        """ Return True if the data has the valid data.
        """
        return self._xx0 is not None
#################################################################
    
    def update4Source(self):   
        """ Update the reduction data due to change of data source.""" 
        try:                
            [self._xx0, self._xy0, self._xu0] = self._readRawData()
        except:
            self._xx0 = None;  self._xy0 = None; self._xu0 = None
        
        self.update4Meta()
#################################################################
        
    def update4Meta(self):
        """ 
            Update the reduction data due to change of meta data.        
        """ 
        data = self.checkXmlReductionData().getXmlDataArray()            
        data[0].setData(self._xx0)        
        if self._xy0 is None:
            data[1].setData(None)
        else:
            data[1].setData(self._xy0*self.getXmlInstrumentData().scale)
#################################################################
    
    def _readRawData(self):
        """ Return the original PDF data array:  [X,Y] 
        """
        filename = getattr(self,DATA_SOURCE_TAG)
        dvals = readAsciiData(filename)
        if len(dvals) > 3:
            del dvals[2]
        elif len(dvals) < 3:
            dvals.append( numpy.ones(len(dvals[0]), dtype=float) )
        return dvals[:3]
#################################################################
                    
    def getXmlInstrumentData(self):
        """ Return the instrument data, one data has only one instrument data."""
        try:
            inst = self.getChildren(PDF_INSTRUMENT_DATA_TAG)[0]
        except:
            inst = PDFInstrumentData()
            self.add(inst)
        
        return inst
#################################################################
    
    def setOriginalScale(self):
        self.getXmlInstrumentData().scale = 1.0        
#################################################################



#################################################################

#####  default values for control parameters of PDF function.
PDF_A0=1.0  # moderator detector distance (meters)

#################################################################
class PDFInstrumentData(XmlAttribute):
    """ PDF Instrument data. """
#################################################################    
    def __init__(self):
        super(PDFInstrumentData, self).__init__(
                     PDF_INSTRUMENT_DATA_TAG)
        self.setDefault()        

#################################################################    
    def _getSelfObject(self):
        """ return a copy of current object.  """
        return PDFInstrumentData()
    
#################################################################    
    def _postParse(self):
        self.setDefault()
        
#################################################################        
    def setDefault(self):
        SetDefault(self, 'scale', PDF_A0 )        
#################################################################

#################################################################
PDF_INST_DATA = PDFInstrumentData()
#################################################################

#################################################################

#################################################################    
# For testing imports
if __name__ == "__main__":
    print "Imports working!"
    
