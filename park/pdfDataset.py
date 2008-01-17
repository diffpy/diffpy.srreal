#!/usr/bin/env python

################################################################

import numpy
import os, sys
import traceback
################################################################

from park.xmlUtil.xmlAttribute import XmlAttribute
from park.xmlUtil.xmlObject import XmlObject

from park.fit.xmlDataSource import XmlDataSource
from park.fit.xmlDataset import XmlDataset, DATASET_TAG
from park.fit.xmlData import XmlData, DATA_TAG

from park.fit.xmlDataArray import XmlDataArray

## functions or class to load SNS refl_tof file.
from park.xmlUtil.xmlHelper import setDefault as SetDefault
from park.theory.utilIO import readAsciiData, writeAsciiData, joinData

from pdfData import PDFData

################################################################
from park.theory.chisq import getScaleMatch, getScaleMatchErrorbar
################################################################

## file suffix for the combined data.
PDF_FILE_SUBFFIX = '.dat'
PDF_ARRAY_LABELS = ('X', 'Y', 'dY')
#################################################################

""" Data structure for pdf data. """
#################################################################

PDF_DATASRC_CLASSNAME = 'SrReal.park.pdfDataset.PDFDataSource' 

PDF_DATASET_CLASSNAME = 'SrReal.park.pdfDataset.PDFDataset'  

PDF_DATA_RESIDUAL_INDEX = [0, 1, 2, 3]
#################################################################

class PDFDataSource(XmlDataSource):
    """ The data source for PDF functions. """
    
    def __init__(self):
        super(PDFDataSource, self).__init__()
        self.setClassName(PDF_DATASRC_CLASSNAME)        
                   
    def _getNodeObject(self, nodename):
        if nodename == DATASET_TAG:
            return GaussDataset() 
        else:
            return super(PDFDataSource, 
                    self)._getNodeObject(nodename)
    
    def _getSelfObject(self):
        """ return a copy of current object.  """
        return GaussDataSource()

#################################################################        
class PDFDataset(XmlDataset):
    """ One dataset of PDF functions."""

    def __init__(self):
        super(PDFDataset, self).__init__()
        self.setClassName(PDF_DATASET_CLASSNAME)

    def _getSelfObject(self):
        """ return a copy of current object.  """
        return PDFDataset()

    def _getNodeObject(self, nodename):
        if nodename == DATA_TAG:
            return PDFData() 
        else:
            return super(PDFDataset, 
                         self)._getNodeObject(nodename)        

    def _checkXmlReductionData(self, reductionData):
        """ Check to make sure there is a valid reduction data"""
        #reduction = super(PDFDataset, self).checkReductionData()
        # [X, Y, dY], all are array.
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
        
    def _updateXmlReductionData(self):#, datalist):
        """ Update the reduction data for the whole dataset. """
        
        try:
            dataarray = []
            for data in self.getXmlData():
                dataarray.append(data.getDataList())
            if len(dataarray) >= 1:  
                mydata = joinData(dataarray)
                        
                if mydata is None:
                    return [None, None, None]
                else:
                    return [mydata[0], mydata[1], mydata[2]]
            else:
                return [None, None, None]
        except:
            print 'exception:', traceback.format_exc()
            return [None, None, None]

    def saveData(self, filename=None):
        if filename is None:
            myfilename = getattr(self, DATA_SOURCE_TAG)
        else:
            myfilename = filename
            setattr(self, DATA_SOURCE_TAG, myfilename)

        data = []
        for dataobj in self.getReductionData().getDataObject():
            data.append(dataobj.getData())
            
        writeAsciiData(myfilename, data)
        
    def autoMatch(self, withoutbar = True):
        """ Automatch the dataset. """
        mydata = [] 
        for data in self.getXmlData():
            data.setOriginalScale()
            data.update4Meta()
            orgdata = data.getXmlReductionData().getXmlDataObject()
            mydata.append(orgdata[0].getData())
            mydata.append(orgdata[1].getData())
            mydata.append(orgdata[2].getData())

        scale = getScaleMatch(mydata)
                               
        ind = 0           
        for data in self.getXmlData():    
            data.getXmlInstrumentData().scale = scale[ind]
            ind += 1
            
        self.update4Meta()

    def setOriginalScale(self):
        """ Set the origianl scaler, scaler=1.0 """
        for data in self.getXmlData():
            data.setOriginalScale()
            
        self.update4Meta()
    
    def getResidualIndex(self):
        """ 
            Return an integer array indicate the X:Y:dY for
            residual representation. [n1, n2, n3, [unit_count]],
            n1 for X, n2 for Y, n3 for dY, and unit_count is 
            the length of each repeated unit.
        """
        return PDF_DATA_RESIDUAL_INDEX    

#################################################################    
    
# For testing imports
if __name__ == "__main__":
    print "Imports working!"
    
