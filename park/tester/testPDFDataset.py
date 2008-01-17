#!/usr/bin/env python

##################################################################

import os, sys
import numpy
import unittest
##################################################################


from park.fit.xmlModel import XmlModel
from SrReal.park.pdfData import PDFData, PDFInstrumentData
from SrReal.park.pdfDataset import PDFDataset
from SrReal.park.pdfTheory import PDFTheory, PDFParameter, getPDFParameters
from park.theory.utilIO import writeAsciiData, readAsciiData, isEqualArray
from testUtil import CHOICE, VERBOSITY, EX_BASE_DIR
from testPDFData import makePDFSource, EPS
##################################################################

##################################################################
class PDFDatasetTest(unittest.TestCase):

    def setUp(self):   
        (self.model, self.xdata, self.ydata) = makePDFSource(False)    
        self.fname = os.path.join(EX_BASE_DIR, 'pdfDataset.xml')
        
        # temperatory file
        self.dfname = os.path.join(EX_BASE_DIR, 'pdfDatasetT.xml')
        # original file
        self.ofname = os.path.join(EX_BASE_DIR, 'pdfDatasetO.xml')
        
        # local file
        self.lfname = os.path.join(EX_BASE_DIR, 'pdfDatasetL.xml')
        
        # imbed file
        self.mfname = os.path.join(EX_BASE_DIR, 'pdfDatasetM.xml')
        
        # reduction file
        self.rfname = os.path.join(EX_BASE_DIR, 'pdfDatasetR.xml')
        
        self.model.getXmlDataset().toFile(self.fname)
        self.obj = PDFDataset()            
        self.obj.parseFile(self.fname)
        #self.obj.toFile(self.fname)
        
        dataset = self.model.getXmlDataset()
        dataset.toFile(self.ofname)
        
        dataset.setLocalSrc()
        dataset.toFile(self.lfname)
        
        dataset.setImbedSrc()
        dataset.update4Source()
        dataset.toFile(self.mfname)
        
        dataset.setReductionSrc()        
        dataset.toFile(self.rfname)
        
        dataset.setImbedSrc()
        
    def testReadWrite(self):
        obj0 =PDFDataset()     
        obj0.parseFile(self.fname)
        
        self.assertEqual(len(self.obj.toXml()), len(obj0.toXml()))
        self.assertEqual(self.obj, obj0)
        
        s0 = self.obj.toXml()
        obj1 =PDFDataset()
        
        obj1.parseString(s0)
        self.assertEqual(obj0, obj1)
        
    def testClone(self):
        obj0 = self.obj.clone()
        self.assertEqual(obj0, self.obj)        

    def testDataSourceType(self):
        obj =PDFDataset() 
        self.assert_(obj.isLocalSrc()) 
        obj.parseFile(self.fname)        
        self.assert_(obj.isLocalSrc())
        
        obj.setLocalSrc()
        self.assert_(obj.isLocalSrc())
        
        obj.setUrlSrc()
        self.assert_(obj.isUrlSrc())
        
        obj.setImbedSrc()
        self.assert_(obj.isImbedSrc())
        
        obj.setUserSrc()
        self.assert_(obj.isUserSrc())
        
        obj.setReductionSrc()
        self.assert_(obj.isReductionSrc())

    def testDataset(self):
        obj = PDFDataset() 
        obj.parseFile(self.mfname)
        
        print 'obj', obj
        print self.model.getXmlDataset()
        o1 = file("temp1.out", 'w')
        o1.write(obj.__str__())
        o1.close()
        o2 = file("temp2.out", 'w')
        o2.write(self.model.getXmlDataset().__str__())
        o2.close()
        import scipy.io
        self.assert_(self.model.getXmlDataset()==obj)
        
        ind = 0        
        for data in obj.getXmlData():
            data.update4Source()
            self.assert_(isEqualArray(data.getData(0), self.xdata[ind], EPS))
            self.assert_(isEqualArray(data.getData(1), self.ydata[ind], EPS))
            ind += 1
            #print 'ind=', ind, '\n y data', ydata[ind-1], data.getData(1)
        
##################################################################        
if __name__=='__main__':

    if (CHOICE == 1):
        suite = unittest.TestLoader().loadTestsFromTestCase(
                                   PDFDatasetTest)
        unittest.TextTestRunner(verbosity=VERBOSITY).run(suite)
    else:
        unittest.main()
##################################################################
