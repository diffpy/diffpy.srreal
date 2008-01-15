#!/usr/bin/env python

##################################################################

import os, sys
import numpy
import unittest
##################################################################

from park.fit.xmlModel import XmlModel
from SrReal.park.pdfTheory import PDFTheory, PDFParameter
from SrReal.park.pdfDataset import PDFDataset
from testUtil import CHOICE, VERBOSITY, EX_BASE_DIR

from testPDFData import makePDFSource, EPS
##################################################################

GAUSS_SRC_FILE = os.path.join(EX_BASE_DIR,'pdfModel.xml')
GAUSS_SRC_LOCAL_FILE = os.path.join(EX_BASE_DIR, 'pdfLocal.xml')
GAUSS_SRC_IMBED_FILE = os.path.join(EX_BASE_DIR, 'pdfImbed.xml')
GAUSS_SRC_STREAM_FILE = os.path.join(EX_BASE_DIR, 'pdfStream.xml')
#################################################################
    
##################################################################
class PDFModelTest(unittest.TestCase):

    def setUp(self):  
        (self.model, self.xdata, self.ydata) = makePDFSource(False)   
            
        self.fname = os.path.join(EX_BASE_DIR,'pdfModel.xml')
        
        self.rfname = os.path.join(EX_BASE_DIR, 'pdfModelR.xml')
                
        # temperatory file
        self.dfname = os.path.join(EX_BASE_DIR, 'pdfModelT.xml')
        # original file
        self.ofname = os.path.join(EX_BASE_DIR, 'pdfModelO.xml')
        # local file
        self.lfname = os.path.join(EX_BASE_DIR, 'pdfModelL.xml')
        # imbed file
        self.mfname = os.path.join(EX_BASE_DIR, 'pdfModelM.xml')
        #print 'model', self.model
        
        # stream file
        self.sfname = os.path.join(EX_BASE_DIR, 'pdfModelS.xml')
        
        # reply file
        self.pfname = os.path.join(EX_BASE_DIR, 'pdfModelP.xml')
        
        self.model.toFile(self.fname)
        self.obj = XmlModel()            
        self.obj.parseFile(self.fname)
        
        dataset = self.model.getXmlDataset()
        self.model.toFile(self.ofname)
        
        dataset.setLocalSrc()
        self.model.toFile(self.lfname)
        
        dataset.setImbedSrc()
        dataset.update4Source()
        self.model.toFile(self.mfname)
        
        self.model.setReplyFlag()                  
        self.model.toFile(self.pfname)

        fp = open(self.sfname, 'w')
        fp.write(self.model.toStream())
        fp.close()
        
    def testReadWrite(self):
        obj0 =XmlModel()     
        obj0.parseFile(self.fname)
        
        self.assertEqual(len(self.obj.toXml()), len(obj0.toXml()))
        
        #print 'attriutes:', self.obj.getAttributeNames()
        #print 'attriutes:', obj0.getAttributeNames()
        #print 'compare attributes:', self.obj._eqAttributes(obj0)
        #for name in self.obj.getAttributeNames():
        #    if getattr(self.obj, name) != getattr(obj0, name):
        #        print 'Not eq for ', name
        #print 'compare children:', self.obj._eqChildren(obj0)
        self.assertEqual(self.obj, obj0)
        
        s0 = self.obj.toXml()
        obj1 =XmlModel()
        
        obj1.parseString(s0)
        self.assertEqual(obj0, obj1)
        
    def testClone(self):
        obj0 = self.obj.clone()
        self.assertEqual(obj0, self.obj)
        #print 'clone of model', self.obj.clone()        

    def testFullReadWrite(self):
        obj0 = XmlModel()             
        obj0.parseFile(self.fname)
        obj0.getXmlDataset().update4Source()
        obj0.getXmlDataset().setImbedSrc()
        obj0.toFile(self.rfname)
        #print obj0
        
        obj1 = XmlModel()             
        obj1.parseFile(self.rfname)
        #self.assertEqual(len(self.obj.toXml()), len(obj0.toXml()))
        
        #print 'attriutes:', self.obj.getAttributeNames()
        #print 'attriutes:', obj0.getAttributeNames()
        #print 'compare attributes:', self.obj._eqAttributes(obj0)
        #for name in self.obj.getAttributeNames():
        #    if getattr(self.obj, name) != getattr(obj0, name):
        #        print 'Not eq for ', name
        #print 'compare children:', self.obj._eqChildren(obj0)
        self.assertEqual(obj1, obj0)
        
        s0 = obj0.toXml()
        obj1 =XmlModel()
        
        obj1.parseString(s0)
        self.assertEqual(obj0, obj1)         
        
    def test_write(self):
        model = XmlModel()
        model.parseFile(GAUSS_SRC_FILE)
        
        # imbed type
        dataset = model.getXmlDataset()
        dataset.setImbedSrc()
        #print 'write imbed data in ',GAUSS_SRC_IMBED_FILE 
        model.toFile(GAUSS_SRC_IMBED_FILE)
        
        model1 = XmlModel()
        model1.parseFile(GAUSS_SRC_IMBED_FILE)
    
        # local type
        dataset.setLocalSrc()
        #print 'write local data in ',GAUSS_SRC_LOCAL_FILE 
        model.toFile(GAUSS_SRC_LOCAL_FILE)  
        
        model1 = XmlModel()
        model1.parseFile(GAUSS_SRC_LOCAL_FILE)
        
        # output stream string        
        fp=open(GAUSS_SRC_STREAM_FILE, 'w')
        fp.write(model.toStream())
        fp.close()
        #print 'toStream:', model.toStream() 
        
    
##################################################################        
if __name__=='__main__':

    if (CHOICE == 1):
        suite = unittest.TestLoader().loadTestsFromTestCase(
                                   PDFModelTest)
        unittest.TextTestRunner(verbosity=VERBOSITY).run(suite)
    else:
        unittest.main()
##################################################################
