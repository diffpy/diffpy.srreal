#!/usr/bin/env python

##################################################################
import math
import numpy
import os, sys
import unittest
##################################################################

from park.fit.xmlModel import XmlModel
from park.fit.xmlData import DATA_TAG
from SrReal.park.pdfData import PDFData, PDFInstrumentData
from SrReal.park.pdfDataset import PDFDataset
from SrReal.park.pdfTheory import PDFTheory, PDFParameter, getPDFParameters
from park.theory.utilIO import writeAsciiData, readAsciiData, isEqualArray
    
from testUtil import CHOICE, VERBOSITY, EX_BASE_DIR
##################################################################
EPS = 1.0e-8
# 0.3 + 7*x**3 - 0.1*x**4
MODEL_NUM = 1
RMIN = 1.0
RMAX = 10
DR = 0.1
SCALE = 10.0

#################################################################
#################################################################
#print 'NUM:', MODEL_NUM

def makeModel():
    model = XmlModel()
    model.name = 'M0' 
    model.theory = 'SrReal.park.pdfTheory.PDFTheory'
    model.weight = 1.0

    from diffpy.pdffit2 import PdfFit
    P = PdfFit()
    P.read_struct(os.path.join(EX_BASE_DIR,"Ni.stru"))
    params = getPDFParameters(P, "X", 45.0)
    model.addChild(params[0])
        
    return model

#################################################################
def makePDFSource(createData = False):
    """  Generate a pdf serial function. """        
    model = makeModel()
    theory = PDFTheory()
    
    dataset = PDFDataset()
    model.add(dataset)
    
    xdata = []
    ydata = []
    
    #print 'pdf model', model
    #print 'len of parameters:',len(model.getParameters())
    
    for j in xrange(MODEL_NUM):
        x0 = numpy.arange(RMIN, RMAX, DR)
        
        y0 = theory._getFx((x0,), model.getXmlParameters())        
        fullname = os.path.join(EX_BASE_DIR, 'pdfData%i.dat' %(j))
        
        if createData:
            writeAsciiData(fullname, (x0, y0[0]))
        
        gdata = PDFData()
        gdata.setSource(fullname)
        gdata.add(PDFInstrumentData())
        
        #print 'pdf data:', gdata
        dataset.add(gdata)
        #print 'pdf data', len(dataset.getXmlData())
        xdata.append(x0)
        ydata.append(y0[0])
        
    #print 'dataset:', dataset
    #dataset.toFile(pdf_DATASET_FILE)  
    #model.toFile(pdf_MODEL_FILE)
        
    return (model, xdata, ydata)
##################################################################
class PDFDataTest(unittest.TestCase):

    def setUp(self):
        self.s0 = """
            <%s classname="pdfData.PDFData" srctype='imbed'/>
            """  %(DATA_TAG)
       
        self.fname = os.path.join(EX_BASE_DIR,'pdfData.xml')
        # remote file
        self.rfname = os.path.join(EX_BASE_DIR, 'pdfDataR.xml')
        # temperatory file
        self.dfname = os.path.join(EX_BASE_DIR, 'pdfDataT.xml')
        # original file
        self.ofname = os.path.join(EX_BASE_DIR, 'pdfDataO.xml')
        # local file
        self.lfname = os.path.join(EX_BASE_DIR, 'pdfDataL.xml')
        # imbed file
        self.mfname = os.path.join(EX_BASE_DIR, 'pdfDataM.xml')
        
        self.obj =PDFData()            
        self.obj.parseString(self.s0)
        self.obj.toFile(self.fname)
        
        (self.model, self.xdata, self.ydata) = makePDFSource(True)
                
        #print  'dataset', self.model.getXmlDataset()
        
        data = self.model.getXmlDataset().getXmlData()[0]
        #print 'pdf data:', data.source
        data.toFile(self.ofname)
        
        data.setLocalSrc()
        data.toFile(self.lfname)
        
        data.setImbedSrc()
        data.update4Source()
        #print 'imbed data:', data
        data.toFile(self.mfname)
        
                        
    def testReadWrite(self):
        obj0 =PDFData()     
        obj0.parseFile(self.fname)
        
        self.assertEqual(len(self.obj.toXml()), len(obj0.toXml()))
        self.assertEqual(self.obj, obj0)
        
        s0 = self.obj.toXml()
        obj1 =PDFData()
        
        obj1.parseString(s0)
        #print 'attributes:', obj0._eqAttributes(obj1)
        #print 'children:', obj0._eqChildren(obj1)
        self.assertEqual(obj0, obj1)
        
    def testClone(self):
        obj0 = self.obj.clone()
        #print 'attributes:', obj0._eqAttributes(self.obj)
        #print 'children:', obj0._eqChildren(self.obj)
        self.assertEqual(obj0, self.obj)        

    def testDataSourceType(self):
        obj =PDFData() 
        self.assert_(obj.isLocalSrc()) 
        
        obj.parseFile(self.fname)        
        self.assert_(obj.isImbedSrc())
        
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

    def testSource(self):
        obj =PDFData() 
        self.assert_(obj.isUnknownSource())
        self.assert_(obj.isUnknownRemote())
        self.assert_(obj.isLocalSrc())
        obj.toFile(self.rfname)
        
        obj1 =PDFData()
        obj1.parseFile(self.rfname)
        self.assertEqual(obj, obj1)
        self.assert_(obj1.isUnknownSource())
        self.assert_(obj1.isUnknownRemote())
        self.assert_(obj1.isLocalSrc())
        
        obj.setSource(self.fname)
        self.assertEqual(obj.getSource(), self.fname)
        self.assert_(not obj.isUnknownSource())
        self.assert_(obj.isUnknownRemote())        
        obj.toFile(self.rfname)
        
        obj1 =PDFData()
        obj1.parseFile(self.rfname)
        self.assertEqual(obj, obj1)
        self.assertEqual(obj1.getSource(), self.fname)
        self.assert_(not obj1.isUnknownSource())
        self.assert_(obj1.isUnknownRemote()) 
                
        obj.setRemote(self.rfname)
        self.assertEqual(obj.getSource(), self.fname)
        self.assertEqual(obj.getRemote(), self.rfname)
        self.assert_(not obj.isUnknownSource()) 
        self.assert_(not obj.isUnknownRemote())        
        obj.toFile(self.rfname)
        
        obj1 =PDFData()
        obj1.parseFile(self.rfname)
        self.assertEqual(obj1.getSource(), self.fname)
        self.assertEqual(obj1.getRemote(), self.rfname)
        self.assert_(not obj1.isUnknownSource()) 
        self.assert_(not obj1.isUnknownRemote()) 
                                      
    def testMakeData(self):
        
        theory = PDFTheory()
        model = makeModel()
    
        for j in xrange(MODEL_NUM):
            x0 = numpy.arange(RMIN, RMAX, DR)
            y0 = theory._getFx((x0,), model.getXmlParameters()) 
            u0 = numpy.ones(len(x0), dtype=float)
            self.assert_(abs(theory._getObjectiveFx((x0, y0[0],u0), 
                                    model.getXmlParameters()))<EPS)    
            fullname = os.path.join(EX_BASE_DIR,'pdfData%i.dat' %(j))
        
            [x1, y1] = readAsciiData(fullname)
            y2 = y0[0]
            #self.assert_(all(x1==x0))
            for ind in xrange(len(y1)):
                self.assert_(abs(x1[ind]-x0[ind]) < EPS) 
                if abs(y1[ind]-y2[ind]) > EPS:
                    print 'ind=', ind, y1[ind]-y2[ind]
                self.assert_(abs(y1[ind]-y2[ind]) < EPS) 
            #self.assert_(all(y1==y2))
    
    def testHasData(self):
        for j in xrange(MODEL_NUM):
            obj = PDFData()
            obj.source = os.path.join(EX_BASE_DIR, 'pdfData%i.dat' %(j) )
            obj.update4Source()
            obj.toFile(self.dfname)
            data = obj.getXmlReductionData().getXmlDataObject()
            
            self.assert_(isEqualArray(data[0].getData(), self.xdata[j], EPS))
            self.assert_(isEqualArray(data[1].getData(), self.ydata[j], EPS))
            
            obj1 =  PDFData()
            obj1.parseFile(self.dfname)
            data1 = obj.getXmlReductionData().getXmlDataObject()
            
            self.assert_(isEqualArray(data1[0].getData(), self.xdata[j], EPS))
            self.assert_(isEqualArray(data1[1].getData(), self.ydata[j], EPS))
            
            obj.getXmlInstrumentData().scale = SCALE
            obj.update4Meta()
            obj.toFile(self.dfname)
            data = obj.getXmlReductionData().getXmlDataObject()
            
            self.assert_(isEqualArray(data[0].getData(), self.xdata[j], EPS))
            self.assert_(isEqualArray(data[1].getData(), self.ydata[j]*SCALE, EPS))
            
            obj1 =  PDFData()
            obj1.parseFile(self.dfname)
            data1 = obj.getXmlReductionData().getXmlDataObject()
            
            self.assert_(isEqualArray(data1[0].getData(), self.xdata[j], EPS))
            self.assert_(isEqualArray(data1[1].getData(), self.ydata[j]*SCALE, EPS))
            
    
##################################################################        
if __name__=='__main__':

    if (CHOICE == 1):
        suite = unittest.TestLoader().loadTestsFromTestCase(
                                   PDFDataTest)
        unittest.TextTestRunner(verbosity=VERBOSITY).run(suite)
    else:
        unittest.main()
##################################################################
