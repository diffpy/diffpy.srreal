#!/usr/bin/env python

##################################################################

import os, sys
import numpy
import unittest
##################################################################

from testUtil import SetEnviron
SetEnviron()
##################################################################

#from xmlModel import XmlModel
from xmlFitting import XmlFitting
from xmlOptimizer import XmlOptimizer
#from pdfTheory import PDFTheory, PDFParameter
#from pdfDataset import PDFDataset
from testUtil import CHOICE, VERBISITY, EX_BASE_DIR

from testPDFData import makePDFSource, EPS

##################################################################

##################################################################
def compareFitting(obj, obj0):
        print 'attributes:', obj0._eqAttributes(obj)        
        print 'children:', obj0._eqChildren(obj)
        print 'optimizer', obj.getXmlOptimizer() == obj0.getXmlOptimizer()
        if not (obj.getXmlOptimizer() == obj0.getXmlOptimizer()):
            print 'optimizer1:', obj.getXmlOptimizer()
            print 'optimizer2:', obj0.getXmlOptimizer()
        print 'mxor', obj.getXmlMultiplexor() == obj0.getXmlMultiplexor()

        cons = obj.getXmlMultiplexor().getXmlConstrains()
        cons0 = obj0.getXmlMultiplexor().getXmlConstrains()
        print 'constrains #:', cons.count()
        for ind in xrange(cons.count()):
            print 'constrain:', cons.get(ind) == cons0.get(ind)
            
        vars = obj.getXmlMultiplexor().getXmlVariables()
        vars0 = obj0.getXmlMultiplexor().getXmlVariables()
        print 'variables #:', vars.count()
        for ind in xrange(vars.count()):
            print 'variable:', vars.get(ind) == vars0.get(ind)
            #print 'variable:', vars.get(ind), vars0.get(ind)
            #var = vars.get(ind); var0 = vars0.get(ind)
            #print var.error==var0.error, var.init==var0.init, \
            #      var.v0==var0.v0, var.value==var0.value, \
            #      var.target==var0.target, var.range==var0.range   
            #print 'variable', type(vars.get(ind)) == type(vars0.get(ind))
            
        models = obj.getXmlMultiplexor().getXmlModels()
        models0 = obj0.getXmlMultiplexor().getXmlModels()
        print 'models #:', len(models)
        for ind in xrange(len(models)):
            print 'models:', models[ind] == models0[ind]
            pm = models[ind].getXmlParameters()
            pm0 = models0[ind].getXmlParameters()
            for ind0 in xrange(len(pm)):
                print 'parameters:', pm[ind0] == pm0[ind0]#, pm._eqAttributes(pm0)
                print 'type', type(pm[ind0]), type(pm0[ind0])
            print 'dataset:', models[ind].getXmlDataset() == models0[ind].getXmlDataset()
            data = models[ind].getXmlDataset().getXmlData()
            data0 = models0[ind].getXmlDataset().getXmlData()
            
            for ind0 in xrange(len(data)):
                print 'data:', data[ind0] == data0[ind0]#, pm._eqAttributes(pm0)
                print 'data attrs:', data[ind0]._eqAttributes(data0[ind0])
                print 'instrument data :', data[ind0].getXmlInstrumentData() == \
                                data0[ind0].getXmlInstrumentData() 
                print 'count:', data[ind0].getChildren()[0] == data0[ind0].getChildren()[0]
                #print  data0[ind0].getChildren()[1]
                #print 'type', type(data[ind0]), type(data0[ind0])

             
##################################################################
class PDFFittingTest(unittest.TestCase):

    def setUp(self):  
        (self.model, self.xdata, self.ydata) = makePDFSource(False)  
        #print 'self.model', self.model.clone()
         
        (self.model2, self.xdata2, self.ydata2) = makePDFSource(False) 
         
        self.fname = os.path.join(EX_BASE_DIR,'pdfFitting.xml')
        
        self.tfname = os.path.join(EX_BASE_DIR, 'pdfFittingT.xml')
                
        # original file
        self.ofname = os.path.join(EX_BASE_DIR, 'pdfFittingO.xml')
        # local file
        self.lfname = os.path.join(EX_BASE_DIR, 'pdfFittingL.xml')
        # imbed file
        self.mfname = os.path.join(EX_BASE_DIR, 'pdfFittingM.xml')
        
        # stream file
        self.sfname = os.path.join(EX_BASE_DIR, 'pdfFittingS.xml')
                        
        # reply file
        self.pfname = os.path.join(EX_BASE_DIR, 'pdfFittingP.xml')
        
        self.obj = XmlFitting()
                            
        opt = self.obj.getXmlOptimizer()
        opt.optfunc="fmin"
        
        xor = self.obj.getXmlMultiplexor()
        xor.add(self.model)
        
        vars = xor.updateVariables()
        #vars = xor.updateVariables()
        cons = xor.getXmlConstrains()
        
        self.obj.toFile(self.fname)    

        ##########################################
        self.obj2 = XmlFitting()
                            
        opt = self.obj2.getXmlOptimizer()
        opt.optfunc="fmin"
        
        xor = self.obj2.getXmlMultiplexor()
        xor.add(self.model2)
        
        vars = xor.updateVariables()
        cons = xor.getXmlConstrains()  
          
        for model in self.obj2.getXmlMultiplexor().getXmlModels():
            dataset = model.getXmlDataset()
            dataset.setLocalSrc()
        #print 'file:', self.lfname 
        self.obj2.toFile(self.lfname)
         
        for model in self.obj2.getXmlMultiplexor().getXmlModels():
            dataset = model.getXmlDataset()
            dataset.setImbedSrc()
            dataset.update4Source()
            
        self.obj2.toFile(self.mfname)
            
        for model in self.obj2.getXmlMultiplexor().getXmlModels():
            model.setReplyFlag()  
                            
        self.obj2.toFile(self.pfname)
        
        fp = open(self.sfname, 'w')
        fp.write(self.obj2.toStream())
        fp.close()    
    """
    def testReadWrite(self):
        obj0 =XmlFitting()     
        obj0.parseFile(self.fname)
        
        self.assertEqual(len(self.obj.toXml()), len(obj0.toXml()))
        self.obj.getXmlModels()[0].getXmlDataset().update4Source()
        self.assertEqual(self.obj, obj0)
        
        s0 = self.obj.toXml()
        obj1 =XmlFitting()
        
        obj1.parseString(s0)
        self.assertEqual(obj0, obj1)
    """
    def testClone(self):
        #print 'obj:', self.obj
        #model = self.obj.getXmlMultiplexor().getXmlModels()[0]
        #print 'model:',  model
        #print 'model clone:', model.clone()
        #print 'model', model.getXmlDataset().clone()
        #pm = model.getXmlParameters()[0]
        #print 'parameter:', pm
        #print 'type(parameter)', type(pm)
        #print 'parameter:', pm.clone()
        #print 'model', model.getXmlParameters()[0].clone()
        
        obj0 = self.obj.clone() 
        
        compareFitting(self.obj, obj0) 
               
        self.assertEqual(obj0, self.obj)  

    """
    def testFullReadWrite(self):
        obj0 = XmlFitting()             
        obj0.parseFile(self.fname)
        
        #compareFitting(self.obj, obj0) 
        self.obj.getXmlModels()[0].getXmlDataset().update4Source()
        self.assertEqual(obj0, self.obj)

        obj0.toFile(self.tfname)
                
        obj1 = XmlFitting()             
        obj1.parseFile(self.tfname)
        self.assertEqual(obj1, obj0)
        
        s0 = obj0.toXml()
        obj1 = XmlFitting()
        
        obj1.parseString(s0)
        self.assertEqual(obj0, obj1)
    """
##################################################################        
if __name__=='__main__':

    if (CHOICE == 1):
        suite = unittest.TestLoader().loadTestsFromTestCase(
                                   PDFFittingTest)
        unittest.TextTestRunner(verbosity=VERBISITY).run(suite)
    else:
        unittest.main()
##################################################################
