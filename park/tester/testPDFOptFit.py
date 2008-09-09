#!/usr/bin/env python

##################################################################
import os, sys
import traceback
import numpy
import unittest
##################################################################

from park.fit.xmlFitting import XmlFitting
from park.fit.xmlOptimizer import XmlOptimizer
from testUtil import CHOICE, VERBOSITY, EX_BASE_DIR
from testPDFData import makePDFSource, EPS

##################################################################
class PDFOptFittingTest(unittest.TestCase):

    def setUp(self):  
        (self.model, self.xdata, self.ydata) = makePDFSource(False)   

        # local file
        self.lfname = os.path.join(EX_BASE_DIR,'pdfFittingL.xml')
        
        # imbed file
        self.mfname = os.path.join(EX_BASE_DIR, 'pdfFittingM.xml')
        
        # stream file
        self.sfname = os.path.join(EX_BASE_DIR, 'pdfFittingS.xml' )  
        
        # reply file
        self.rfname = os.path.join(EX_BASE_DIR, 'pdfReply.xml'   )
        # reply imbed file
        self.rmfname = os.path.join(EX_BASE_DIR, 'pdfReplyM.xml' )
        # reply local file
        self.rlfname = os.path.join(EX_BASE_DIR, 'pdfReplyL.xml'  )                  
    
    """ 
    def testFitting(self):
        obj0 =XmlFitting()     
        obj0.parseFile(self.lfname)
        
        res0 = obj0.doFitting()
        
        obj1 =XmlFitting()     
        obj1.parseFile(self.mfname)
        
        res1 = obj1.doFitting()
        
        obj2 =XmlFitting()     
        obj2.parseFile(self.sfname)
        
        res2 = obj0.doFitting()
        
        self.assert_(res0 == res1)
        self.assert_(res1 == res2)
        self.assert_(res2 == res0) 
    """
      
    def testFittingReply(self):
        obj0 =XmlFitting()     
        obj0.parseFile(self.lfname)
        
        res0 = obj0.doFitting()
        
        print 'Fitting results is stored in :', self.rfname
        res0.toFile(self.rfname)
        
        #######################
        obj1 =XmlFitting()     
        obj1.parseFile(self.mfname)
        
        res1 = obj1.doFitting()
        
        print 'Imbed Fitting results is stored in :', self.rmfname
        res1.toFile(self.rmfname) 
                
        obj2 =XmlFitting()     
        obj2.parseFile(self.sfname)
        
        res2 = obj0.doFitting()             
        
        print 'Local Fitting results is stored in :', self.rlfname
        res2.toFile(self.rlfname)
        
        self.assert_(res2 == res0)
        
##################################################################
        
if __name__=='__main__':

    if (CHOICE == 1):
        suite = unittest.TestLoader().loadTestsFromTestCase(
                                   PDFOptFittingTest)
        unittest.TextTestRunner(verbosity=VERBOSITY).run(suite)
    else:
        unittest.main()
##################################################################
