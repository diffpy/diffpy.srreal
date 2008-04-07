#!/usr/bin/env python
#####################################################################

import os, time, sys
import traceback
#####################################################################

from park.script.parkUniFit import ParkUniFit
from park.script.parkUniFit import DEFAULT_MODEL_NUMBER

from park.fit.xmlParameter import XmlParameter
from park.fit.xmlModel import XmlModel
from pdfData import PDFData
from pdfDataset import PDFDataset
from pdfTheory import PDFTheory, getPDFParameters
#####################################################################

PDF_PARAM_NAME_PREFIX = "p"

PDF_THEORY_CLASS_NAME = "SrReal.park.pdfTheory.PDFTheory"

PDF_GUI_TYPE_NAME = 'pdf'
#####################################################################

""" The adapter class to simplify the interface between the script
    application and the framework of fitting service.
"""    
#####################################################################

#####################################################################
class ParkPDFFit(ParkUniFit):
    """ 
        The subclass class of ParkUnitFit where all the models are
        PDF model.
    """
#####################################################################    
    def __init__(self, nparams=None): 
        """ 
            constructor. 
            n: number of PDF models
            nparams: a list of n integers, indicating 
                the number of PDF parameters.
        """        
        super(self.__class__, self).__init__(nparams)
#####################################################################

    def _getDataset(self):
        """ Return the dataset for the model. The subclass must 
            implement this function.
        """
        return PDFDataset()
#####################################################################

    def _getData(self):
        """ Return the data for the model. The subclass must 
            implement this function.
        """
        return PDFData()
#####################################################################

    def _getTheoryClassName(self):
        """ Return the name of theory class.for the model. The  
            subclass must implement this function.
        """
        return PDF_THEORY_CLASS_NAME
#####################################################################
    
    def _getDefaultParameter(self):
        """ Return the default parameter for the model.  
            The subclass may override this function.
        """
        return XmlParameter()   
#####################################################################

    def _getParamNamePrefix(self):
        """ Return the prefix string to label the parameters in the model.  
            The subclass may override this function.
        """
        return PDF_PARAM_NAME_PREFIX 
#####################################################################

    def _getModelTypeName(self):
        """ Return the name of the type that can be handled by ParkGUI client. 
            subclass must implement this function.
        """
        return PDF_GUI_TYPE_NAME 
                
#####################################################################

def main():
        
    from park.fit.xmlFitting import XmlFitting
    from diffpy.pdffit2 import PdfFit

    datadir = os.path.abspath('examples')
    strufile = "Ni.stru"

    # Set up the PdfFit instance
    P = PdfFit()
    P.read_struct(os.path.join(datadir, strufile))

    fit = ParkPDFFit([0])
    modelNames = fit.getModelNames()
    data = os.path.join(datadir, 'Ni_2-8.dat')
    fit.setDataSource(modelNames[0], [data])        
    model = fit.getModel(modelNames[0])
    
    # set the initial values.
    for pm in getPDFParameters(P, "X", 35.0):
        model.addChild(pm)

    fit.updateVariables()
    # Freeze the variables.
    for var in fit.getVariables():
        var.setFixed()

    # Optimize the lattice parameters, thermal parameters, delta2 and scale
    vp ="%s.%s" % (model.name, pm.name)
    fit.getVariable("%s.%s"%(vp, "lat_1")).setOptimized()
    fit.setConstrain("%s.%s"%(vp,"lat_2"), "%s.%s"%(vp,"lat_1"))
    fit.setConstrain("%s.%s"%(vp,"lat_3"), "%s.%s"%(vp,"lat_1"))

    fit.getVariable("%s.%s"%(vp, "u11_1")).setOptimized()
    fit.setConstrain("%s.%s"%(vp, "u22_1"),"%s.%s"%(vp, "u11_1"))
    fit.setConstrain("%s.%s"%(vp, "u33_1"),"%s.%s"%(vp, "u11_1"))
    for i in range(2,257):
        fit.setConstrain("%s.%s_%i"%(vp,"u11",i),"%s.%s"%(vp, "u11_1"))
        fit.setConstrain("%s.%s_%i"%(vp,"u22",i),"%s.%s"%(vp, "u11_1"))
        fit.setConstrain("%s.%s_%i"%(vp,"u33",i),"%s.%s"%(vp, "u11_1"))

    delta2 = fit.getVariable("%s.%s"%(vp, "delta2"))
    delta2.range = [1,4]
    delta2.value = 2.7
    delta2.v0 = 2.7
    delta2.setOptimized()

    fit.getVariable("%s.%s"%(vp, "dscale")).setOptimized()
    fit.updateVariables()

    obj = XmlFitting()
    obj.parseString(fit.toStream())
    opt = obj.getXmlOptimizer()
    opt.optfunc = 'leastsq'
        
    res = obj.doFitting()

    #print 'result:', res

    return
        
#####################################################################

if __name__ == '__main__':

    main()
    
#####################################################################
######################   EOF   ######################################        

