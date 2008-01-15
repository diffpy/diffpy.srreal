#!/usr/bin/env python
#####################################################################

import os, time, sys
import traceback
#####################################################################

from parkUniFit import ParkUniFit
from parkUniFit import DEFAULT_MODEL_NUMBER

from scriptUtil import SetPDFEnviron

SetPDFEnviron()

from xmlParameter import XmlParameter
from xmlModel import XmlModel
from pdfData import PDFData
from pdfDataset import PDFDataset
from pdfTheory import PDFTheory, getPDFParameters
#####################################################################

from parkPDFFit import ParkPDFFit

def main():
        
    from xmlFitting import XmlFitting
    from diffpy.pdffit2 import PdfFit

    datadir = os.path.abspath('../examples/pdf')
    strufile = "Ni_4x4x4.stru"

    # Set up the PdfFit instance
    P = PdfFit()
    P.read_struct(os.path.join(datadir, strufile))

    p0 = [1]
        
    fit = ParkPDFFit(p0)
    modelNames = fit.getModelNames()
    data = os.path.join(datadir, 'Ni_2-8.dat')
    fit.setDataSource(modelNames[0], [data])        
    model = fit.getModel(modelNames[0])
    model.removeChild("p0")
    
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
        
    res = obj.doFitting()

    #print 'result:', res
    return
        
#####################################################################

if __name__ == '__main__':

    main()
    
#####################################################################
######################   EOF   ######################################        

