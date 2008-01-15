#!/usr/bin/env python
#####################################################################

import os, time, sys
import traceback
#####################################################################

from park.script.parkFitClient import ParkFitClient
from parkPDFFit import ParkPDFFit

from pdfTheory import getPDFParameters

#####################################################################

def buildPDFFitting():        
    from park.fit.xmlFitting import XmlFitting
    from diffpy.pdffit2 import PdfFit, redirect_stdout

    datadir = os.path.abspath('examples')
    strufile = "Ni.stru"

    # Set up the PdfFit instance
    redirect_stdout(os.tmpfile())
    P = PdfFit()
    P.read_struct(os.path.join(datadir, strufile))

    p0 = [0]
        
    fit = ParkPDFFit(p0)
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
    for i in range(2,5):
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

    return fit
        
        
#####################################################################

""" 
    An example to show how to write a script-based service client
    application: build models for the fitting, submit the request,
    wait for the reply.
"""    
#####################################################################

def main():
    """ An example for PDF fitting """
    client = ParkFitClient()
    
    req = buildPDFFitting()
    client.submit(req, 1)
    
    while (not client.isCompleted()):
        job = client.getJob()
        # job is type of ParkFit
        #print 'job', job
        print 'job overview:', job.getFittingResultOverview()
    
    del client

#####################################################################

if __name__ == '__main__':
    
    main()
    
#####################################################################
######################   EOF   ######################################        

