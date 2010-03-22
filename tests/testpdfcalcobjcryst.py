#!/usr/bin/env python

"""Unit tests for pdfcalculator.py on ObjCryst crystal structures
"""

# version
__id__ = '$Id$'

import os
import re
import unittest

import numpy
from diffpy.srreal.pdfcalculator import PDFCalculator

# useful variables
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')

# helper functions

def _loadTestStructure(basefilename):
    from pyobjcryst.crystal import CreateCrystalFromCIF
    fullpath = os.path.join(testdata_dir, basefilename)
    crst = CreateCrystalFromCIF(open(fullpath))
    return crst


def _loadExpectedPDF(basefilename):
    '''Read expected result and return a tuple of (r, g, cfgdict).
    '''
    rxf = re.compile(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?')
    fullpath = os.path.join(testdata_dir, basefilename)
    cfgdict = {}
    for line in open(fullpath):
        if line[:1] != '#':  break
        w = line.split()
        has_cfgdata = (len(w) == 4 and w[2] == '=')
        if not has_cfgdata:  continue
        cfgdict[w[1]] = w[3]
        if rxf.match(w[3]):  cfgdict[w[1]] = float(w[3])
    r, g = numpy.loadtxt(fullpath, usecols=(0, 1), unpack=True)
    rv = (r, g, cfgdict)
    return rv


def _makePDFCalculator(crst, cfgdict):
    '''Return a PDFCalculator object evaluated for a pyobjcryst.Crystal crst.
    '''
    inpdfcalc = lambda kv: kv[0] not in ('biso', 'type')
    pdfcargs = dict(filter(inpdfcalc, cfgdict.items()))
    pdfc = PDFCalculator(**pdfcargs)
    if 'biso' in cfgdict:
        setbiso = lambda sc: sc.mpScattPow.SetBiso(cfgdict['biso'])
        map(setbiso, crst.GetScatteringComponentList())
    if 'type' in cfgdict:
        pdfc.setScatteringFactorTable(cfgdict['type'])
    pdfc.eval(crst)
    # avoid metadata override by PDFFitStructure
    for k, v in pdfcargs.items():
        setattr(pdfc, k, v)
    return pdfc


def _maxNormDiff(yobs, ycalc):
    '''Returned maximum difference normalized by RMS of the yobs
    '''
    yobsa = numpy.array(yobs)
    yrms = numpy.sqrt(numpy.mean(yobsa ** 2))
    ynmdiff = (yobsa - ycalc) / yrms
    rv = max(numpy.fabs(ynmdiff))
    return rv


##############################################################################
class TestPDFCalcObjcryst(unittest.TestCase):

    # constants

    tol_maxnormdiff = 1e-6


    def _comparePDFs(self, nickname, pdfbasename, cifbasename):
        def setself(**kwtoset):
            for n, v in kwtoset.iteritems():
                setattr(self, nickname + '_' + n, v)
            return
        r, gobs, cfg = _loadExpectedPDF(pdfbasename)
        setself(r=r, gobs=gobs, cfg=cfg)
        crst = _loadTestStructure(cifbasename)
        setself(crst=crst)
        pdfc = _makePDFCalculator(crst, cfg)
        gcalc = pdfc.getPDF()
        mxnd = _maxNormDiff(gobs, gcalc)
        setself(gcalc=gcalc, mxnd=mxnd)
        return


    def test_CdSeN(self):
        self._comparePDFs('cdsen',
                'CdSe_cadmoselite_N.fgr', 'CdSe_cadmoselite.cif')
        self.failUnless(self.cdsen_mxnd < self.tol_maxnormdiff)
        return


    def test_CdSeX(self):
        self._comparePDFs('cdsex',
                'CdSe_cadmoselite_X.fgr', 'CdSe_cadmoselite.cif')
        self.failUnless(self.cdsex_mxnd < self.tol_maxnormdiff)
        return


    def test_rutileaniso(self):
        self._comparePDFs('rutileaniso',
                'TiO2_rutile-fit.fgr', 'TiO2_rutile-fit.cif')
        print self.rutileaniso_mxnd
        self.failUnless(self.rutileaniso_mxnd < 0.012)
        return


# End of class TestPDFCalcObjcryst


if __name__ == '__main__':
#   unittest.main()
    # temporary comparison of diffpy.Structure and objcryst results
    # to be removed once this gets fixed.
    from diffpy.Structure import Structure
    r, gobs, cfg = _loadExpectedPDF('TiO2_rutile-fit.fgr')
    stru = Structure(filename='testdata/TiO2_rutile-fit.cif')
    pcstru = _makePDFCalculator(stru, cfg)
    crst = _loadTestStructure('TiO2_rutile-fit.cif')
    pccrst = _makePDFCalculator(crst, cfg)
    import pylab
    pylab.plot(r, gobs, pcstru.getRgrid(), pcstru.getPDF(),
            pccrst.getRgrid(), pccrst.getPDF())
    pylab.show()
