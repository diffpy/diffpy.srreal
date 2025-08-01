#!/usr/bin/env python

"""Unit tests for pdfcalculator.py on ObjCryst crystal structures."""


import re
import unittest

import numpy
import pytest
from testutils import _maxNormDiff, datafile, loadObjCrystCrystal

from diffpy.srreal.pdfcalculator import PDFCalculator

# helper functions


def _loadExpectedPDF(basefilename):
    """Read expected result and return a tuple of (r, g, cfgdict)."""
    rxf = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")
    fullpath = datafile(basefilename)
    cfgdict = {}
    fp = open(fullpath)
    for line in fp:
        if line[:1] != "#":
            break
        w = line.split()
        has_cfgdata = len(w) == 4 and w[2] == "="
        if not has_cfgdata:
            continue
        cfgdict[w[1]] = w[3]
        if rxf.match(w[3]):
            cfgdict[w[1]] = float(w[3])
    fp.close()
    r, g = numpy.loadtxt(fullpath, usecols=(0, 1), unpack=True)
    rv = (r, g, cfgdict)
    return rv


def _makePDFCalculator(crst, cfgdict):
    """Return a PDFCalculator object evaluated for a pyobjcryst.Crystal
    crst."""
    pdfcargs = {k: v for k, v in cfgdict.items() if k not in ("biso", "type")}
    pdfc = PDFCalculator(**pdfcargs)
    if "biso" in cfgdict:
        reg = crst.GetScatteringPowerRegistry()
        for i in range(reg.GetNb()):
            sp = reg.GetObj(i)
            sp.SetBiso(cfgdict["biso"])
    if "type" in cfgdict:
        pdfc.scatteringfactortable = cfgdict["type"]
    pdfc.eval(crst)
    # avoid metadata override by PDFFitStructure
    for k, v in pdfcargs.items():
        setattr(pdfc, k, v)
    return pdfc


# ----------------------------------------------------------------------------


class TestPDFCalcObjcryst(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _check_pyobjcryst(self, has_pyobjcryst, _msg_nopyobjcryst):
        if not has_pyobjcryst:
            pytest.skip(_msg_nopyobjcryst)

    def _comparePDFs(self, nickname, pdfbasename, cifbasename):
        def setself(**kwtoset):
            for n, v in kwtoset.items():
                setattr(self, nickname + "_" + n, v)
            return

        r, gobs, cfg = _loadExpectedPDF(pdfbasename)
        setself(r=r, gobs=gobs, cfg=cfg)
        crst = loadObjCrystCrystal(cifbasename)
        setself(crst=crst)
        pdfc = _makePDFCalculator(crst, cfg)
        gcalc = pdfc.pdf
        mxnd = _maxNormDiff(gobs, gcalc)
        setself(gcalc=gcalc, mxnd=mxnd)
        return

    def test_CdSeN(self):
        """Check PDFCalculator on ObjCryst loaded CIF, neutrons."""
        self._comparePDFs(
            "cdsen", "CdSe_cadmoselite_N.fgr", "CdSe_cadmoselite.cif"
        )
        self.assertTrue(self.cdsen_mxnd < 0.01)
        return

    def test_CdSeX(self):
        """Check PDFCalculator on ObjCryst loaded CIF, xrays."""
        self._comparePDFs(
            "cdsex", "CdSe_cadmoselite_X.fgr", "CdSe_cadmoselite.cif"
        )
        self.assertTrue(self.cdsex_mxnd < 0.01)
        return

    def test_rutileaniso(self):
        """Check PDFCalculator on ObjCryst loaded anisotropic rutile."""
        self._comparePDFs(
            "rutileaniso", "TiO2_rutile-fit.fgr", "TiO2_rutile-fit.cif"
        )
        self.assertTrue(self.rutileaniso_mxnd < 0.057)
        return


# End of class TestPDFCalcObjcryst

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()

# End of file
