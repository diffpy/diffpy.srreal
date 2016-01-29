#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     Complex Modeling Initiative
#                   (c) 2016 Brookhaven Science Associates,
#                   Brookhaven National Laboratory.
#                   All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################


"""\
Compositional averaging of atom scattering factors.


Examples
--------
::

    import numpy as np
    from diffpy.srreal.scatteringfactortable import SFTXray
    xtb = SFTXray()
    qa = np.linspace(0, 25)
    sfavg1 = SFAverage.fromComposition(xtb, {'Na' : 1, 'Cl' : 1}, qa)
    sfavg2 = SFAverage.fromComposition(xtb, [('Mn', 1), ('O', 2)], qa)

    from diffpy.Structure import loadStructure
    from diffpy.srreal.pdfcalculator import DebyePDFCalculator
    dpdfc = DebyePDFCalculator()
    dpdfc(loadStructure('rutile.cif'))
    sfavg3 = SFAverage.fromStructure(dpdfc.scatteringfactortable,
                                     dpdfc.getStructure(), dpdfc.qgrid)
"""


# class SFAverage ------------------------------------------------------------

class SFAverage(object):
    """\
    Calculate compositional statistics of atom scattering factors.

    Compositional averages can be calculated for an array of Q-values.
    Results are stored in the class attributes.

    Attributes
    ----------
    f1sum :
        Sum of scattering factors from all atoms.
        Float or NumPy array.
    f2sum :
        Sum of squared scattering factors from all atoms.
        Float or NumPy array.
    count :
        Total number of atoms.  Can be non-integer in case of
        fractional occupancies.
    f1avg :
        Compositional average of scattering factors.
        Float or NumPy array.
    f2avg :
        Compositional average of squared scattering factors.
        Float or NumPy array.
    composition :
        Dictionary of atom symbols and their total abundancies.
    """

    f1sum = 0
    f2sum = 0
    count = 0
    f1avg = 0
    f2avg = 0
    composition = None

    @classmethod
    def fromStructure(cls, sftb, stru, q=0):
        """\
        Calculate average scattering factors from a structure object.

        Parameters
        ----------
        sftb : ScatteringFactorTable
            The ScatteringFactorTable object for looking up the values.
        stru : diffpy Structure or pyobjcryst Crystal or StructureAdapter
            The structure object that stores the atom species and their
            occupancies.  Can be any type with a registered conversion
            to the StructureAdapter class.
        q : float or NumPy array (optional)
            The Q value in inverse Angstroms for which to lookup
            the scattering factor values.

        See also
        --------
        RegisterStructureAdapter : to add support for more structure types.

        Returns
        -------
        SFAverage
            The calculated scattering factor averages.
        """
        # a bit of duck-typing for faster handling of diffpy.Structure
        if hasattr(type(stru), 'composition'):
            composition = stru.composition
            if isinstance(composition, dict):
                return cls.fromComposition(sftb, composition, q)
        # otherwise let's convert to a known structure type
        from diffpy.srreal.structureadapter import createStructureAdapter
        adpt = createStructureAdapter(stru)
        composition = {}
        for i in range(adpt.countSites()):
            smbl = adpt.siteAtomType(i)
            cnt = adpt.siteOccupancy(i) * adpt.siteMultiplicity(i)
            composition[smbl] = composition.get(smbl, 0) + cnt
        return cls.fromComposition(sftb, composition, q)


    @classmethod
    def fromComposition(cls, sftb, composition, q=0):
        """\
        Calculate average scattering factors from atom concentrations.

        Parameters
        ----------
        sftb : ScatteringFactorTable
            The ScatteringFactorTable object for looking up the values.
        composition : dictionary or a list of (symbol, amount) pairs.
            The chemical composition for evaluating the average.  Atom
            symbols may repeat when it is a list of (symbol, amount) pairs.
        q : float or NumPy array (optional)
            The Q value in inverse Angstroms for which to lookup
            the scattering factor values.

        Returns
        -------
        SFAverage
            The calculated scattering factor averages.
        """
        sfa = cls()
        sfa.composition = {}
        if isinstance(composition, dict):
            sfa.composition.update(composition)
        else:
            for smbl, cnt in composition:
                if not smbl in sfa.composition:
                    sfa.composition[smbl] = 0
                sfa.composition[smbl] += cnt
        sfa.f1sum = 0.0 * q
        sfa.f2sum = 0.0 * q
        for smbl, cnt in sfa.composition.items():
            sfq = sftb.lookup(smbl, q)
            sfa.f1sum += cnt * sfq
            sfa.f2sum += cnt * sfq**2
            sfa.count += cnt
        denom = sfa.count if sfa.count > 0 else 1
        sfa.f1avg = sfa.f1sum / denom
        sfa.f2avg = sfa.f2sum / denom
        return sfa

# End of class SFAverage
