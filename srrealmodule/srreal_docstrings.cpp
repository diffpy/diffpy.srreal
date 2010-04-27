/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2010 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Python docstrings for the wrapped functions and classes.
*
* $Id$
*
*****************************************************************************/

namespace srrealmodule {

// BasePairQuantity ----------------------------------------------------------

const char* doc_BasePairQuantity_value = "\
Return total internal contributions as numpy array.\n\
";

const char* doc_BasePairQuantity_eval = "\
Calculate a pair quantity for the specified structure.\n\
\n\
stru -- structure object that can be converted to StructureAdapter\n\
\n\
Return a copy of the internal total contributions.\n\
May need to be further transformed to get the desired value.\n\
";

const char* doc_PairQuantityWrap__value = "\
Reference to the internal vector of total contributions.\n\
";

// PeakProfile ---------------------------------------------------------------

const char* doc_PeakProfile_getRegisteredTypes = "\
Set of string identifiers for registered PeakProfile classes.\n\
These are allowed arguments for the createByType static method.\n\
";

// PDFBaseline ---------------------------------------------------------------

const char* doc_PDFBaseline_getRegisteredTypes = "\
Set of string identifiers for registered PDFBaseline classes.\n\
These are allowed arguments for the createByType static method.\n\
";

// PDFEnvelope ---------------------------------------------------------------

const char* doc_PDFEnvelope_getRegisteredTypes = "\
Set of string identifiers for registered PDFEnvelope classes.\n\
These are allowed arguments for the createByType static method.\n\
";

// DebyePDFCalculator and PDFCalculator --------------------------------------

const char* doc_getPeakWidthModelTypes = "\
Set of string identifiers for registered PeakWidthModel classes.\n\
These are allowed arguments for the setPeakWidthModel method.\n\
";

const char* doc_getScatteringFactorTableTypes = "\
Set of string identifiers for registered ScatteringFactorTable classes.\n\
These are allowed arguments for the setScatteringFactorTable method.\n\
";

// BVSCalculator -------------------------------------------------------------

const char* doc_BVSCalculator_valences = "\
Return valences expected at each site of the evaluated structure.\n\
";

const char* doc_BVSCalculator_bvdiff = "\
Difference between expected and calculated valence magnitudes at each site.\n\
Positive for underbonding, negative for overbonding.\n\
";

const char* doc_BVSCalculator_bvmsdiff = "\
Mean square difference between expected and calculated valences.\n\
Adjusted for multiplicity and occupancy of atom sites in the structure.\n\
";

const char* doc_BVSCalculator_bvrmsdiff = "\
Root mean square difference between expected and calculated valences.\n\
Adjusted for multiplicity and occupancy of atom sites in the structure.\n\
";

}   // namespace srrealmodule

// End of file
