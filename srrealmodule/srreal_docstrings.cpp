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

}   // namespace srrealmodule

// End of file
