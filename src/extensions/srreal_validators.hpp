/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   (c) 2013 Brookhaven Science Associates,
*                   Brookhaven National Laboratory.
*                   All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* Helper functions for argument checking.
*
*****************************************************************************/

#ifndef SRREAL_VALIDATORS_HPP_INCLUDED
#define SRREAL_VALIDATORS_HPP_INCLUDED

namespace srrealmodule {

void ensure_index_bounds(int idx, int lo, int hi);
void ensure_non_negative(int value);

}   // namespace srrealmodule

#endif  // SRREAL_VALIDATORS_HPP_INCLUDED
