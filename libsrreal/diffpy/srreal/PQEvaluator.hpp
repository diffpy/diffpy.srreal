/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* class PQEvaluatorBasic -- robust PairQuantity evaluator, the result
*     is always calculated from scratch.

* class PQEvaluatorOptimized -- optimized PairQuantity evaluator with fast
*     quantity updates
*
* $Id$
*
*****************************************************************************/


#ifndef PQEVALUATOR_HPP_INCLUDED
#define PQEVALUATOR_HPP_INCLUDED

namespace diffpy {
namespace srreal {

class PairQuantity;

enum PQEvaluatorType {BASIC, OPTIMIZED};

class PQEvaluatorBasic
{
    public:

        // methods
        virtual PQEvaluatorType typeint() const;
        virtual void updateValue(PairQuantity& pq);

};


class PQEvaluatorOptimized : public PQEvaluatorBasic
{
    public:

        // methods
        virtual PQEvaluatorType typeint() const;
        virtual void updateValue(PairQuantity& pq);

};


// Factory function for PairQuantity evaluators ------------------------------

PQEvaluatorBasic* createPQEvaluator(PQEvaluatorType pqtp);


}   // namespace srreal
}   // namespace diffpy

#endif  // PQEVALUATOR_HPP_INCLUDED
