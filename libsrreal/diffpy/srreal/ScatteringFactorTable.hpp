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
* class ScatteringFactorTable -- base class for looking up scattering factors
*
* $Id$
*
*****************************************************************************/

#ifndef SCATTERINGFACTORTABLE_HPP_INCLUDED
#define SCATTERINGFACTORTABLE_HPP_INCLUDED

#include <string>
#include <map>
#include <set>

namespace diffpy {
namespace srreal {

class ScatteringFactorTable
{
    public:

        // constructors
        virtual ScatteringFactorTable* create() const = 0;
        virtual ScatteringFactorTable* copy() const = 0;
        ~ScatteringFactorTable()  { }

        // methods
        virtual const std::string& type() const = 0;
        virtual const std::string& radiationType() const = 0;
        const double& lookup(const std::string& smbl) const;
        void setCustom(const std::string& smbl, double value);
        void resetCustom(const std::string& smbl);
        void resetAll();

    protected:

        virtual double fetch(const std::string& smbl) const = 0;
        mutable std::map<std::string,double> mtable;
};

// Factory functions for Scattering Factor Tables ----------------------------

ScatteringFactorTable* createScatteringFactorTable(const std::string& tp);
bool registerScatteringFactorTable(const ScatteringFactorTable&);
bool aliasScatteringFactorTable(const std::string& tp, const std::string& al);
std::set<std::string> getScatteringFactorTableTypes();

}   // namespace srreal
}   // namespace diffpy

#endif  // SCATTERINGFACTORTABLE_HPP_INCLUDED
