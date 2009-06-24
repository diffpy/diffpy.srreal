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
* class ClassRegistry -- template class providing registry for factories
*   of various concrete classes.  Make sure that for a particular base,
*   this template class is used in just one cpp file, otherwise the registry
*   may not be unique.  Usually the ClassRegistry is used via
*   createSomeBase, registerSomeBase, aliasSomeBase wrappers.
*
* $Id$
*
*****************************************************************************/

#ifndef CLASSREGISTRY_HPP_INCLUDED
#define CLASSREGISTRY_HPP_INCLUDED

#include <string>
#include <map>
#include <set>
#include <stdexcept>
#include <sstream>
#include <memory>

#include <boost/shared_ptr.hpp>

namespace diffpy {

template <class TBase>
class ClassRegistry
{
    public:

        // types
        typedef std::map<std::string,
                boost::shared_ptr<const TBase> > RegistryType;

        // class methods

        static bool add(const TBase& prot)
        {
            using namespace std;
            RegistryType& reg = getRegistry();
            if (reg.count(prot.type()))
            {
                ostringstream emsg;
                emsg << "Prototype type '" << prot.type() <<
                    "' is already registered.";
                throw logic_error(emsg.str());
            }
            reg[prot.type()].reset(prot.copy());
            return true;
        }


        static bool alias(const std::string& tp, const std::string& al)
        {
            using namespace std;
            RegistryType& reg = getRegistry();
            if (!reg.count(tp))
            {
                ostringstream emsg;
                emsg << "Cannot create alias for unknown prototype '" <<
                    tp << "'.";
                throw logic_error(emsg.str());
            }
            if (reg.count(al) && reg[al] != reg[tp])
            {
                ostringstream emsg;
                emsg << "Prototype type '" << al <<
                    "' is already registered.";
                throw logic_error(emsg.str());
            }
            reg[al] = reg[tp];
            return true;
        }

        static TBase* create(const std::string& tp)
        {
            using namespace std;
            typename RegistryType::iterator irg;
            RegistryType& reg = getRegistry();
            irg = reg.find(tp);
            if (irg == reg.end())
            {
                ostringstream emsg;
                emsg << "Unknown type '" << tp << "'.";
                throw invalid_argument(emsg.str());
            }
            TBase* rv = irg->second->create();
            return rv;
        }


        static std::set<std::string> getTypes()
        {
            using namespace std;
            set<string> rv;
            RegistryType& reg = getRegistry();
            typename RegistryType::iterator irg;
            for (irg = reg.begin(); irg != reg.end(); ++irg)
            {
                rv.insert(irg->second->type());
            }
            return rv;
        }


    private:

        static RegistryType& getRegistry()
        {
            static std::auto_ptr<RegistryType> the_registry;
            if (!the_registry.get())
            {
                the_registry.reset(new RegistryType());
            }
            return *the_registry;
        }

};  // End of class ClassRegistry


}   // namespace diffpy

#endif  // CLASSREGISTRY_HPP_INCLUDED
