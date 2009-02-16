// vim:set ft=cpp:

#ifndef BASEPEAKWIDTHMODEL_IPP_INCLUDED
#define BASEPEAKWIDTHMODEL_IPP_INCLUDED

#include <sstream>
#include <stdexcept>

namespace diffpy {
namespace srreal {

// class BasePeakWidthModel --------------------------------------------------

inline
std::map<std::string, BasePeakWidthModel*>& BasePeakWidthModel::getRegistry()
{
    static std::map<std::string, BasePeakWidthModel*> the_registry;
    return the_registry;
}

// Factory Functions ---------------------------------------------------------

inline
BasePeakWidthModel* createPeakWidthModel(const std::string& tp)
{
    using namespace std;
    map<string, BasePeakWidthModel*>& reg = BasePeakWidthModel::getRegistry();
    map<string, BasePeakWidthModel*>::iterator ipwm;
    ipwm = reg.find(tp);
    if (ipwm == reg.end())
    {
        stringstream emsg;
        emsg << "Unknown type of BasePeakWidthModel '" << tp << "'.";
        throw invalid_argument(emsg.str());
    }
    BasePeakWidthModel* rv = ipwm->second->create();
    return rv;
}


inline
bool registerPeakWidthModel(const BasePeakWidthModel& pwm)
{
    using namespace std;
    map<string, BasePeakWidthModel*>& reg = BasePeakWidthModel::getRegistry();
    if (reg.count(pwm.type()))
    {
        stringstream emsg;
        emsg << "BasePeakWidthModel type '" << pwm.type() <<
            "' is already registered.";
        throw logic_error(emsg.str());
    }
    reg[pwm.type()] = pwm.create();
    return true;
}


}   // namespace srreal
}   // namespace diffpy



#endif  // BASEPEAKWIDTHMODEL_IPP_INCLUDED
