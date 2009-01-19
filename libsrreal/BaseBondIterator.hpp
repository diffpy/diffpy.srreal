#ifndef BASEBONDITERATOR_HPP_INCLUDED
#define BASEBONDITERATOR_HPP_INCLUDED

#include <memory>
#include "R3linalg.hpp"

namespace diffpy {

class BaseStructure;

class BaseBondIterator
{
    public:

        // constructor
        BaseBondIterator(const BaseStructure*);

        // methods
        // loop control
        void rewind();
        bool finished() const;
        void next();

        // configuration
        void selectAnchorSite(int);
        void selectSiteRange(int first, int last);
        void includeSelfPairs(bool);

        // get data
        virtual const R3::Vector& r0() const = 0;
        virtual const R3::Vector& r1() const = 0;
        double distance() const;

    protected:

        // data
        int msite_anchor;
        int msite_first;
        int msite_last;
        int msite_current;
        bool minclude_self_pairs;
        const BaseStructure* mstructure;

        // bond data
        R3::Vector mbond_r0;
        R3::Vector mbond_r1;

        // methods
        virtual bool iterateSymmetry();

    private:

        // methods
        void skipSelfPair();
        void setFinishedFlag();

};


}   // namespace diffpy

#endif  // BASEBONDITERATOR_HPP_INCLUDED
