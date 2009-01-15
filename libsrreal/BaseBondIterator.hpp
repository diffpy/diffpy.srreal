#ifndef BASEBONDITERATOR_HPP_INCLUDED
#define BASEBONDITERATOR_HPP_INCLUDED

#include <memory>

namespace diffpy {

class BaseStructure;
class BaseBondPair;

class BaseBondIterator
{
    public:

        // constructor
        BaseBondIterator(const BaseStructure*);

        // methods
        // loop control
        virtual void rewind();
        bool finished() const;
        virtual void next();

        // configuration
        void selectAnchorSite(int);
        void selectSiteRange(int first, int last);

        // get data
        const BaseBondPair& getBondPair() const;

    protected:

        // data
        int msite_anchor;
        int msite_first;
        int msite_last;
        int msite_current;
        const BaseStructure* mstructure;
        std::auto_ptr<BaseBondPair> mbond_pair;

        // methods
        virtual bool iterateSymmetry();

    private:

        // methods
        void setFinishedFlag();

};


}   // namespace diffpy

#endif  // BASEBONDITERATOR_HPP_INCLUDED
