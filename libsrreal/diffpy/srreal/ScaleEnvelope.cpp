/*****************************************************************************
*
* diffpy.srreal     by DANSE Diffraction group
*                   Simon J. L. Billinge
*                   (c) 2009 Trustees of the Columbia University
*                   in the City of New York.  All rights reserved.
*
* File coded by:    Christopher Farrow, Pavol Juhas
*
* See AUTHORS.txt for a list of people who contributed.
* See LICENSE.txt for license information.
*
******************************************************************************
*
* class ScaleEnvelope -- constant scaling factor
*
* $Id$
*
*****************************************************************************/

#include <diffpy/srreal/PDFEnvelope.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

class ScaleEnvelope : public PDFEnvelope
{
    public:

        // constructors

        ScaleEnvelope()
        {
            this->setScale(1.0);
        }


        PDFEnvelope* create() const
        {
            PDFEnvelope* rv = new ScaleEnvelope();
            return rv;
        }


        PDFEnvelope* copy() const
        {
            PDFEnvelope* rv = new ScaleEnvelope(*this);
            return rv;
        }

        // methods

        const string& type() const
        {
            static string rv = "scale";
            return rv;
        }


        double operator()(const double& r) const
        {
            return this->getScale();
        }


        void setScale(double sc)
        {
            mscale = sc;
        }


        const double& getScale() const
        {
            return mscale;
        }

    private:

        double mscale;

};  // class ScaleEnvelope

// Registration --------------------------------------------------------------

bool reg_ScaleEnvelope = registerPDFEnvelope(ScaleEnvelope());

}   // namespace srreal
}   // namespace diffpy

// End of file
