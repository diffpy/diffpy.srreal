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
* class QResolutionEnvelope -- constant scaling factor
*
* $Id$
*
*****************************************************************************/

#include <cmath>

#include <diffpy/srreal/PDFEnvelope.hpp>

using namespace std;

namespace diffpy {
namespace srreal {

class QResolutionEnvelope : public PDFEnvelope
{
    public:

        // constructors

        QResolutionEnvelope()
        {
            this->setQDamp(0.0);
        }


        PDFEnvelope* create() const
        {
            PDFEnvelope* rv = new QResolutionEnvelope();
            return rv;
        }


        PDFEnvelope* copy() const
        {
            PDFEnvelope* rv = new QResolutionEnvelope(*this);
            return rv;
        }

        // methods

        const string& type() const
        {
            static string rv = "qresolution";
            return rv;
        }


        double operator()(const double& r) const
        {
            double rv = (mqdamp > 0.0) ?
                exp(-pow(r * mqdamp, 2) / 2) :
                1.0;
            return rv;
        }


        void setQDamp(double sc)
        {
            mqdamp = sc;
        }


        const double& getQDamp() const
        {
            return mqdamp;
        }

    private:

        double mqdamp;

};  // class QResolutionEnvelope

// Registration --------------------------------------------------------------

bool reg_QResolutionEnvelope = registerPDFEnvelope(QResolutionEnvelope());

}   // namespace srreal
}   // namespace diffpy

// End of file
