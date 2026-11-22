/*****************************************************************************
*
* diffpy.srreal     Complex Modeling Initiative
*                   (c) 2026 The Trustees of Columbia University
*                   in the City of New York.  All rights reserved.
*
* See AUTHORS.rst for a list of people who contributed.
* See LICENSE.rst for license information.
*
******************************************************************************
*
* Bindings to the selection of the calculation kernel in libdiffpy.
*
*****************************************************************************/

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <diffpy/srreal/CalculationKernel.hpp>

namespace nb = nanobind;

namespace srrealmodule {
namespace nswrap_CalculationKernel {

// docstrings ----------------------------------------------------------------

const char* doc_availableCalculationKernels = "\
Return the kernels this machine can run, CPU kernels first in order of\n\
preference.  CUDA builds add 'cuda' and 'cuda-mixed' on compatible\n\
devices; these fall back to the CPU for masked sums, custom peak\n\
profiles and per-bond evaluations.\n\
";

const char* doc_getCalculationKernel = "\
Return the name of the calculation kernel in use.\n\
";

const char* doc_setCalculationKernel = "\
Use the named calculation kernel in all calculators.\n\
\n\
name     -- one of the names from availableCalculationKernels\n\
\n\
Raise ValueError when the kernel is not available.  Do not call this\n\
while a calculator evaluates in another thread.\n\
";

}   // namespace nswrap_CalculationKernel

// Wrapper definition --------------------------------------------------------

void wrap_CalculationKernel(nb::module_& m)
{
    using namespace nswrap_CalculationKernel;
    using namespace diffpy::srreal;

    m.def("availableCalculationKernels", availableCalculationKernels,
            doc_availableCalculationKernels);
    m.def("getCalculationKernel", getCalculationKernel,
            doc_getCalculationKernel);
    m.def("setCalculationKernel", setCalculationKernel,
            nb::arg("name"), doc_setCalculationKernel);
}

}   // namespace srrealmodule

// End of file
