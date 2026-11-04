#pragma once
#include "../core/types.hpp"

namespace fld::wetting
{
    struct diagnostics
    {
        float reynolds=0, weber=0, ohnesorge=0, splash_parameter=0;
        bool splash=false;
    };

    // Generic response for every fluid phase and every receiving substrate.
    // Calibrated dimensionless impact approximation, not a molecular wetting
    // model. Caller supplies the actual target surface parameters.
    diagnostics resolve(const material &fluid, const particle &drop,
                        collision &contact);
}
