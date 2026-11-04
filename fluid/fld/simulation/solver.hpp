#pragma once
#include "../core/system.hpp"

namespace fld::simulation
{
    // Parallel-friendly Jacobi density solve: poly6/spiky kernels,
    // PBF tensile-instability correction, physical viscosity,
    // continuous-segment collisions. No render or network dependency.
    void solve(system::state &world, float dt,
               const collision_callback &collision,
               const contact_callback &contact);
}
