#pragma once
#include "../continuum/continuum.hpp"

namespace fld::blood
{
    // Blood is a material/presentation profile of the shared VOF/MAC fluid.
    // No independent blood particles, pressure grid, or collision solver.
    struct material
    {
        float density=1060.f;                 // kg/m^3, whole blood
        float viscosity_zero=.25f;           // Pa*s, low-shear Carreau fit
        float viscosity_high=.0035f;         // Pa*s, high-shear limit
        float relaxation_seconds=25.f;       // Carreau time constant
        float shear_exponent=.25f;           // n<1 gives shear thinning
        float surface_tension=.058f;         // N/m (approximate)
        float advancing_angle=1.4f;         // radians
        float receding_angle=1.1f;          // radians
        float substrate_drag=3.f;           // s^-1, visual contact damping
        continuum::vec3 shallow_color{.68f,.055f,.041f}; // linear RGB
        float optical_density=26.f;          // 1/m, thicker = darker

    };

    continuum::options configure(continuum::options settings,
                                 const material& blood={});
    continuum::shader_paths shaders(const std::string& directory);

    void init(continuum::options settings,
              const continuum::drop& volume,
              const continuum::shader_paths& programs,
              std::uint32_t frames,std::uint32_t width,std::uint32_t height,
              const material& blood={});
    bool active();
    void record(vk::CommandBuffer command,std::uint32_t image,
                float seconds,const continuum::render_input& target);
    void shutdown();
}
