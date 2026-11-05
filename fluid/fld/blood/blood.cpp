#include "blood.hpp"

namespace fld::blood
{
    continuum::options configure(continuum::options settings,
                                 const material& blood)
    {
        settings.liquid.density=blood.density;
        settings.liquid.dynamic_viscosity=blood.viscosity_high;
        settings.liquid.surface_tension=blood.surface_tension;
        settings.liquid.advancing_angle=blood.advancing_angle;
        settings.liquid.receding_angle=blood.receding_angle;
        settings.liquid.index_of_refraction=1.35f;
        settings.flow.zero_shear_viscosity=blood.viscosity_zero;
        settings.flow.high_shear_viscosity=blood.viscosity_high;
        settings.flow.relaxation_seconds=blood.relaxation_seconds;
        settings.flow.power_index=blood.shear_exponent;
        settings.flow.substrate_drag=blood.substrate_drag;
        return settings;
    }

    continuum::shader_paths shaders(const std::string& directory)
    {
        const char* stages[]={
            "SEED","ADVECT_VELOCITY","CURVATURE","DIVERGENCE",
            "PRESSURE","PROJECT","ADVECT_VOLUME"
        };
        continuum::shader_paths paths{};
        for(std::size_t i=0;i<paths.simulation.size();++i)
            paths.simulation[i]=directory+"continuum_"+stages[i]+".comp.spv";
        paths.rendering={
            directory+"continuum_blood_surface.comp.spv",
            directory+"continuum_upscale.comp.spv"
        };
        return paths;
    }

    void init(continuum::options settings,
              const continuum::drop& volume,
              const continuum::shader_paths& programs,
              std::uint32_t frames,std::uint32_t width,std::uint32_t height,
              const material& blood)
    {
        continuum::init(configure(std::move(settings),blood),volume,programs,
                        frames,width,height);
    }

    bool active(){return continuum::active();}

    void record(vk::CommandBuffer command,std::uint32_t image,
                float seconds,const continuum::render_input& target)
    {
        continuum::record(command,image,seconds,target);
    }

    void shutdown(){continuum::shutdown();}
}
