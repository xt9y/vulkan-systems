#include "renderer.hpp"
#include "../core/system.hpp"
#include "../material/blood.hpp"
#include <stdexcept>

namespace bld::render
{
    namespace { state *data=nullptr; }
    state *get() { return data; }

    void init(const std::string &shader_path,std::uint32_t frames,
              std::uint32_t width,std::uint32_t height)
    {
        if(data)throw std::logic_error("Blood renderer initialized twice");
        data=new state{};
        try
        {
            if(!fld::render::get())
            {
                fld::render::init(shader_path,frames,width,height);
                data->owns_renderer=true;
            }
        }
        catch(...)
        {
            delete data;
            data=nullptr;
            throw;
        }
    }

    bool active()
    {
        if(!system::get() || !fld::system::get())return false;
        for(const auto &drop:fld::system::get()->particles)
            if(drop.phase==system::get()->phase)return true;
        return false;
    }

    void record(vk::CommandBuffer command,std::uint32_t frame,
                const input &inputs)
    {
        if(!data || !system::get())
            throw std::logic_error("Blood renderer not initialized");
        input target=inputs;
        target.phase=system::get()->phase;
        target.use_phase_appearance=true;
        fld::render::record(command,frame,target);
    }

    void shutdown()
    {
        if(!data)return;
        if(data->owns_renderer)fld::render::shutdown();
        delete data;
        data=nullptr;
    }
}
