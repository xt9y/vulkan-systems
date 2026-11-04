#include "wounds.hpp"
#include "../surfaces/stains.hpp"
#include "../core/system.hpp"
#include <algorithm>
#include <stdexcept>

namespace bld::render::wounds
{
    namespace
    {
        state *data=nullptr;

        vk::buffer::resource allocate(std::size_t bytes)
        {
            return vk::buffer::create(
                std::max<std::size_t>(sizeof(std::uint32_t),bytes),
                vk::BufferUsageFlagBits::eStorageBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
        }

        void destroy(cache &item)
        {
            for(auto &buffer:item.buffers)
                if(buffer.handle)vk::buffer::destroy(buffer);
            item.buffers.clear();
            item.generations.clear();
        }

        vk::DescriptorBufferInfo write(cache &entry,std::uint32_t frame,
                                       const std::vector<std::uint32_t> &pixels,
                                       std::uint64_t version)
        {
            if(entry.buffers.empty())
            {
                entry.buffers.resize(data->frames);
                entry.generations.assign(data->frames,~std::uint64_t{0});
            }
            auto &buffer=entry.buffers.at(frame);
            const auto bytes=pixels.size()*sizeof(std::uint32_t);
            if(!buffer.handle || buffer.size<bytes)
            {
                if(buffer.handle)vk::buffer::destroy(buffer);
                buffer=allocate(bytes);
                entry.generations.at(frame)=~std::uint64_t{0};
            }
            if(entry.generations.at(frame)!=version)
            {
                vk::buffer::write(buffer,pixels.data(),bytes);
                entry.generations.at(frame)=version;
            }
            return vk::buffer::descriptor(buffer);
        }
    }

    state *get() { return data; }

    void init(std::uint32_t frames)
    {
        if(data)throw std::logic_error("Blood GPU already initialized");
        if(!frames)throw std::invalid_argument("Invalid frame count");
        data=new state{};
        data->frames=frames;
        try
        {
            const auto system=bld::system::get();
            const auto resolution=system?system->settings.wound_resolution:256u;
            const std::vector<std::uint32_t> zeros(
                std::size_t(resolution)*resolution,0u);
            data->fallback=allocate(zeros.size()*sizeof(std::uint32_t));
            vk::buffer::write(data->fallback,zeros.data(),
                              zeros.size()*sizeof(std::uint32_t));
        }
        catch(...)
        {
            shutdown();
            throw;
        }
    }

    bool visible(instance_id id)
    {
        const auto *a=surface::find(id);
        return a && !a->marks.empty();
    }

    vk::DescriptorBufferInfo descriptor(instance_id id,std::uint32_t frame)
    {
        if(!data || frame>=data->frames)
            throw std::invalid_argument("Invalid GPU frame");
        const auto *a=surface::find(id);
        if(!a || a->pixels.empty())
            return vk::buffer::descriptor(data->fallback);
        return write(data->actors[id],frame,a->pixels,a->generation);
    }

    vk::DescriptorBufferInfo descriptor_batch(
        std::uint32_t mesh,const std::vector<instance_id> &ids,
        std::uint32_t frame)
    {
        if(!data || frame>=data->frames || !mesh)
            throw std::invalid_argument("Invalid GPU batch");
        const auto *settings=bld::system::get();
        const auto resolution=settings?
            settings->settings.wound_resolution:256u;
        const std::size_t words=std::size_t(resolution)*resolution;
        if(ids.empty())return vk::buffer::descriptor(data->fallback);

        bool any=false;
        std::uint64_t signature=14695981039346656037ull;
        for(auto id:ids)
        {
            const auto *a=surface::find(id);
            signature=(signature^id)*1099511628211ull;
            signature=(signature^(a?a->generation:0ull))*1099511628211ull;
            if(a && !a->pixels.empty())any=true;
        }
        if(!any)return vk::buffer::descriptor(data->fallback);

        auto &entry=data->batches[mesh];
        if(!entry.buffers.empty() && entry.generations.at(frame)==signature)
            return vk::buffer::descriptor(entry.buffers.at(frame));
        std::vector<std::uint32_t> packed(ids.size()*words,0u);
        for(std::size_t i=0;i<ids.size();++i)
        {
            const auto *a=surface::find(ids[i]);
            if(a && a->pixels.size()==words)
                std::copy(a->pixels.begin(),a->pixels.end(),
                          packed.begin()+i*words);
        }
        return write(entry,frame,packed,signature);
    }

    void clear_batches()
    {
        if(!data)return;
        for(auto &[id,item]:data->batches)destroy(item);
        data->batches.clear();
    }

    void shutdown()
    {
        if(!data)return;
        clear_batches();
        for(auto &[id,item]:data->actors)destroy(item);
        if(data->fallback.handle)vk::buffer::destroy(data->fallback);
        delete data;
        data=nullptr;
    }
}
