#include "wounds.hpp"
#include "../surfaces/stains.hpp"
#include "../core/system.hpp"
#include <algorithm>
#include <limits>
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
            // Zero instances = clean mask, even when accidentally sampled.
            const std::uint32_t empty=0;
            data->fallback=allocate(sizeof(empty));
            vk::buffer::write(data->fallback,&empty,sizeof(empty));
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
        return a && (!a->marks.empty() ||
               !a->baked_pixels.empty());
    }

    vk::DescriptorBufferInfo descriptor(instance_id id,std::uint32_t frame)
    {
        if(!data || frame>=data->frames)
            throw std::invalid_argument("Invalid GPU frame");
        const auto *a=surface::find(id);
        if(!a || a->pixels.empty())
            return vk::buffer::descriptor(data->fallback);
        // Standalone descriptor uses the same sparse batch format as
        // descriptor_batch(), with a single occupied slot.
        std::vector<std::uint32_t> packed;
        packed.reserve(a->pixels.size()+2);
        packed.push_back(1);
        packed.push_back(0);
        packed.insert(packed.end(),a->pixels.begin(),a->pixels.end());
        return write(data->actors[id],frame,packed,a->generation);
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
        std::size_t stained=0;
        for(auto id:ids)
        {
            const auto *a=surface::find(id);
            const bool dirty=a && a->pixels.size()==words;
            signature=(signature^id)*1099511628211ull;
            signature=(signature^(dirty?a->generation:0ull))*1099511628211ull;
            if(dirty)
            {
                any=true;
                ++stained;
            }
        }
        if(!any)return vk::buffer::descriptor(data->fallback);

        auto &entry=data->batches[mesh];
        if(!entry.buffers.empty() && entry.generations.at(frame)==signature)
            return vk::buffer::descriptor(entry.buffers.at(frame));

        // Slot table occupies one word per instance, while the expensive
        // 256x256 atlas is stored ONLY for stained actors.
        if(ids.size()>std::numeric_limits<std::uint32_t>::max() ||
           stained>(std::numeric_limits<std::uint32_t>::max()-1-ids.size())/words)
            throw std::overflow_error("Blood atlas address space exceeded");
        std::vector<std::uint32_t> packed(
            1+ids.size()+stained*words,0u);
        packed[0]=std::uint32_t(ids.size());
        std::uint32_t slot=0;
        for(std::size_t i=0;i<ids.size();++i)
        {
            const auto *a=surface::find(ids[i]);
            if(a && a->pixels.size()==words)
            {
                packed[1+i]=slot++;
                std::copy(a->pixels.begin(),a->pixels.end(),
                          packed.begin()+1+ids.size()+
                          std::size_t(slot-1)*words);
            }
            else packed[1+i]=~std::uint32_t{0};
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
