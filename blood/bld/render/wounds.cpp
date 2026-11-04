#include "wounds.hpp"
#include "../surfaces/stains.hpp"
#include "../core/system.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
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
        const auto *surfaces=surface::get();
        return (a && (!a->marks.empty() || !a->baked_pixels.empty())) ||
            (surfaces && surfaces->rigid_targets.count(id)!=0);
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

    vk::DescriptorBufferInfo descriptor_projected_batch(
        std::uint32_t mesh,const std::vector<instance_id> &ids,
        std::uint32_t frame)
    {
        if(!data || frame>=data->frames || !mesh)
            throw std::invalid_argument("Invalid projected blood batch");
        const auto *surfaces=surface::get();
        if(!surfaces || ids.empty() || surfaces->stains.empty())
            return vk::buffer::descriptor(data->fallback);

        std::unordered_map<instance_id,std::vector<const surface::rigid_stain*>> grouped;
        grouped.reserve(surfaces->rigid_targets.size());
        for(const auto &stain:surfaces->stains)
            grouped[stain.target].push_back(&stain);
        constexpr std::size_t resolution=48;
        constexpr std::size_t stride=12+resolution*resolution;
        constexpr std::size_t max_marks_per_instance=16;
        std::size_t total=0;
        for(const auto id:ids)
        {
            const auto iter=grouped.find(id);
            if(iter!=grouped.end())
                total+=std::min(max_marks_per_instance,iter->second.size());
        }
        if(!total)return vk::buffer::descriptor(data->fallback);
        if(ids.size()>std::numeric_limits<std::uint32_t>::max() ||
           ids.size()>(std::numeric_limits<std::uint32_t>::max()-1)/2 ||
           total>(std::numeric_limits<std::uint32_t>::max()-1-2*ids.size())/stride)
            throw std::overflow_error("Projected blood batch too large");
        std::uint64_t signature=14695981039346656037ull;
        signature=(signature^surfaces->generation)*1099511628211ull;
        for(const auto id:ids)
            signature=(signature^id)*1099511628211ull;
        auto &entry=data->projected_batches[mesh];
        if(!entry.buffers.empty() && entry.generations.at(frame)==signature)
            return vk::buffer::descriptor(entry.buffers.at(frame));

        // Per-instance compact offsets/counts, followed by a 12-word
        // local surface frame and a 48x48 evolving film field per patch.
        std::vector<std::uint32_t> packed(1+2*ids.size()+stride*total,0);
        packed[0]=std::uint32_t(ids.size());
        const auto bits=[](float value)
        {
            std::uint32_t result=0;
            std::memcpy(&result,&value,sizeof(result));
            return result;
        };
        std::size_t slot=0;
        const auto base=1+2*ids.size();
        for(std::size_t i=0;i<ids.size();++i)
        {
            const auto iter=grouped.find(ids[i]);
            packed[1+i]=std::uint32_t(slot);
            if(iter==grouped.end())continue;
            const auto &marks=iter->second;
            const auto begin=marks.size()>max_marks_per_instance?
                marks.size()-max_marks_per_instance:0;
            packed[1+ids.size()+i]=std::uint32_t(marks.size()-begin);
            for(std::size_t j=begin;j<marks.size();++j)
            {
                const auto &mark=*marks[j];
                const auto at=base+stride*slot++;
                packed[at+0]=bits(mark.world(0,3));
                packed[at+1]=bits(mark.world(1,3));
                packed[at+2]=bits(mark.world(2,3));
                packed[at+3]=bits(mark.film.metres_per_uv.x);
                packed[at+4]=bits(mark.world(0,0));
                packed[at+5]=bits(mark.world(1,0));
                packed[at+6]=bits(mark.world(2,0));
                packed[at+7]=bits(mark.film.metres_per_uv.y);
                packed[at+8]=bits(mark.world(0,2));
                packed[at+9]=bits(mark.world(1,2));
                packed[at+10]=bits(mark.world(2,2));
                packed[at+11]=bits(mark.wetness);
                if(mark.pixels.size()==resolution*resolution)
                    std::copy(mark.pixels.begin(),mark.pixels.end(),
                              packed.begin()+at+12);
            }
        }
        return write(entry,frame,packed,signature);
    }

    void clear_batches()
    {
        if(!data)return;
        for(auto &[id,item]:data->batches)destroy(item);
        data->batches.clear();
        for(auto &[id,item]:data->projected_batches)destroy(item);
        data->projected_batches.clear();
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
