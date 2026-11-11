#pragma once
#include <vk/vk.hpp>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gpu
{
    // Independent, nonblocking GPU timestamp intervals, one query pool per
    // swapchain image. Call begin_frame only once the image's fence is complete.
    // No query-result wait or device idle is performed on the render path.
    class timestamps
    {
    public:
        timestamps()=default;
        ~timestamps(){reset();}
        timestamps(const timestamps&)=delete;
        timestamps& operator=(const timestamps&)=delete;

        void create(std::uint32_t images,std::uint32_t intervals,bool supported)
        {
            reset();
            if(!supported || !images || !intervals)return;
            intervals_=intervals;
            period_=vk::physical_device::get()->handle
                .getProperties().limits.timestampPeriod;
            auto device=vk::device::get()->handle;
            try
            {
                for(std::uint32_t i=0;i<images;++i)
                {
                    vk::QueryPoolCreateInfo info{};
                    info.queryType=vk::QueryType::eTimestamp;
                    info.queryCount=intervals_*2;
                    pools_.push_back(device.createQueryPool(info));
                }
                submitted_.assign(images,false);
            }
            catch(...)
            {
                reset();
                throw;
            }
        }

        void reset() noexcept
        {
            if(!pools_.empty())
            {
                try
                {
                    auto device=vk::device::get()->handle;
                    if(device)
                        for(auto pool:pools_)device.destroyQueryPool(pool);
                }
                catch(...) {}
            }
            pools_.clear();
            submitted_.clear();
            intervals_=0;
            period_=0.f;
        }

        bool supported() const noexcept {return !pools_.empty();}

        // Empty means unavailable or not yet finished. The caller should
        // sample only after any previous use of this image has completed.
        std::vector<double> read(std::uint32_t image) const
        {
            if(image>=pools_.size() || !submitted_[image])return {};
            std::vector<std::uint64_t> ticks(intervals_*2);
            auto device=vk::device::get()->handle;
            const auto result=device.getQueryPoolResults(
                pools_[image],0,static_cast<std::uint32_t>(ticks.size()),
                ticks.size()*sizeof(ticks[0]),ticks.data(),
                sizeof(ticks[0]),vk::QueryResultFlagBits::e64);
            if(result==vk::Result::eNotReady)return {};
            if(result!=vk::Result::eSuccess)
                throw std::runtime_error("GPU timestamp query failed");
            std::vector<double> durations(intervals_);
            for(std::size_t i=0;i<durations.size();++i)
                durations[i]=double(ticks[2*i+1]-ticks[2*i])*
                             double(period_)/1000000.0;
            return durations;
        }

        void begin_frame(vk::CommandBuffer command,std::uint32_t image)
        {
            if(image>=pools_.size())return;
            command.resetQueryPool(pools_[image],0,intervals_*2);
            submitted_[image]=false;
        }

        void start(vk::CommandBuffer command,std::uint32_t image,
                   std::uint32_t interval) const
        {
            if(image<pools_.size() && interval<intervals_)
                // A top-of-pipe start can run ahead of earlier raster
                // commands. The optical interval lives inside presentation:
                // bottom-of-pipe establishes a finished prior-work boundary.
                command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                                       pools_[image],interval*2);
        }

        void end(vk::CommandBuffer command,std::uint32_t image,
                 std::uint32_t interval) const
        {
            if(image<pools_.size() && interval<intervals_)
                command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                                       pools_[image],interval*2+1);
        }

        void submitted(std::uint32_t image)
        {
            if(image<submitted_.size())submitted_[image]=true;
        }

    private:
        std::vector<vk::QueryPool> pools_;
        std::vector<bool> submitted_;
        std::uint32_t intervals_=0;
        float period_=0.f;
    };
}
