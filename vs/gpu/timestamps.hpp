#pragma once
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vk/vk.hpp>
#include <vs/gpu/resource.hpp>

namespace vs::gpu
{
struct timestamp_sample
{
    std::vector<double> passes;

    double frame = 0.0;
};

struct timestamp_options
{
    bool isolate_passes = false;
    bool measure_frame = false;
};

// Independent, nonblocking GPU timestamp intervals, one query pool per
// swapchain image. Call begin_frame only once the image's fence is complete.
// No query-result wait or device idle is performed on the render path.
class timestamps
{
public:
    timestamps() = default;
    ~timestamps() { reset(); }
    timestamps(const timestamps &) = delete;
    timestamps &operator=(const timestamps &) = delete;

    void create(std::uint32_t images, std::uint32_t intervals, bool supported,
                timestamp_options options = {})
    {
        reset();

        if (!supported || !images || !intervals) return;
        intervals_ = intervals;
        measure_frame_ = options.measure_frame;
        period_ = vk::physical_device::get()->handle.getProperties().limits.timestampPeriod;

        auto device = vk::device::get()->handle;

        try
        {
            for (std::uint32_t i = 0; i < images; ++i)
            {
                vk::QueryPoolCreateInfo info{};
                info.queryType = vk::QueryType::eTimestamp;
                info.queryCount = query_count();
                pools_.push_back(device.createQueryPool(info));

                if (options.isolate_passes)
                    boundaries_.push_back(create_buffer(
                        vk::DeviceSize(query_count()) * sizeof(std::uint32_t),
                        vk::BufferUsageFlagBits::eTransferDst,
                        vk::MemoryPropertyFlagBits::eDeviceLocal));
            }

            submitted_.assign(images, false);
        }

        catch (...)
        {
            reset();
            throw;
        }
    }

    void reset() noexcept
    {
        if (!pools_.empty()) {
            try
            {
                auto device = vk::device::get()->handle;

                if (device)
                    for (auto pool : pools_)
                        device.destroyQueryPool(pool);
            }

            catch (...)
            {
            }
        }

        pools_.clear();
        boundaries_.clear();
        submitted_.clear();
        intervals_ = 0;
        measure_frame_ = false;
        period_ = 0.f;
    }

    bool supported() const noexcept { return !pools_.empty(); }
    bool isolates_passes() const noexcept { return !boundaries_.empty(); }

    // Empty means unavailable or not yet finished. The caller should
    // sample only after any previous use of this image has completed.
    std::vector<double> read(std::uint32_t image) const
    {
        return read_sample(image).passes;
    }

    timestamp_sample read_sample(std::uint32_t image) const
    {
        if (image >= pools_.size() || !submitted_[image]) return {};
        std::vector<std::uint64_t> ticks(query_count());

        auto device = vk::device::get()->handle;
        const auto result = device.getQueryPoolResults(
            pools_[image], 0, static_cast<std::uint32_t>(ticks.size()),
            ticks.size() * sizeof(ticks[0]), ticks.data(),
            sizeof(ticks[0]), vk::QueryResultFlagBits::e64);
        if (result == vk::Result::eNotReady) return {};
        if (result != vk::Result::eSuccess)
            throw std::runtime_error("GPU timestamp query failed");

        const auto duration = [&](std::size_t i)
        {
            return double(ticks[2 * i + 1] - ticks[2 * i]) * double(period_) / 1000000.0;
        };

        timestamp_sample sample{};
        sample.passes.reserve(intervals_);

        for (std::size_t i = 0; i < intervals_; ++i)
            sample.passes.push_back(duration(i));

        if (measure_frame_) sample.frame = duration(intervals_);

        return sample;
    }

    void begin_frame(vk::CommandBuffer command, std::uint32_t image)
    {
        if (image >= pools_.size()) return;
        command.resetQueryPool(pools_[image], 0, query_count());
        submitted_[image] = false;

        if (measure_frame_)
            command.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe,
                                   pools_[image], intervals_ * 2);
    }

    void end_frame(vk::CommandBuffer command, std::uint32_t image) const
    {
        if (!measure_frame_ || image >= pools_.size()) return;
        command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                               pools_[image], intervals_ * 2 + 1);
    }

    void start(vk::CommandBuffer command, std::uint32_t image,
               std::uint32_t interval) const
    {
        if (image < pools_.size() && interval < intervals_) {
            boundary(command, image, interval * 2);
            command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                                   pools_[image], interval * 2);
        }
    }

    void end(vk::CommandBuffer command, std::uint32_t image,
             std::uint32_t interval) const
    {
        if (image < pools_.size() && interval < intervals_) {
            command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
                                   pools_[image], interval * 2 + 1);
            boundary(command, image, interval * 2 + 1);
        }
    }

    void submitted(std::uint32_t image)
    {
        if (image < submitted_.size()) submitted_[image] = true;
    }

private:
    std::uint32_t query_count() const noexcept
    {
        return (intervals_ + std::uint32_t(measure_frame_)) * 2;
    }

    void boundary(vk::CommandBuffer command, std::uint32_t image,
                  std::uint32_t query) const
    {
        if (boundaries_.empty()) return;
        // MoltenVK stage counter queries are deferred to their Metal
        // encoder boundary. This tiny transfer ends the preceding
        // compute/render encoder during an isolated pass capture.
        const std::uint32_t zero = 0;
        command.updateBuffer(boundaries_[image].handle,
                             vk::DeviceSize(query) * sizeof(zero), sizeof(zero), &zero);
    }

    std::vector<vk::QueryPool> pools_;
    std::vector<owned_buffer> boundaries_;
    std::vector<bool> submitted_;
    std::uint32_t intervals_ = 0;

    bool measure_frame_ = false;
    float period_ = 0.f;
};
} // namespace vs::gpu
