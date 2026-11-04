#include "renderer.hpp"
#include "../core/system.hpp"
#include "../math/geometry.hpp"
#include "../simulation/solver.hpp"
#include "../surfaces/decals.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace bld::render
{
    namespace
    {
        state *data = nullptr;
        constexpr std::uint32_t tile_size = 16;
        constexpr std::uint32_t decal_cap = 48;
        constexpr std::uint32_t drop_cap = 32;

        struct gpu_decal
        {
            mat4 inverse_world;
            vec4 normal_kind;
            vec4 properties;
        };
        static_assert(sizeof(gpu_decal) == 96);

        struct gpu_drop
        {
            vec4 endpoints;
            vec4 properties;
        };
        static_assert(sizeof(gpu_drop) == 32);

        struct parameters
        {
            mat4 inverse_view_projection;
            std::uint32_t width, height, decals, flags;
        };
        static_assert(sizeof(parameters) == 80);

        struct bins
        {
            std::vector<std::uint32_t> counts, indices;
            std::uint32_t columns = 0, rows = 0, capacity = 0;

            bins(std::uint32_t width, std::uint32_t height, std::uint32_t per_tile)
            {
                columns = (width + tile_size - 1) / tile_size;
                rows = (height + tile_size - 1) / tile_size;
                capacity = per_tile;
                counts.assign(std::size_t(columns) * rows, 0);
            }

            void add(std::uint32_t id, float min_x, float min_y,
                     float max_x, float max_y)
            {
                if (max_x < 0 || max_y < 0 ||
                    min_x >= float(columns * tile_size) ||
                    min_y >= float(rows * tile_size)) return;

                const auto x0 = std::uint32_t(std::clamp(
                    int(std::floor(min_x / tile_size)), 0, int(columns)-1));
                const auto y0 = std::uint32_t(std::clamp(
                    int(std::floor(min_y / tile_size)), 0, int(rows)-1));
                const auto x1 = std::uint32_t(std::clamp(
                    int(std::floor(max_x / tile_size)), 0, int(columns)-1));
                const auto y1 = std::uint32_t(std::clamp(
                    int(std::floor(max_y / tile_size)), 0, int(rows)-1));

                if (indices.empty())
                    indices.assign(std::size_t(columns) * rows * capacity, 0);
                for (auto y = y0; y <= y1; ++y)
                    for (auto x = x0; x <= x1; ++x)
                    {
                        const std::size_t tile = std::size_t(y) * columns + x;
                        auto &count = counts[tile];
                        if (count >= capacity) continue;
                        indices[tile * capacity + count++] = id;
                    }
            }
        };

        vec3 screen(const mat4 &vp, vec3 world, std::uint32_t width,
                    std::uint32_t height, bool &visible)
        {
            const auto c = vp * vec4{world, 1};
            visible = c.w > .000001f;
            if (!visible) return {};
            return {(c.x / c.w * .5f + .5f) * width,
                    (c.y / c.w * .5f + .5f) * height, c.z / c.w};
        }

        void bin_decal(bins &index, std::uint32_t id,
                       const surfaces::projector &mark,
                       const input &input)
        {
            const auto matrix = math::inverse_affine(mark.world_to_local);
            float x0 = float(input.width), y0 = float(input.height);
            float x1 = -1.f, y1 = -1.f;
            bool behind = false;
            for (float x : {-.5f, .5f})
                for (float y : {-.5f, .5f})
                    for (float z : {-.5f, .5f})
                    {
                        const auto c = matrix * vec4{x, y, z, 1};
                        bool front = false;
                        const auto p = screen(input.view_projection,
                            {c.x, c.y, c.z}, input.width, input.height, front);
                        if (!front)
                        {
                            behind = true;
                            continue;
                        }
                        x0 = std::min(x0, p.x);
                        y0 = std::min(y0, p.y);
                        x1 = std::max(x1, p.x);
                        y1 = std::max(y1, p.y);
                    }
            if (behind)
                index.add(id, 0, 0, float(input.width), float(input.height));
            else
                index.add(id, x0, y0, x1, y1);
        }

        void buffer(frame &frame, std::uint32_t width, std::uint32_t height,
                    std::uint32_t max_decals, std::uint32_t max_drops)
        {
            const auto memory = vk::MemoryPropertyFlagBits::eHostVisible |
                                vk::MemoryPropertyFlagBits::eHostCoherent;
            const auto usage = vk::BufferUsageFlagBits::eStorageBuffer;
            const std::size_t tiles = std::size_t((width + 15) / 16) *
                                      std::size_t((height + 15) / 16);
            frame.decals = vk::buffer::create(
                std::max<std::size_t>(1, max_decals) * sizeof(gpu_decal),
                usage, memory);
            frame.decal_counts = vk::buffer::create(
                std::max<std::size_t>(1, tiles) * sizeof(std::uint32_t),
                usage, memory);
            frame.decal_indices = vk::buffer::create(
                std::max<std::size_t>(1, tiles * decal_cap) *
                    sizeof(std::uint32_t), usage, memory);
            frame.drops = vk::buffer::create(
                std::max<std::size_t>(1, max_drops) * sizeof(gpu_drop),
                usage, memory);
            frame.drop_counts = vk::buffer::create(
                std::max<std::size_t>(1, tiles) * sizeof(std::uint32_t),
                usage, memory);
            frame.drop_indices = vk::buffer::create(
                std::max<std::size_t>(1, tiles * drop_cap) *
                    sizeof(std::uint32_t), usage, memory);
        }

        void write_descriptors(frame &frame, const input &target)
        {
            std::array<vk::DescriptorImageInfo, 3> images{};
            images[0] = {target.sampler, target.depth, target.depth_layout};
            images[1] = {target.sampler, target.normals, target.normal_layout};
            images[2] = {{}, target.output, vk::ImageLayout::eGeneral};
            const std::array<vk::Buffer, 6> handles{{
                frame.decals.handle, frame.decal_counts.handle,
                frame.decal_indices.handle, frame.drops.handle,
                frame.drop_counts.handle, frame.drop_indices.handle
            }};
            std::array<vk::DescriptorBufferInfo, 6> buffers{};
            std::array<vk::WriteDescriptorSet, 9> writes{};
            for (std::uint32_t i = 0; i < 9; ++i)
            {
                auto &write = writes[i];
                write.dstSet = frame.set;
                write.dstBinding = i;
                write.descriptorCount = 1;
                if (i < 3)
                {
                    write.descriptorType = i == 2 ?
                        vk::DescriptorType::eStorageImage :
                        vk::DescriptorType::eCombinedImageSampler;
                    write.pImageInfo = &images[i];
                }
                else
                {
                    buffers[i-3].buffer = handles[i-3];
                    buffers[i-3].range = VK_WHOLE_SIZE;
                    write.descriptorType = vk::DescriptorType::eStorageBuffer;
                    write.pBufferInfo = &buffers[i-3];
                }
            }
            vk::device::get()->handle.updateDescriptorSets(writes, {});
        }
    }

    state *get() { return data; }

    void init(const std::vector<std::uint32_t> &spirv,
              std::uint32_t count, std::uint32_t width, std::uint32_t height)
    {
        if (data) throw std::logic_error("Blood renderer already initialized");
        if (!count || !width || !height || width > 8192 || height > 8192 ||
            spirv.empty() || !vk::device::get()->handle)
            throw std::invalid_argument("Invalid blood renderer initialization");

        data = new state{};
        data->max_width = width;
        data->max_height = height;
        if (auto *system = bld::system::get())
        {
            data->max_decals = system->options.max_decals;
            data->max_drops = system->options.max_particles;
        }
        const auto device = vk::device::get()->handle;
        try
        {
            std::array<vk::DescriptorSetLayoutBinding, 9> bindings{};
            for (std::uint32_t i = 0; i < bindings.size(); ++i)
            {
                bindings[i].binding = i;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = vk::ShaderStageFlagBits::eCompute;
                bindings[i].descriptorType = i < 2 ?
                    vk::DescriptorType::eCombinedImageSampler :
                    i == 2 ? vk::DescriptorType::eStorageImage :
                             vk::DescriptorType::eStorageBuffer;
            }
            vk::DescriptorSetLayoutCreateInfo layout{};
            layout.bindingCount = bindings.size();
            layout.pBindings = bindings.data();
            data->descriptor_layout = vk::descriptor_layout::create(layout);

            vk::PushConstantRange push{};
            push.stageFlags = vk::ShaderStageFlagBits::eCompute;
            push.size = sizeof(parameters);
            vk::PipelineLayoutCreateInfo pipeline_layout{};
            pipeline_layout.setLayoutCount = 1;
            pipeline_layout.pSetLayouts = &data->descriptor_layout;
            pipeline_layout.pushConstantRangeCount = 1;
            pipeline_layout.pPushConstantRanges = &push;
            data->layout = vk::pipeline_layout::create(pipeline_layout);

            auto shader = vk::shader::create(spirv);
            try
            {
                vk::ComputePipelineCreateInfo pipeline{};
                pipeline.stage.stage = vk::ShaderStageFlagBits::eCompute;
                pipeline.stage.module = shader;
                pipeline.stage.pName = "main";
                pipeline.layout = data->layout;
                data->pipeline = vk::pipeline::compute(pipeline);
            }
            catch (...)
            {
                vk::shader::destroy(shader);
                throw;
            }
            vk::shader::destroy(shader);

            std::array<vk::DescriptorPoolSize, 3> sizes{{
                {vk::DescriptorType::eCombinedImageSampler, count * 2},
                {vk::DescriptorType::eStorageImage, count},
                {vk::DescriptorType::eStorageBuffer, count * 6}
            }};
            vk::DescriptorPoolCreateInfo pool{};
            pool.maxSets = count;
            pool.poolSizeCount = sizes.size();
            pool.pPoolSizes = sizes.data();
            data->descriptor_pool = device.createDescriptorPool(pool);

            std::vector<vk::DescriptorSetLayout> layouts(
                count, data->descriptor_layout);
            vk::DescriptorSetAllocateInfo allocate{};
            allocate.descriptorPool = data->descriptor_pool;
            allocate.descriptorSetCount = count;
            allocate.pSetLayouts = layouts.data();
            const auto sets = device.allocateDescriptorSets(allocate);
            data->frames.resize(count);
            for (std::uint32_t i = 0; i < count; ++i)
            {
                data->frames[i].set = sets[i];
                buffer(data->frames[i], width, height,
                       data->max_decals, data->max_drops);
            }
        }
        catch (...)
        {
            shutdown();
            throw;
        }
    }

    void record(vk::CommandBuffer command, std::uint32_t index,
                const input &target)
    {
        if (!data || index >= data->frames.size() || !command ||
            !target.depth || !target.normals || !target.output ||
            !target.sampler || !target.width || !target.height ||
            target.width > data->max_width || target.height > data->max_height)
            throw std::invalid_argument("Invalid blood render targets");

        auto &frame = data->frames[index];
        const auto decals = surfaces::resolve();
        static const std::vector<simulation::drop> empty_particles;
        const auto *sim = simulation::get();
        const auto &particles = sim ? sim->particles : empty_particles;

        bins decal_tiles(target.width, target.height, decal_cap);
        bins drop_tiles(target.width, target.height, drop_cap);
        std::vector<gpu_decal> gpu_decals;
        std::vector<gpu_drop> gpu_drops;
        gpu_decals.reserve(std::min(decals.size(), std::size_t(data->max_decals)));
        gpu_drops.reserve(std::min(particles.size(), std::size_t(data->max_drops)));

        for (const auto &source : decals)
        {
            if (gpu_decals.size() >= data->max_decals) break;
            const auto id = std::uint32_t(gpu_decals.size());
            gpu_decal item{};
            item.inverse_world = source.world_to_local;
            item.normal_kind = {source.normal.x, source.normal.y,
                source.normal.z, float(source.variant + (source.pool ? 8u : 0u))};
            item.properties = {source.wetness,
                               float(id) * 2.126f + source.variant * 1.17f, 0, 0};
            gpu_decals.push_back(item);
            bin_decal(decal_tiles, id, source, target);
        }

        for (const auto &d : particles)
        {
            if (gpu_drops.size() >= data->max_drops) break;
            bool a_ok = false, b_ok = false;
            const auto a = screen(target.view_projection, d.previous,
                                  target.width, target.height, a_ok);
            const auto b = screen(target.view_projection, d.position,
                                  target.width, target.height, b_ok);
            if (!a_ok || !b_ok || b.z < 0.f || b.z > 1.f) continue;
            const float width = std::abs(target.view_projection(1, 1));
            const auto clip = target.view_projection * vec4{d.position, 1};
            const float radius = std::clamp(
                d.radius * width * .5f * target.height /
                    std::max(.01f, clip.w), 1.5f, 20.f);
            gpu_drop item{};
            item.endpoints = {a.x, a.y, b.x, b.y};
            item.properties = {radius, b.z,
                               std::max(0.f, 1.f - d.age / d.lifetime),
                               float(d.seed % 1024u)};
            const auto id = std::uint32_t(gpu_drops.size());
            gpu_drops.push_back(item);
            const float padding = radius + 2.f;
            drop_tiles.add(id, std::min(a.x, b.x) - padding,
                           std::min(a.y, b.y) - padding,
                           std::max(a.x, b.x) + padding,
                           std::max(a.y, b.y) + padding);
        }

        if (!gpu_decals.empty())
            vk::buffer::write(frame.decals, gpu_decals.data(),
                              gpu_decals.size() * sizeof(gpu_decal));
        if (!gpu_drops.empty())
            vk::buffer::write(frame.drops, gpu_drops.data(),
                              gpu_drops.size() * sizeof(gpu_drop));
        vk::buffer::write(frame.decal_counts, decal_tiles.counts.data(),
                          decal_tiles.counts.size() * sizeof(std::uint32_t));
        if (!decal_tiles.indices.empty())
            vk::buffer::write(frame.decal_indices, decal_tiles.indices.data(),
                              decal_tiles.indices.size() * sizeof(std::uint32_t));
        vk::buffer::write(frame.drop_counts, drop_tiles.counts.data(),
                          drop_tiles.counts.size() * sizeof(std::uint32_t));
        if (!drop_tiles.indices.empty())
            vk::buffer::write(frame.drop_indices, drop_tiles.indices.data(),
                              drop_tiles.indices.size() * sizeof(std::uint32_t));

        write_descriptors(frame, target);
        parameters push{};
        push.inverse_view_projection = target.inverse_view_projection;
        push.width = target.width;
        push.height = target.height;
        push.decals = std::uint32_t(gpu_decals.size());
        push.flags = (target.reverse_depth ? 1u : 0u) |
                     (target.encoded_normals ? 2u : 0u);
        vk::record::pipeline(command, vk::PipelineBindPoint::eCompute, data->pipeline);
        command.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   data->layout, 0, {frame.set}, {});
        command.pushConstants(data->layout, vk::ShaderStageFlagBits::eCompute,
                              0, sizeof(push), &push);
        vk::record::dispatch(command,
                             (target.width + 7u) / 8u,
                             (target.height + 7u) / 8u, 1);
    }

    void shutdown()
    {
        if (!data) return;
        for (auto &frame : data->frames)
        {
            if (frame.decals.handle) vk::buffer::destroy(frame.decals);
            if (frame.decal_counts.handle) vk::buffer::destroy(frame.decal_counts);
            if (frame.decal_indices.handle) vk::buffer::destroy(frame.decal_indices);
            if (frame.drops.handle) vk::buffer::destroy(frame.drops);
            if (frame.drop_counts.handle) vk::buffer::destroy(frame.drop_counts);
            if (frame.drop_indices.handle) vk::buffer::destroy(frame.drop_indices);
        }
        if (data->pipeline) vk::pipeline::destroy(data->pipeline);
        if (data->layout) vk::pipeline_layout::destroy(data->layout);
        if (data->descriptor_pool)
            vk::device::get()->handle.destroyDescriptorPool(data->descriptor_pool);
        if (data->descriptor_layout)
            vk::descriptor_layout::destroy(data->descriptor_layout);
        delete data;
        data = nullptr;
    }
}
