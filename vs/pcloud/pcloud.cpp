#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vs/bvh/gpu_regions.hpp>
#include <vs/bvh/stackless.hpp>
#include <vs/gpu/compute.hpp>
#include <vs/gpu/descriptors.hpp>
#include <vs/gpu/graph.hpp>
#include <vs/gpu/light.hpp>
#include <vs/gpu/resource.hpp>
#include <vs/jobs/pool.hpp>
#include <vs/lighting/transmission.hpp>
#include <vs/pcloud/pcloud.hpp>
#include <vs/pcloud/tone_workgroup.h>
#include <vs/render/pipeline.hpp>
#include <vs/render/postprocess.hpp>
#include <vs/render/renderer.hpp>
#include <vs/spatial/bricks.hpp>
namespace vs::pcloud
{
namespace
{
struct image_target
{
    vk::Image image{};
    vk::ImageView view{};
};

struct frame
{
    image_target hdr;
    image_target ldr;
    ::vs::gpu::owned_buffer points;
    ::vs::gpu::owned_buffer tile_lights;
    ::vs::gpu::owned_buffer dynamic_vertices, dynamic_nodes, changed_points;
    std::vector<std::uint32_t> tile_cpu;
    std::uint32_t dynamic_count = 0;

    bool dynamic_has_spheres = false;

    std::vector<std::uint8_t> dynamic_packet;
    std::vector<std::uint32_t> affected_cpu;
    std::uint32_t gi_interval = 0;
    std::uint32_t gi_cursor = 0;

    bool gi_complete = false, cloud_submitted = false;

    collision_box dynamic_bounds{};

    bool has_dynamic_bounds = false, cloud_partial = false;

    std::uint32_t changed_count = 0;
    ::vs::gpu::owned_image grid;
    vk::ImageView grid_view{};

    bool dirty = true;
    bool partial_grid = false;
    bool grid_initialized = false;

    std::array<std::uint32_t, 3> grid_first{}, grid_end{};
    vk::DescriptorSet cloud{};
    vk::DescriptorSet gi{};
    vk::DescriptorSet lighting{};
    vk::DescriptorSet tone{};
    vk::DescriptorSet presentation{};
};

struct reflection_frame
{
    image_target hdr, ldr;
    ::vs::gpu::owned_buffer tile_lights;
    vk::DescriptorSet lighting{}, tone{};
};

struct state
{
    std::vector<frame> frames;
    std::vector<reflection_frame> reflections;
    std::unique_ptr<::vs::render::postprocess_targets> reflection_targets;
    std::vector<::vs::gpu::graph> frame_graphs;
    vk::Sampler sampler{};
    ::vs::gpu::owned_buffer lights, light_shapes;
    ::vs::gpu::owned_buffer triangles;
    ::vs::gpu::owned_buffer nodes;
    ::vs::gpu::owned_buffer shadow_light_ranges;
    ::vs::bvh::gpu_regions region_shadows;

    std::optional<::vs::scene::collision_box> probe_coverage;

    bool coverage_dirty = false;
    ::vs::gpu::owned_buffer transmission;
    ::vs::gpu::owned_buffer glass_vertices, glass_nodes, glass_filters;
    std::uint32_t glass_node_count = 0;
    ::vs::gpu::owned_buffer neighborhood_offsets;
    ::vs::gpu::owned_buffer neighborhood_indices;
    std::shared_ptr<const ::vs::scene::data> snapshot;
    std::shared_ptr<const ::vs::scene::data> incoming;
    std::vector<light> live_lights;
    std::vector<vec3> sample_positions;
    std::vector<::vs::bvh::stackless_node> dynamic_topology, prepared_nodes;
    std::vector<std::uint32_t> dynamic_order;

    bool dynamic_topology_pending = false;

    std::uint64_t pending_dynamic_signature = 0;
    std::size_t pending_dynamic_primitives = 0;
    std::uint64_t dynamic_group_signature = 0;
    std::vector<std::array<float, 4>> prepared_vertices;

    float dynamic_reference_cost = 0.f;

    std::uint32_t dynamic_refit_age = 0;
    collision_box prepared_bounds{};
    std::array<std::uint32_t, 2> dynamic_shadow_mask{~0u, ~0u};
    std::uint32_t prepared_primitives = 0;

    bool prepared_has_spheres = false;

    bool dynamic_prepared = false;

    configuration shader_paths{};
    presentation_target output{};
    std::uint32_t light_count = 0, point_count = 0, vertex_count = 0, node_count = 0;
    std::uint32_t lighting_pixel_rate = 1;
    std::uint32_t visual_node_count = 0;
    std::uint32_t tile_columns = 0, tile_rows = 0;
    vk::util::math::vec3 grid_origin{};

    float grid_spacing = 0.2f;

    vk::Extent3D grid_extent{1, 1, 1};
    std::unique_ptr<::vs::gpu::descriptors> cloud_sets;
    std::unique_ptr<::vs::gpu::descriptors> gi_sets;
    std::unique_ptr<::vs::gpu::descriptors> lighting_sets;
    std::unique_ptr<::vs::gpu::descriptors> tone_sets;
    std::unique_ptr<::vs::gpu::descriptors> present_sets;
    std::unique_ptr<::vs::render::postprocess_targets> targets;
    std::unique_ptr<::vs::render::presentation_targets> presentation;
    vk::PipelineLayout present_layout{};
    std::unique_ptr<::vs::gpu::compute> cloud_program;
    std::unique_ptr<::vs::gpu::compute> gi_program;
    std::unique_ptr<::vs::gpu::compute> lighting_program, opaque_lighting_program;
    std::unique_ptr<::vs::gpu::compute> tone_program;
    std::unique_ptr<::vs::render::pipeline> present_program;
};

state &get()
{
    static state data;

    return data;
}

::vs::jobs::pool dynamic_preparation;

bool dynamic_preparation_started = false;

std::atomic<float> exposure_value{};
constexpr float min_grid_spacing = 0.2f;
constexpr float bounce_reach = 2.6f;
constexpr std::uint64_t max_grid_probes = 32768;
constexpr std::uint32_t tile_size = 16, tile_mask_words = 2;

using point_light = ::vs::gpu::point_light;

struct probe_point
{
    std::array<float, 4> position{};
    std::array<float, 4> normal{};
    std::array<float, 4> radiance{};
};

static_assert(sizeof(probe_point) == 48);

using shadow_node = ::vs::bvh::stackless_node;

struct cloud_push
{
    std::uint32_t points, vertices, lights, nodes, partial;
};

struct grid_push
{
    std::uint32_t points, vertices, nodes, lights = 0;
    std::array<float, 4> origin_spacing{};
    std::array<std::uint32_t, 4> region_first{};
    std::uint32_t glass_node_count = 0;
};

static_assert(sizeof(grid_push) == 52);

struct push
{
    float exposure;

    std::uint32_t indirect_enabled = 1u;
    // Exact conservative eligibility: a light with no overlapping
    // dynamic geometry never needs animated BVH ray traversal.
    std::array<std::uint32_t, 2> dynamic_shadow_mask{};
    std::array<float, 4> eye_near{};
    std::array<float, 4> forward_far{};
    std::array<float, 4> right_tan{};
    std::array<float, 4> up_tan{};
    // x: light count, y: reflected-view marker, w: adaptive pixel rate.
    std::array<float, 4> light_count{};
    std::array<float, 4> sky{};
    std::uint32_t brdf_enabled = 1u;
    std::uint32_t reconstruction_enabled = 1u;
    std::uint32_t direct_enabled = 1u;
    std::uint32_t dispatch_origin = 0u;
};

static_assert(sizeof(push) == 128);
static_assert(offsetof(push, dispatch_origin) == 124);

struct compute_area
{
    std::uint32_t columns, rows, origin;
};

compute_area reflection_area(vk::Extent2D extent, vk::Rect2D pixels, std::uint32_t tile = 8u)
{
    if (!pixels.extent.width || !pixels.extent.height)
        return {(extent.width + tile - 1u) / tile, (extent.height + tile - 1u) / tile, 0u};

    if (pixels.offset.x < 0 || pixels.offset.y < 0 ||
        std::uint64_t(pixels.offset.x) + pixels.extent.width > extent.width ||
        std::uint64_t(pixels.offset.y) + pixels.extent.height > extent.height)
        throw std::invalid_argument("Reflection lighting region outside target");

    const auto left = std::uint32_t(pixels.offset.x);
    const auto top = std::uint32_t(pixels.offset.y);
    const auto x = left / tile;
    const auto y = top / tile;

    if (x > 65535u || y > 65535u)
        return {(extent.width + tile - 1u) / tile, (extent.height + tile - 1u) / tile, 0u};

    return {(left + pixels.extent.width + tile - 1u) / tile - x, (top + pixels.extent.height + tile - 1u) / tile - y, (y << 16u) | x};
}

bool indirect_lighting_enabled()
{
    static const char *value = std::getenv("VS_INDIRECT_LIGHTING");
    const bool enabled = !value || *value != '0';

    if (!enabled) {
        static bool reported = false;

        if (!reported) {
            std::cout << "pc: indirect probe lighting disabled by diagnostic override\n";
            reported = true;
        }
    }

    return enabled;
}

bool direct_brdf_enabled()
{
    static const char *value = std::getenv("VS_DIRECT_BRDF");
    const bool enabled = !value || *value != '0';

    if (!enabled) {
        static bool reported = false;

        if (!reported) {
            std::cout << "pc: direct BRDF evaluation disabled by diagnostic override\n";
            reported = true;
        }
    }

    return enabled;
}

bool lighting_reconstruction_enabled()
{
    static const char *value = std::getenv("VS_LIGHTING_RECONSTRUCTION");
    const bool enabled = !value || *value != '0';

    if (!enabled) {
        static bool reported = false;

        if (!reported) {
            std::cout << "pc: adaptive lighting reconstruction disabled by diagnostic override\n";
            reported = true;
        }
    }

    return enabled;
}

bool direct_lighting_enabled()
{
    static const char *value = std::getenv("VS_DIRECT_LIGHTING");
    const bool enabled = !value || *value != '0';

    if (!enabled) {
        static bool reported = false;

        if (!reported) {
            std::cout << "pc: direct lighting disabled by diagnostic override\n";
            reported = true;
        }
    }

    return enabled;
}

std::vector<vk::DescriptorSetLayoutBinding> bindings(
    std::uint32_t sampled, std::uint32_t storage, vk::ShaderStageFlags stages,
    std::uint32_t buffers = 0)
{
    std::vector<vk::DescriptorSetLayoutBinding> result(sampled + storage + buffers);

    for (std::uint32_t i = 0; i < result.size(); ++i)
    {
        auto &binding = result[i];
        binding.binding = i;
        binding.descriptorCount = 1;
        binding.stageFlags = stages;
        binding.descriptorType = i < sampled
                                     ? vk::DescriptorType::eCombinedImageSampler
                                 : i < sampled + storage ? vk::DescriptorType::eStorageImage
                                                         : vk::DescriptorType::eStorageBuffer;
    }

    return result;
}

void create_grid(frame &current, vk::Extent3D extent)
{
    if (current.grid_view) {
        vk::image_view::destroy(current.grid_view);
        current.grid_view = vk::ImageView{};
    }

    current.grid.reset();
    vk::ImageCreateInfo grid_info{};
    grid_info.imageType = vk::ImageType::e3D;
    grid_info.format = vk::Format::eR16G16B16A16Sfloat;
    grid_info.extent = extent;
    grid_info.mipLevels = 1;
    grid_info.arrayLayers = 1;
    grid_info.samples = vk::SampleCountFlagBits::e1;
    grid_info.tiling = vk::ImageTiling::eOptimal;
    grid_info.usage = vk::ImageUsageFlagBits::eStorage |
                      vk::ImageUsageFlagBits::eSampled;
    current.grid = ::vs::gpu::create_image(grid_info, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vk::ImageViewCreateInfo view{};
    view.image = current.grid.handle;
    view.viewType = vk::ImageViewType::e3D;
    view.format = grid_info.format;
    view.subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    current.grid_view = vk::image_view::create(view);
    current.dirty = true;
    current.grid_initialized = false;
    current.gi_cursor = 0;
    current.gi_complete = false;
    current.cloud_submitted = false;
    current.partial_grid = false;
}

vk::PipelineLayout pipeline_layout(vk::DescriptorSetLayout set)
{
    vk::PipelineLayoutCreateInfo info{};
    info.setLayoutCount = 1;
    info.pSetLayouts = &set;

    return vk::device::get()->handle.createPipelineLayout(info);
}

void compute(vk::CommandBuffer command, const ::vs::gpu::compute &program,
             vk::DescriptorSet set, const camera_frame &frame,
             std::uint32_t pixel_rate = 1u,
             std::uint32_t local_size = 8u)
{
    const auto extent = get().output.extent;

    const auto &view = frame;

    const auto basis = ::vs::scene::axes(view.yaw, view.pitch);
    const float tan_half = std::tan(view.vertical_fov * 0.5f);
    const float aspect = float(extent.width) / float(extent.height);

    const auto &position = frame.position;
    const auto &grid = get();
    const push controls{
        exposure_value.load(std::memory_order_relaxed),
        indirect_lighting_enabled() ? 1u : 0u,
        grid.dynamic_shadow_mask,
        {position.x, position.y, position.z, 0.1f},
        {basis.forward.x, basis.forward.y, basis.forward.z, 1000.0f},
        {basis.right.x, basis.right.y, basis.right.z, tan_half * aspect},
        {basis.up.x, basis.up.y, basis.up.z, tan_half},
        {float(get().light_count), 0.f,
         0.f,
         float(pixel_rate)},
        {grid.grid_origin.x, grid.grid_origin.y, grid.grid_origin.z,
         grid.grid_spacing},
        direct_brdf_enabled() ? 1u : 0u,
        lighting_reconstruction_enabled() ? 1u : 0u,
        direct_lighting_enabled() ? 1u : 0u
    };

    const auto width = extent.width;
    const auto height = extent.height;
    const auto tile = local_size * pixel_rate;
    program.dispatch(command, (width + tile - 1u) / tile,
                     (height + tile - 1u) / tile, 1,
                     {set}, &controls, sizeof(controls));
}
} // namespace

gi_progress progress(std::uint32_t image)
{
    const auto &data = get();
    const auto &frame = data.frames.at(image);
    const std::array<std::uint32_t, 3> extent{
        frame.partial_grid ? frame.grid_end[0] - frame.grid_first[0] : data.grid_extent.width,
        frame.partial_grid ? frame.grid_end[1] - frame.grid_first[1] : data.grid_extent.height,
        frame.partial_grid ? frame.grid_end[2] - frame.grid_first[2] : data.grid_extent.depth
    };

    const std::uint32_t total =
        ((extent[0] + 3u) / 4u) * ((extent[1] + 3u) / 4u) *
        ((extent[2] + 3u) / 4u);

    return {data.point_count, frame.dirty ? std::min(frame.gi_cursor, total) : total, total, frame.dirty, frame.grid_initialized};
}

lighting_statistics lighting_stats(std::uint32_t image)
{
    const auto &data = get();
    const auto &frame = data.frames.at(image);
    std::uint32_t eligible = 0;

    for (std::uint32_t i = 0; i < data.light_count && i < 64u; ++i)
        if ((data.dynamic_shadow_mask[i >> 5u] & (1u << (i & 31u))) != 0u)
            ++eligible;
    std::uint32_t lit_tiles = 0;
    std::uint64_t references = 0;

    const auto popcount = [](std::uint32_t word)
    {
        std::uint32_t count = 0;

        while (word)
        {
            word &= word - 1u;
            ++count;
        }

        return count;
    };

    for (std::size_t i = 0; i + 1u < frame.tile_cpu.size(); i += 2u)
    {
        const std::uint32_t count =
            popcount(frame.tile_cpu[i]) +
            popcount(frame.tile_cpu[i + 1u]);
        lit_tiles += count != 0u;
        references += count;
    }

    return {data.light_count, eligible, data.lighting_pixel_rate, frame.dynamic_count, lit_tiles, data.tile_columns * data.tile_rows, references};
}

void updated(std::uint32_t image)
{
    auto &frame = get().frames.at(image);

    if (!frame.dirty) return;
    frame.cloud_submitted = true;

    if (!frame.gi_complete) return;
    frame.dirty = false;
    frame.partial_grid = false;
    frame.cloud_partial = false;
    frame.gi_cursor = 0;
    frame.gi_complete = false;
    frame.cloud_submitted = false;
}

void set_exposure(float exposure)
{
    exposure_value.store(std::max(0.0f, exposure), std::memory_order_relaxed);
}

void create(const std::vector<gbuffer> &gbuffers,
            const configuration &shaders)
{
    const auto *swapchain = vk::swapchain::get();

    if (!swapchain)
        throw std::logic_error("::vs::pcloud::create requires an initialized VulkanWrapper swapchain");
    create(shaders, gbuffers,
           presentation_target{swapchain->images, swapchain->format, swapchain->extent});
}

void create(const configuration &shaders,
            const std::vector<gbuffer> &gbuffers,
            const presentation_target &output)
{
    auto &data = get();
    data.shader_paths = shaders;

    if (output.images.empty() || output.format == vk::Format::eUndefined ||
        !output.extent.width || !output.extent.height ||
        gbuffers.size() != output.images.size())
        throw std::invalid_argument("pc presentation/G-buffer mismatch");
    data.output = output;

    const auto device = vk::device::get()->handle;

    try
    {
        vk::SamplerCreateInfo sampler{};
        sampler.magFilter = vk::Filter::eNearest;
        sampler.minFilter = vk::Filter::eNearest;
        sampler.mipmapMode = vk::SamplerMipmapMode::eNearest;
        sampler.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        sampler.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        sampler.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        data.sampler = device.createSampler(sampler);

        const auto count = static_cast<std::uint32_t>(get().output.images.size());
        const auto extent = get().output.extent;
        data.tile_columns = (extent.width + tile_size - 1u) / tile_size;
        data.tile_rows = (extent.height + tile_size - 1u) / tile_size;
        data.targets = std::make_unique<::vs::render::postprocess_targets>();
        data.targets->create(count, get().output.extent);
        data.presentation = std::make_unique<::vs::render::presentation_targets>();
        data.presentation->create(get().output.images,
                                  get().output.format, get().output.extent);
        data.cloud_sets = std::make_unique<::vs::gpu::descriptors>();
        data.cloud_sets->create(bindings(0, 0, vk::ShaderStageFlagBits::eCompute, 12), count);
        data.gi_sets = std::make_unique<::vs::gpu::descriptors>();
        data.gi_sets->create(bindings(0, 1, vk::ShaderStageFlagBits::eCompute, 14), count);
        data.lighting_sets = std::make_unique<::vs::gpu::descriptors>();
        data.lighting_sets->create(bindings(5, 1, vk::ShaderStageFlagBits::eCompute, 15), count * 2);
        data.tone_sets = std::make_unique<::vs::gpu::descriptors>();
        data.tone_sets->create(bindings(1, 1, vk::ShaderStageFlagBits::eCompute), count * 2);
        data.present_sets = std::make_unique<::vs::gpu::descriptors>();
        data.present_sets->create(bindings(1, 0, vk::ShaderStageFlagBits::eFragment), count);
        data.present_layout = pipeline_layout(data.present_sets->layout());
        data.cloud_program = std::make_unique<::vs::gpu::compute>();
        data.cloud_program->create(data.shader_paths.cloud_shader,
                                   {data.cloud_sets->layout()}, sizeof(cloud_push));
        data.gi_program = std::make_unique<::vs::gpu::compute>();
        data.gi_program->create(data.shader_paths.gi_shader,
                                {data.gi_sets->layout()}, sizeof(grid_push));
        data.lighting_program = std::make_unique<::vs::gpu::compute>();
        data.lighting_program->create(
            data.shader_paths.lighting_shader,
            {data.lighting_sets->layout()}, sizeof(push));
        if (!data.shader_paths.opaque_lighting_shader.empty()) {
            data.opaque_lighting_program = std::make_unique<::vs::gpu::compute>();
            data.opaque_lighting_program->create(
                data.shader_paths.opaque_lighting_shader,
                {data.lighting_sets->layout()}, sizeof(push));
        }

        data.tone_program = std::make_unique<::vs::gpu::compute>();
        data.tone_program->create(
            data.shader_paths.tonemap_shader,
            {data.tone_sets->layout()}, sizeof(push));

        data.present_program = std::make_unique<::vs::render::pipeline>();
        ::vs::render::raster_state raster{};
        raster.pass = data.presentation->pass();
        raster.layout = data.present_layout;
        raster.cull = vk::CullModeFlagBits::eNone;
        raster.depth_test = false;
        raster.depth_write = false;
        data.present_program->create(raster, {{vk::ShaderStageFlagBits::eVertex,
                                               data.shader_paths.fullscreen_shader},
                                              {vk::ShaderStageFlagBits::eFragment,
                                               data.shader_paths.presentation_shader}});

        const auto storage = vk::BufferUsageFlagBits::eStorageBuffer;
        const auto memory = vk::MemoryPropertyFlagBits::eHostVisible |
                            vk::MemoryPropertyFlagBits::eHostCoherent;
        data.lights = ::vs::gpu::create_buffer(64 * sizeof(point_light), storage, memory);
        data.light_shapes = ::vs::gpu::create_buffer(
            64 * sizeof(::vs::gpu::light_shape), storage, memory);
        data.triangles = ::vs::gpu::create_buffer(3 * sizeof(std::array<float, 4>), storage, memory);
        data.nodes = ::vs::gpu::create_buffer(sizeof(shadow_node), storage, memory);
        data.shadow_light_ranges = ::vs::gpu::create_buffer(
            64 * sizeof(::vs::bvh::light_tree_range), storage, memory);
        data.region_shadows.create(count);
        const std::array<::vs::bvh::light_tree_range, 64> initial_ranges{};
        vk::buffer::write(data.shadow_light_ranges, initial_ranges.data(),
                          sizeof(initial_ranges));
        data.neighborhood_offsets = ::vs::gpu::create_buffer(2 * sizeof(std::uint32_t), storage, memory);
        data.neighborhood_indices = ::vs::gpu::create_buffer(sizeof(std::uint32_t), storage, memory);
        // Always bind a valid 64-word no-glass header. Full directional
        // caches are allocated only when a transmitted scene is installed.
        data.transmission = ::vs::gpu::create_buffer(
            ::vs::lighting::transmission_header_words * sizeof(std::uint32_t), storage, memory);
        const std::array<std::uint32_t, ::vs::lighting::transmission_header_words> no_glass{};
        vk::buffer::write(data.transmission, no_glass.data(), sizeof(no_glass));
        data.glass_vertices = ::vs::gpu::create_buffer(
            3 * sizeof(std::array<float, 4>), storage, memory);
        data.glass_nodes = ::vs::gpu::create_buffer(sizeof(shadow_node), storage, memory);
        data.glass_filters = ::vs::gpu::create_buffer(
            sizeof(std::array<float, 4>), storage, memory);

        data.frames.reserve(count);
        data.frame_graphs.resize(count);

        for (std::uint32_t i = 0; i < count; ++i)
        {
            data.frames.emplace_back();
            auto &current = data.frames.back();

            const auto attachments = data.targets->at(i);
            current.hdr = {attachments.hdr, attachments.hdr_view};
            current.ldr = {attachments.ldr, attachments.ldr_view};
            current.points = ::vs::gpu::create_buffer(sizeof(probe_point), storage, memory);
            current.dynamic_vertices = ::vs::gpu::create_buffer(
                3 * sizeof(std::array<float, 4>), storage, memory);
            current.dynamic_nodes = ::vs::gpu::create_buffer(
                16 + sizeof(shadow_node), storage, memory);
            current.changed_points = ::vs::gpu::create_buffer(
                sizeof(std::uint32_t), storage, memory);
            const std::array<std::uint32_t, 4> empty_dynamic{};
            vk::buffer::write(current.dynamic_nodes, empty_dynamic.data(),
                              sizeof(empty_dynamic));
            current.tile_cpu.resize(std::size_t(data.tile_columns) *
                                    data.tile_rows * tile_mask_words);
            current.tile_lights = ::vs::gpu::create_buffer(
                current.tile_cpu.size() * sizeof(std::uint32_t), storage, memory);

            create_grid(current, data.grid_extent);

            current.cloud = data.cloud_sets->set(i);
            current.gi = data.gi_sets->set(i);
            current.lighting = data.lighting_sets->set(i);
            current.tone = data.tone_sets->set(i);
            current.presentation = data.present_sets->set(i);

            const auto g = gbuffers.at(i);

            vk::DescriptorImageInfo grid_storage{};
            grid_storage.imageView = current.grid_view;
            grid_storage.imageLayout = vk::ImageLayout::eGeneral;
            data.gi_sets->write_image(i, 0, grid_storage);
            data.cloud_sets->write_buffer(i, 0, vk::buffer::descriptor(current.points));
            data.cloud_sets->write_buffer(i, 1, vk::buffer::descriptor(data.triangles));
            data.cloud_sets->write_buffer(i, 2, vk::buffer::descriptor(data.lights));
            data.cloud_sets->write_buffer(i, 3, vk::buffer::descriptor(data.nodes));
            data.cloud_sets->write_buffer(i, 4, vk::buffer::descriptor(data.transmission));
            data.cloud_sets->write_buffer(i, 5, vk::buffer::descriptor(data.light_shapes));
            data.cloud_sets->write_buffer(i, 6, vk::buffer::descriptor(current.dynamic_vertices));
            data.cloud_sets->write_buffer(i, 7, vk::buffer::descriptor(current.dynamic_nodes));
            data.cloud_sets->write_buffer(i, 8, vk::buffer::descriptor(current.changed_points));

            const auto region = data.region_shadows.descriptors(i);
            data.cloud_sets->write_buffer(i, 9, region.records);
            data.cloud_sets->write_buffer(i, 10, region.vertices);
            data.cloud_sets->write_buffer(i, 11, region.nodes);
            data.gi_sets->write_buffer(i, 1, vk::buffer::descriptor(current.points));
            data.gi_sets->write_buffer(i, 2, vk::buffer::descriptor(data.triangles));
            data.gi_sets->write_buffer(i, 3, vk::buffer::descriptor(data.nodes));
            data.gi_sets->write_buffer(i, 4, vk::buffer::descriptor(data.lights));
            data.gi_sets->write_buffer(i, 5, vk::buffer::descriptor(data.neighborhood_offsets));
            data.gi_sets->write_buffer(i, 6, vk::buffer::descriptor(data.neighborhood_indices));
            data.gi_sets->write_buffer(i, 7, vk::buffer::descriptor(data.glass_vertices));
            data.gi_sets->write_buffer(i, 8, vk::buffer::descriptor(data.glass_nodes));
            data.gi_sets->write_buffer(i, 9, vk::buffer::descriptor(data.glass_filters));
            data.gi_sets->write_buffer(i, 10, vk::buffer::descriptor(current.dynamic_vertices));
            data.gi_sets->write_buffer(i, 11, vk::buffer::descriptor(current.dynamic_nodes));
            data.gi_sets->write_buffer(i, 12, region.records);
            data.gi_sets->write_buffer(i, 13, region.vertices);
            data.gi_sets->write_buffer(i, 14, region.nodes);

            std::array<vk::DescriptorImageInfo, 6> lighting{};

            for (std::size_t j = 0; j < g.size(); ++j)
            {
                lighting[j].sampler = data.sampler;
                lighting[j].imageView = g[j];
                lighting[j].imageLayout = j == 3
                                              ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                              : vk::ImageLayout::eShaderReadOnlyOptimal;
            }

            lighting[4].sampler = data.sampler;
            lighting[4].imageView = current.grid_view;
            lighting[4].imageLayout = vk::ImageLayout::eGeneral;
            lighting[5].imageView = current.hdr.view;
            lighting[5].imageLayout = vk::ImageLayout::eGeneral;

            for (std::uint32_t j = 0; j < lighting.size(); ++j)
                data.lighting_sets->write_image(i, j, lighting[j]);
            data.lighting_sets->write_buffer(i, 6, vk::buffer::descriptor(data.lights));
            data.lighting_sets->write_buffer(i, 7, vk::buffer::descriptor(data.triangles));
            data.lighting_sets->write_buffer(i, 8, vk::buffer::descriptor(data.nodes));
            data.lighting_sets->write_buffer(i, 9, vk::buffer::descriptor(current.tile_lights));
            data.lighting_sets->write_buffer(i, 10, vk::buffer::descriptor(data.shadow_light_ranges));
            data.lighting_sets->write_buffer(i, 11, vk::buffer::descriptor(data.transmission));
            data.lighting_sets->write_buffer(i, 12,
                                             vk::buffer::descriptor(data.glass_vertices));
            data.lighting_sets->write_buffer(i, 13,
                                             vk::buffer::descriptor(data.glass_nodes));
            data.lighting_sets->write_buffer(i, 14,
                                             vk::buffer::descriptor(data.glass_filters));
            data.lighting_sets->write_buffer(i, 15,
                                             vk::buffer::descriptor(data.light_shapes));
            data.lighting_sets->write_buffer(i, 16,
                                             vk::buffer::descriptor(current.dynamic_vertices));
            data.lighting_sets->write_buffer(i, 17,
                                             vk::buffer::descriptor(current.dynamic_nodes));
            data.lighting_sets->write_buffer(i, 18, region.records);
            data.lighting_sets->write_buffer(i, 19, region.vertices);
            data.lighting_sets->write_buffer(i, 20, region.nodes);
            std::array<vk::DescriptorImageInfo, 2> tone{};
            tone[0].sampler = data.sampler;
            tone[0].imageView = current.hdr.view;
            tone[0].imageLayout = vk::ImageLayout::eGeneral;
            tone[1].imageView = current.ldr.view;
            tone[1].imageLayout = vk::ImageLayout::eGeneral;

            for (std::uint32_t j = 0; j < tone.size(); ++j)
                data.tone_sets->write_image(i, j, tone[j]);
            std::array<vk::DescriptorImageInfo, 1> presented{};
            presented[0].sampler = data.sampler;
            presented[0].imageView = current.ldr.view;
            presented[0].imageLayout = vk::ImageLayout::eGeneral;
            data.present_sets->write_image(i, 0, presented[0]);
        }
    }

    catch (...)
    {
        release();
        throw;
    }
}

void create_reflection(const std::vector<gbuffer> &gbuffers)
{
    auto &data = get();

    const auto count = std::uint32_t(data.frames.size());

    if (!count || gbuffers.size() != count || !data.reflections.empty())
        throw std::invalid_argument("pc: invalid reflected G-buffer count");
    data.reflection_targets = std::make_unique<::vs::render::postprocess_targets>();
    data.reflection_targets->create(count, data.output.extent);

    const auto storage = vk::BufferUsageFlagBits::eStorageBuffer;
    const auto host = vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent;
    data.reflections.resize(count);

    for (std::uint32_t i = 0; i < count; ++i)
    {
        auto &current = data.reflections[i];

        const auto target = data.reflection_targets->at(i);
        current.hdr = {target.hdr, target.hdr_view};
        current.ldr = {target.ldr, target.ldr_view};
        current.lighting = data.lighting_sets->set(i + count);
        current.tone = data.tone_sets->set(i + count);
        current.tile_lights = ::vs::gpu::create_buffer(
            std::size_t(data.tile_columns) * data.tile_rows * tile_mask_words *
                sizeof(std::uint32_t),
            storage, host);
        const auto &g = gbuffers[i];

        for (std::uint32_t j = 0; j < g.size(); ++j)
        {
            vk::DescriptorImageInfo descriptor{};
            descriptor.sampler = data.sampler;
            descriptor.imageView = g[j];
            descriptor.imageLayout = j == 3 ? vk::ImageLayout::eDepthStencilReadOnlyOptimal : vk::ImageLayout::eShaderReadOnlyOptimal;
            data.lighting_sets->write_image(i + count, j, descriptor);
        }

        vk::DescriptorImageInfo grid{};
        grid.sampler = data.sampler;
        grid.imageView = data.frames[i].grid_view;
        grid.imageLayout = vk::ImageLayout::eGeneral;
        data.lighting_sets->write_image(i + count, 4, grid);
        vk::DescriptorImageInfo hdr{};
        hdr.imageView = current.hdr.view;
        hdr.imageLayout = vk::ImageLayout::eGeneral;
        data.lighting_sets->write_image(i + count, 5, hdr);
        data.lighting_sets->write_buffer(i + count, 6,
                                         vk::buffer::descriptor(data.lights));
        data.lighting_sets->write_buffer(i + count, 7,
                                         vk::buffer::descriptor(data.triangles));
        data.lighting_sets->write_buffer(i + count, 8,
                                         vk::buffer::descriptor(data.nodes));
        data.lighting_sets->write_buffer(i + count, 9,
                                         vk::buffer::descriptor(current.tile_lights));
        data.lighting_sets->write_buffer(i + count, 10,
                                         vk::buffer::descriptor(data.shadow_light_ranges));
        data.lighting_sets->write_buffer(i + count, 11,
                                         vk::buffer::descriptor(data.transmission));
        data.lighting_sets->write_buffer(i + count, 12,
                                         vk::buffer::descriptor(data.glass_vertices));
        data.lighting_sets->write_buffer(i + count, 13,
                                         vk::buffer::descriptor(data.glass_nodes));
        data.lighting_sets->write_buffer(i + count, 14,
                                         vk::buffer::descriptor(data.glass_filters));
        data.lighting_sets->write_buffer(i + count, 15,
                                         vk::buffer::descriptor(data.light_shapes));
        data.lighting_sets->write_buffer(i + count, 16,
                                         vk::buffer::descriptor(data.frames[i].dynamic_vertices));
        data.lighting_sets->write_buffer(i + count, 17,
                                         vk::buffer::descriptor(data.frames[i].dynamic_nodes));
        const auto region = data.region_shadows.descriptors(i);
        data.lighting_sets->write_buffer(i + count, 18, region.records);
        data.lighting_sets->write_buffer(i + count, 19, region.vertices);
        data.lighting_sets->write_buffer(i + count, 20, region.nodes);
        vk::DescriptorImageInfo tone{};
        tone.sampler = data.sampler;
        tone.imageView = current.hdr.view;
        tone.imageLayout = vk::ImageLayout::eGeneral;
        data.tone_sets->write_image(i + count, 0, tone);
        tone.imageView = current.ldr.view;
        data.tone_sets->write_image(i + count, 1, tone);
    }
}

void reflection_lighting(vk::CommandBuffer command, std::uint32_t image,
                         const view_basis &view, vk::Rect2D pixels)
{
    get().region_shadows.upload(image);
    const auto &data = get();
    const auto &current = data.reflections.at(image);

    const auto extent = data.output.extent;

    constexpr std::uint32_t rate = 4u;

    const auto area = reflection_area(extent, pixels, 8u * rate);
    const float tangent = std::tan(view.fov * .5f);
    const float aspect = float(extent.width) / float(extent.height);

    const push controls{
        exposure_value.load(std::memory_order_relaxed),
        indirect_lighting_enabled() ? 1u : 0u,
        data.dynamic_shadow_mask,
        {view.eye.x, view.eye.y, view.eye.z, .1f},
        {view.forward.x, view.forward.y, view.forward.z, 1000.f},
        {view.right.x, view.right.y, view.right.z, tangent * aspect},
        {view.up.x, view.up.y, view.up.z, tangent},
        {float(data.light_count), -1.f,
         0.f, float(rate)},
        {data.grid_origin.x, data.grid_origin.y, data.grid_origin.z,
         data.grid_spacing},
        direct_brdf_enabled() ? 1u : 0u,
        lighting_reconstruction_enabled() ? 1u : 0u,
        direct_lighting_enabled() ? 1u : 0u,
        area.origin
    };

    data.lighting_program->dispatch(command,
                                    area.columns, area.rows, 1,
                                    {current.lighting}, &controls, sizeof(controls));
}

void reflection_tonemap(vk::CommandBuffer command, std::uint32_t image,
                        vk::Rect2D pixels)
{
    const auto &data = get();

    const auto area = reflection_area(data.output.extent, pixels,
                                      tone_workgroup_size);

    push controls{};
    controls.exposure = exposure_value.load(std::memory_order_relaxed);
    controls.dispatch_origin = area.origin;
    data.tone_program->dispatch(command, area.columns, area.rows, 1,
                                {data.reflections.at(image).tone}, &controls,
                                sizeof(controls));
}

vk::ImageView reflection_ldr_view(std::uint32_t image)
{
    return get().reflections.at(image).ldr.view;
}

targets reflection_images(std::uint32_t image)
{
    const auto &r = get().reflections.at(image);

    return {{}, {}, r.hdr.image, r.ldr.image};
}

// Rogue derives its point cloud from static floor/wall geometry.
// Reuse the server's world-space collision surfaces without expanding
// arbitrary imported mesh polygons into brute-force visibility rays.
void set_scene(std::shared_ptr<const ::vs::scene::data> value)
{
    auto &data = get();
    // Re-submitting a mutable caller-owned scene forces fresh geometry;
    // pointer equality alone must not discard edits to in-place data.
    if (data.snapshot == value) data.snapshot.reset();
    data.incoming = std::move(value);
}

void update_lights(const std::vector<light> &lights)
{
    auto &data = get();

    if (!data.snapshot || !data.lights.handle ||
        lights.size() != data.live_lights.size() ||
        lights.size() != data.light_count)
        throw std::invalid_argument("pc: light update requires identical scene topology");

    const auto active = [](const light &l)
    { return l.enabled && l.strength > 0.f; };
    std::array<std::uint32_t, 3> lower{
        data.grid_extent.width, data.grid_extent.height, data.grid_extent.depth
    };

    std::array<std::uint32_t, 3> upper{};

    for (std::size_t i = 0; i < lights.size(); ++i)
    {
        const auto &a = data.live_lights[i];
        const auto &b = lights[i];

        if (a.position.x != b.position.x || a.position.y != b.position.y ||
            a.position.z != b.position.z || a.radius != b.radius ||
            active(a) != active(b))
            throw std::invalid_argument("pc: light topology changed");

        if (!active(b) ||
            (a.color.x == b.color.x && a.color.y == b.color.y &&
             a.color.z == b.color.z && a.strength == b.strength &&
             a.kind == b.kind && a.direction.x == b.direction.x &&
             a.direction.y == b.direction.y && a.direction.z == b.direction.z &&
             a.cone_degrees == b.cone_degrees))
            continue;
        // A light can affect an indirect grid cell only through
        // a source point inside its radius and the 2.6 m GI gather.
        // The bounding cube is conservative, so no changed cell is
        // omitted and all untouched grid values stay bit-identical.
        const float reach = b.radius + bounce_reach + data.grid_spacing;

        const float center[3]{b.position.x, b.position.y, b.position.z};
        const float origin[3]{
            data.grid_origin.x, data.grid_origin.y, data.grid_origin.z
        };

        const std::uint32_t extent[3]{
            data.grid_extent.width, data.grid_extent.height,
            data.grid_extent.depth
        };

        for (int axis = 0; axis < 3; ++axis)
        {
            const auto clamp_cell = [&](double x)
            {
                return std::uint32_t(std::clamp(
                    x, 0.0, double(extent[axis])));
            };

            const double first = std::floor(
                (double(center[axis]) - reach - origin[axis]) /
                data.grid_spacing);
            const double last = std::ceil(
                                    (double(center[axis]) + reach - origin[axis]) /
                                    data.grid_spacing) +
                                1.0;
            lower[axis] = std::min(lower[axis], clamp_cell(first));
            upper[axis] = std::max(upper[axis], clamp_cell(last));
        }
    }

    std::vector<point_light> packed;
    packed.reserve(lights.size());

    for (const auto &l : lights)
        packed.push_back(::vs::gpu::pack(l));

    if (!packed.empty())
        vk::buffer::write(data.lights, packed.data(),
                          packed.size() * sizeof(point_light));
    std::vector<::vs::gpu::light_shape> shapes;
    shapes.reserve(lights.size());

    for (const auto &l : lights)
        shapes.push_back(::vs::gpu::pack_shape(l));

    if (!shapes.empty()) vk::buffer::write(
        data.light_shapes, shapes.data(),
        shapes.size() * sizeof(::vs::gpu::light_shape));
    data.live_lights = lights;

    if (lower[0] >= upper[0] || lower[1] >= upper[1] ||
        lower[2] >= upper[2]) return;
    for (auto &frame : data.frames)
    {
        frame.cloud_partial = false;
        frame.cloud_submitted = false;
        frame.gi_cursor = 0;
        frame.gi_complete = false;

        if (!frame.dirty) {
            frame.partial_grid = true;
            frame.grid_first = lower;
            frame.grid_end = upper;
            frame.dirty = true;
        }
        else if (frame.partial_grid) {
            for (int axis = 0; axis < 3; ++axis)
            {
                frame.grid_first[axis] = std::min(frame.grid_first[axis], lower[axis]);
                frame.grid_end[axis] = std::max(frame.grid_end[axis], upper[axis]);
            }
        }
        // A pending full-grid update remains full. This handles scene
        // updates before the swapchain image's first completed frame.
    }
}

// One shared cache-install path for full scene loads and glass-only edits.
// Glass updates never touch the existing opaque shadow BVH or GI atlas.
void upload_transmission(const ::vs::scene::data *source)
{
    auto &data = get();

    const auto colored = source ? ::vs::lighting::bake_transmission(*source) : ::vs::lighting::transmission_atlas{};

    const std::array<std::uint32_t, ::vs::lighting::transmission_header_words> empty{};

    const auto *words = colored.words.empty() ? empty.data() : colored.words.data();
    const std::size_t count = colored.words.empty() ? empty.size() : colored.words.size();

    const auto storage = vk::BufferUsageFlagBits::eStorageBuffer;
    const auto memory = vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent;

    auto next = ::vs::gpu::create_buffer(
        count * sizeof(std::uint32_t), storage, memory);
    vk::buffer::write(next, words, count * sizeof(std::uint32_t));

    auto vertices = ::vs::gpu::create_buffer(
        std::max<std::size_t>(3 * sizeof(std::array<float, 4>),
                              colored.triangles.size() * sizeof(std::array<float, 4>)),
        storage, memory);
    auto nodes = ::vs::gpu::create_buffer(
        std::max<std::size_t>(sizeof(shadow_node),
                              colored.nodes.size() * sizeof(shadow_node)),
        storage, memory);
    auto filters = ::vs::gpu::create_buffer(
        std::max<std::size_t>(sizeof(std::array<float, 4>),
                              colored.filters.size() * sizeof(std::array<float, 4>)),
        storage, memory);
    if (!colored.triangles.empty())
        vk::buffer::write(vertices, colored.triangles.data(),
                          colored.triangles.size() * sizeof(std::array<float, 4>));
    if (!colored.nodes.empty())
        vk::buffer::write(nodes, colored.nodes.data(),
                          colored.nodes.size() * sizeof(shadow_node));
    if (!colored.filters.empty())
        vk::buffer::write(filters, colored.filters.data(),
                          colored.filters.size() * sizeof(std::array<float, 4>));
    data.transmission = std::move(next);
    data.glass_vertices = std::move(vertices);
    data.glass_nodes = std::move(nodes);
    data.glass_filters = std::move(filters);
    data.glass_node_count = std::uint32_t(colored.nodes.size());

    for (std::uint32_t i = 0; i < data.frames.size(); ++i)
    {
        data.cloud_sets->write_buffer(i, 4,
                                      vk::buffer::descriptor(data.transmission));
        data.gi_sets->write_buffer(i, 7,
                                   vk::buffer::descriptor(data.glass_vertices));
        data.gi_sets->write_buffer(i, 8,
                                   vk::buffer::descriptor(data.glass_nodes));
        data.gi_sets->write_buffer(i, 9,
                                   vk::buffer::descriptor(data.glass_filters));
        data.lighting_sets->write_buffer(i, 11,
                                         vk::buffer::descriptor(data.transmission));
        data.lighting_sets->write_buffer(i, 12,
                                         vk::buffer::descriptor(data.glass_vertices));
        data.lighting_sets->write_buffer(i, 13,
                                         vk::buffer::descriptor(data.glass_nodes));
        data.lighting_sets->write_buffer(i, 14,
                                         vk::buffer::descriptor(data.glass_filters));
        if (!data.reflections.empty()) {
            const auto reflected = i + std::uint32_t(data.frames.size());
            data.lighting_sets->write_buffer(reflected, 11,
                                             vk::buffer::descriptor(data.transmission));
            data.lighting_sets->write_buffer(reflected, 12,
                                             vk::buffer::descriptor(data.glass_vertices));
            data.lighting_sets->write_buffer(reflected, 13,
                                             vk::buffer::descriptor(data.glass_nodes));
            data.lighting_sets->write_buffer(reflected, 14,
                                             vk::buffer::descriptor(data.glass_filters));
        }

        data.frames[i].dirty = true;
        data.frames[i].partial_grid = false;
        data.frames[i].gi_cursor = 0;
        data.frames[i].gi_complete = false;
        data.frames[i].cloud_submitted = false;
    }
}

// Recompute direct-radiance points and GI cells only within a bounded
// affected region. This handles illumination revisions; changes to
// underlying static triangle topology still require set_scene/sync.
void invalidate_radiance(const collision_box &affected_bounds)
{
    auto &data = get();

    if (!data.snapshot) return;
    using namespace vk::util::math;

    if (!std::isfinite(affected_bounds.minimum.x) ||
        !std::isfinite(affected_bounds.minimum.y) ||
        !std::isfinite(affected_bounds.minimum.z) ||
        !std::isfinite(affected_bounds.maximum.x) ||
        !std::isfinite(affected_bounds.maximum.y) ||
        !std::isfinite(affected_bounds.maximum.z) ||
        affected_bounds.minimum.x > affected_bounds.maximum.x ||
        affected_bounds.minimum.y > affected_bounds.maximum.y ||
        affected_bounds.minimum.z > affected_bounds.maximum.z)
        throw std::invalid_argument("Invalid PCloud radiance invalidation");
    constexpr auto storage = vk::BufferUsageFlagBits::eStorageBuffer;
    constexpr auto host = vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eHostCoherent;

    const auto component = [](vec3 p, int axis)
    {
        return axis == 0 ? p.x : axis == 1 ? p.y
                                           : p.z;
    };

    std::array<std::uint32_t, 3> first{}, last{};

    for (int axis = 0; axis < 3; ++axis)
    {
        const auto origin = component(data.grid_origin, axis);
        const auto low = component(affected_bounds.minimum, axis);
        const auto high = component(affected_bounds.maximum, axis);
        const double start = std::floor((double(low) - bounce_reach - origin) /
                                        data.grid_spacing) -
                             1.0;

        const double finish = std::ceil((double(high) + bounce_reach - origin) /
                                        data.grid_spacing) +
                              2.0;

        const double maximum = double(axis == 0 ? data.grid_extent.width : axis == 1 ? data.grid_extent.height
                                                                                     : data.grid_extent.depth);
        first[axis] = std::uint32_t(std::clamp(start, 0.0, maximum));
        last[axis] = std::uint32_t(std::clamp(finish, 0.0, maximum));
    }

    if (first[0] >= last[0] || first[1] >= last[1] || first[2] >= last[2])
        return;

    for (std::uint32_t image = 0; image < data.frames.size(); ++image)
    {
        auto &frame = data.frames[image];

        if (frame.dirty && !frame.partial_grid) continue;
        auto &affected = frame.affected_cpu;

        if (!frame.dirty || !frame.cloud_partial) affected.clear();

        for (std::uint32_t i = 0; i < data.sample_positions.size(); ++i)
        {
            const auto p = data.sample_positions[i];

            if (p.x < affected_bounds.minimum.x - bounce_reach ||
                p.x > affected_bounds.maximum.x + bounce_reach ||
                p.y < affected_bounds.minimum.y - bounce_reach ||
                p.y > affected_bounds.maximum.y + bounce_reach ||
                p.z < affected_bounds.minimum.z - bounce_reach ||
                p.z > affected_bounds.maximum.z + bounce_reach)
                continue;
            affected.push_back(i);
        }

        std::sort(affected.begin(), affected.end());
        affected.erase(std::unique(affected.begin(), affected.end()), affected.end());
        frame.changed_count = std::uint32_t(affected.size());

        const auto bytes = affected.size() * sizeof(std::uint32_t);

        if (frame.changed_points.size < std::max<std::size_t>(bytes, 4u)) {
            frame.changed_points = ::vs::gpu::create_buffer(
                std::max<std::size_t>(bytes, 4u) * 2u, storage, host);
            data.cloud_sets->write_buffer(image, 8,
                                          vk::buffer::descriptor(frame.changed_points));
        }

        if (bytes) vk::buffer::write(frame.changed_points, affected.data(), bytes);
        frame.cloud_partial = true;
        frame.cloud_submitted = false;
        frame.gi_cursor = 0;
        frame.gi_complete = false;

        if (!frame.dirty) {
            frame.dirty = true;
            frame.partial_grid = true;
            frame.grid_first = first;
            frame.grid_end = last;
        }
        else {
            for (int axis = 0; axis < 3; ++axis)
            {
                frame.grid_first[axis] = std::min(frame.grid_first[axis], first[axis]);
                frame.grid_end[axis] = std::max(frame.grid_end[axis], last[axis]);
            }
        }
    }
}

bool replace_shadow_region(::vs::spatial::region_id id,
                           std::uint64_t revision,
                           std::span<const triangle> faces)
{
    auto &data = get();

    const auto previous = data.region_shadows.bounds(id);

    if (!data.region_shadows.replace(id, revision, faces)) return false;
    // Only the changed region's probes lose their cached radiance.
    if (previous) invalidate_region(*previous);
    if (const auto updated = data.region_shadows.bounds(id))
        invalidate_region(*updated);

    return true;
}

bool remove_shadow_region(::vs::spatial::region_id id,
                          std::uint64_t revision)
{
    auto &data = get();

    const auto previous = data.region_shadows.bounds(id);

    if (!data.region_shadows.erase(id, revision)) return false;
    if (previous) invalidate_region(*previous);

    return true;
}

bool forget_shadow_region(::vs::spatial::region_id id)
{
    auto &data = get();

    const auto previous = data.region_shadows.bounds(id);

    if (!data.region_shadows.forget(id)) return false;
    if (previous) invalidate_region(*previous);

    return true;
}

void clear_shadow_regions()
{
    auto &data = get();

    const auto previous = data.region_shadows.all_bounds();

    data.region_shadows.clear_regions();

    if (previous) invalidate_region(*previous);
}

void invalidate_region(const ::vs::scene::collision_box &region)
{
    auto &data = get();

    if (!data.snapshot || !data.point_count) return;
    const float lo[3]{region.minimum.x, region.minimum.y, region.minimum.z};
    const float hi[3]{region.maximum.x, region.maximum.y, region.maximum.z};
    const float origin[3]{data.grid_origin.x, data.grid_origin.y, data.grid_origin.z};
    const std::uint32_t size[3]{data.grid_extent.width, data.grid_extent.height, data.grid_extent.depth};
    std::array<std::uint32_t, 3> lower{}, upper{};

    for (int axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(lo[axis]) || !std::isfinite(hi[axis]) ||
            lo[axis] > hi[axis])
            throw std::invalid_argument("pc: invalid changed region bounds");

        const double expanded = bounce_reach + data.grid_spacing;
        const auto clamp_cell = [&](double cell)
        {
            return static_cast<std::uint32_t>(std::clamp(
                cell, 0.0, double(size[axis])));
        };

        lower[axis] = clamp_cell(std::floor(
            (double(lo[axis]) - expanded - origin[axis]) / data.grid_spacing));
        upper[axis] = clamp_cell(std::ceil(
                                     (double(hi[axis]) + expanded - origin[axis]) / data.grid_spacing) +
                                 1.0);
        if (lower[axis] >= upper[axis]) return;
    }

    for (auto &frame : data.frames)
    {
        // Refresh radiance because receiver/source materials may change;
        // leave previously populated cells outside this region untouched.
        frame.cloud_partial = false;
        frame.cloud_submitted = false;
        frame.gi_cursor = 0;
        frame.gi_complete = false;
        frame.gi_interval = 0;

        if (!frame.dirty) {
            frame.dirty = true;
            frame.partial_grid = true;
            frame.grid_first = lower;
            frame.grid_end = upper;
        }
        else if (frame.partial_grid) {
            for (int axis = 0; axis < 3; ++axis)
            {
                frame.grid_first[axis] = std::min(frame.grid_first[axis], lower[axis]);
                frame.grid_end[axis] = std::max(frame.grid_end[axis], upper[axis]);
            }
        }
    }
}

void update_transmission(const ::vs::scene::data &source)
{
    auto &data = get();

    if (!data.snapshot || source.lights.size() != data.live_lights.size())
        throw std::invalid_argument("pc: transmission update needs matching lights");

    for (std::size_t i = 0; i < source.lights.size(); ++i)
    {
        const auto &a = data.live_lights[i];
        const auto &b = source.lights[i];

        const auto active = [](const light &l)
        {
            return l.enabled && l.strength > 0.f;
        };

        if (a.position.x != b.position.x || a.position.y != b.position.y ||
            a.position.z != b.position.z || a.radius != b.radius ||
            active(a) != active(b))
            throw std::invalid_argument("pc: glass-only update changed light topology");
    }

    upload_transmission(&source);
}

void set_probe_coverage(std::optional<::vs::scene::collision_box> bounds)
{
    auto &data = get();

    if (bounds && (bounds->maximum.x < bounds->minimum.x ||
                   bounds->maximum.y < bounds->minimum.y ||
                   bounds->maximum.z < bounds->minimum.z ||
                   !std::isfinite(bounds->minimum.x) || !std::isfinite(bounds->minimum.y) ||
                   !std::isfinite(bounds->minimum.z) || !std::isfinite(bounds->maximum.x) ||
                   !std::isfinite(bounds->maximum.y) || !std::isfinite(bounds->maximum.z)))
        throw std::invalid_argument("Invalid PCloud coverage bounds");
    data.probe_coverage = bounds;
    data.coverage_dirty = true;
}

void synchronize_scene()
{
    auto &data = get();

    const auto source = data.incoming;
    if (source == data.snapshot && !data.coverage_dirty) return;

    using namespace vk::util::math;

    // Cover every server-authored light radius and its local bounce reach.
    // Preserve the full spatial coverage; reduce probe density if necessary
    // rather than silently cropping a light to a hardcoded box.
    vec3 origin{};

    float spacing = min_grid_spacing;

    vk::Extent3D extent{1, 1, 1};

    if (source && (!source->lights.empty() || data.probe_coverage)) {
        const float inf = std::numeric_limits<float>::infinity();
        vec3 minimum{inf, inf, inf}, maximum{-inf, -inf, -inf};

        bool any = false;

        for (const auto &light : source->lights)
        {
            if (!std::isfinite(light.position.x) ||
                !std::isfinite(light.position.y) ||
                !std::isfinite(light.position.z) ||
                !std::isfinite(light.radius) || light.radius <= 0.0f ||
                !std::isfinite(light.strength))
                throw std::runtime_error("Invalid streamed light bounds");

            if (!light.enabled || light.strength <= 0.0f) continue;

            const float reach = light.radius + bounce_reach;
            const vec3 low = light.position - vec3{reach, reach, reach};
            const vec3 high = light.position + vec3{reach, reach, reach};
            minimum = {std::min(minimum.x, low.x), std::min(minimum.y, low.y), std::min(minimum.z, low.z)};
            maximum = {std::max(maximum.x, high.x), std::max(maximum.y, high.y), std::max(maximum.z, high.z)};
            any = true;
        }

        if (data.probe_coverage) {
            const auto &b = *data.probe_coverage;
            minimum = {
                std::min(minimum.x, b.minimum.x),
                std::min(minimum.y, b.minimum.y),
                std::min(minimum.z, b.minimum.z)
            };

            maximum = {
                std::max(maximum.x, b.maximum.x),
                std::max(maximum.y, b.maximum.y),
                std::max(maximum.z, b.maximum.z)
            };

            any = true;
        }

        if (any) {
            const vec3 dimensions = maximum - minimum;
            const auto max_dim = vk::physical_device::get()->handle.getProperties().limits.maxImageDimension3D;
            const double dx = dimensions.x, dy = dimensions.y, dz = dimensions.z;
            double step = std::max({double(min_grid_spacing), std::max({dx, dy, dz}) / std::max(2u, max_dim - 1u), std::cbrt((dx * dy * dz) / double(max_grid_probes))});

            for (;;)
            {
                const auto nx = std::uint32_t(std::ceil(dx / step)) + 1u;
                const auto ny = std::uint32_t(std::ceil(dy / step)) + 1u;
                const auto nz = std::uint32_t(std::ceil(dz / step)) + 1u;

                if (nx <= max_dim && ny <= max_dim && nz <= max_dim &&
                    std::uint64_t(nx) * ny * nz <= max_grid_probes)
                {
                    extent = vk::Extent3D{nx, ny, nz};

                    break;
                }

                step *= 1.1;
            }

            origin = minimum;
            spacing = static_cast<float>(step);
        }
    }

    const vec3 limit{
        origin.x + (extent.width - 1) * spacing,
        origin.y + (extent.height - 1) * spacing,
        origin.z + (extent.depth - 1) * spacing
    };

    std::vector<point_light> lights;
    std::vector<probe_point> points;
    std::vector<std::array<float, 4>> triangles;

    if (source) {
        if (source->lights.size() > 64)
            throw std::runtime_error("Scene exceeds 64 point lights");

        for (const auto &light : source->lights)
            lights.push_back(::vs::gpu::pack(light));

        // Rogue's probe cloud is built from static room surfaces, not
        // every polygon of imported character/prop meshes. The server
        // already streams those static surfaces as collision boxes.
        const auto face = [&](vec3 base, vec3 u, vec3 v, vec3 normal)
        {
            // Samples use the original 0.3-unit surface spacing.
            // Restrict iteration to the probe volume, even for a huge floor.
            const float lu = length(u), lv = length(v);

            if (lu <= 0.0f || lv <= 0.0f) return;

            const vec3 U = u / lu, V = v / lv;
            const int nu = std::max(1, int(std::ceil(lu / 0.3f)));
            const int nv = std::max(1, int(std::ceil(lv / 0.3f)));
            const float du = lu / float(nu), dv = lv / float(nv);
            const auto interval = [&](vec3 axis, float step, int count)
            {
                // Project the complete probe AABB onto this face axis.
                // Opposite corners alone miss parts of rotated faces.
                const vec3 center = (origin + limit) * 0.5f;
                const vec3 half = (limit - origin) * 0.5f;
                const float radius = std::abs(axis.x) * half.x +
                                     std::abs(axis.y) * half.y +
                                     std::abs(axis.z) * half.z;

                const float lo = dot(center - base, axis) - radius;
                const float hi = dot(center - base, axis) + radius;
                const int begin = std::clamp(int(std::ceil(lo / step - 0.5f)), 0, count);
                const int finish = std::clamp(int(std::ceil(hi / step + 0.5f)), 0, count);

                return std::pair{begin, finish};
            };

            const auto [u0, u1] = interval(U, du, nu);
            const auto [v0, v1] = interval(V, dv, nv);

            for (int i = u0; i < u1; ++i)
                for (int j = v0; j < v1; ++j)
                {
                    const vec3 p = base + U * ((float(i) + 0.5f) * du) + V * ((float(j) + 0.5f) * dv);

                    if (p.x < origin.x || p.y < origin.y || p.z < origin.z ||
                        p.x > limit.x || p.y > limit.y || p.z > limit.z) continue;
                    // Do not generate surface samples in dark gaps between lights.
                    bool reached = data.probe_coverage &&
                                   p.x >= data.probe_coverage->minimum.x &&
                                   p.y >= data.probe_coverage->minimum.y &&
                                   p.z >= data.probe_coverage->minimum.z &&
                                   p.x <= data.probe_coverage->maximum.x &&
                                   p.y <= data.probe_coverage->maximum.y &&
                                   p.z <= data.probe_coverage->maximum.z;

                    for (const auto &light : source->lights)
                    {
                        const float radius = light.radius + bounce_reach;

                        if (light.enabled && light.strength > 0.0f &&
                            length_squared(p - light.position) <= radius * radius)
                        {
                            reached = true;

                            break;
                        }
                    }

                    if (!reached) continue;
                    points.push_back({{p.x, p.y, p.z, 0.0f}, {normal.x, normal.y, normal.z, 0.0f}, {}});
                }
        };

        for (const auto &box : source->collision_boxes)
        {
            // Keep all casters that can intersect a light's radius,
            // including those just outside the probe volume.
            const bool inside =
                box.maximum.x >= origin.x - bounce_reach &&
                box.maximum.y >= origin.y - bounce_reach &&
                box.maximum.z >= origin.z - bounce_reach &&
                box.minimum.x <= limit.x + bounce_reach &&
                box.minimum.y <= limit.y + bounce_reach &&
                box.minimum.z <= limit.z + bounce_reach;

            bool near_light = false;

            for (const auto &light : source->lights)
            {
                const vec3 c{
                    std::clamp(light.position.x, box.minimum.x, box.maximum.x),
                    std::clamp(light.position.y, box.minimum.y, box.maximum.y),
                    std::clamp(light.position.z, box.minimum.z, box.maximum.z)
                };

                if (length_squared(c - light.position) <= light.radius * light.radius) {
                    near_light = true;

                    break;
                }
            }

            if (!inside && !near_light) continue;

            const auto a = box.minimum, b = box.maximum;
            const float dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
            face({b.x, a.y, a.z}, {0, dy, 0}, {0, 0, dz}, {1, 0, 0});
            face({a.x, a.y, b.z}, {0, dy, 0}, {0, 0, -dz}, {-1, 0, 0});
            face({a.x, b.y, b.z}, {dx, 0, 0}, {0, 0, -dz}, {0, 1, 0});
            face({a.x, a.y, a.z}, {dx, 0, 0}, {0, 0, dz}, {0, -1, 0});
            face({a.x, a.y, b.z}, {dx, 0, 0}, {0, dy, 0}, {0, 0, 1});
            face({b.x, a.y, a.z}, {-dx, 0, 0}, {0, dy, 0}, {0, 0, -1});
        }
        // Both procedural faces and instanced mesh faces come from the
        // same application-owned vs::scene::data used by PMap and Lumen.
        ::vs::scene::each_surface(*source, [&](const ::vs::scene::surface &s)
                                  {
                const vec3 a=s.origin,b=a+s.u,c=a+s.v;
                const vec3 n=length_squared(s.normal)>.5f?s.normal:
                    vk::util::math::normalize(cross(s.u,s.v));
                if(length_squared(cross(s.u,s.v))<1e-12f)return;
                if(s.probe_receiver)face(a,s.u,s.v,n);
                if(!s.shadow_caster)return;
                const auto emit=[&](vec3 x,vec3 y,vec3 z)
                {
                    if(!relevant_triangle({x,y,z},source->lights))return;
                    for(const auto& p:{x,y,z})
                        triangles.push_back({p.x,p.y,p.z,0.f});
                };
                emit(a,b,c);
                if(!s.triangle)emit(b,b+s.v,c); });
    }
    // Build independent CPU acceleration structures concurrently.
    // Only immutable point coordinates cross to the job; GPU resources
    // and descriptor writes remain owned by the render thread.
    std::vector<vec3> surface_positions;
    surface_positions.reserve(points.size());

    for (const auto &point : points)
        surface_positions.push_back({point.position[0], point.position[1], point.position[2]});
    ::vs::jobs::pool preparation;
    preparation.start(1);
    std::optional<::vs::spatial::bricked_neighborhood> indexed;

    if (!preparation.submit(
            [positions = std::move(surface_positions), origin, spacing, extent,
             &indexed]() mutable -> ::vs::jobs::completion
            {
                auto result = ::vs::spatial::build_bricked_neighborhood(
                    positions, origin, spacing,
                    {extent.width, extent.height, extent.depth},
                    bounce_reach + 0.0001f);
                return [&indexed, result = std::move(result)]() mutable
                {
                    indexed = std::move(result);
                };
            }))
        throw std::runtime_error("Failed to schedule lighting preparation");

    auto nodes = ::vs::bvh::build_stackless(triangles);
    // The original tree remains at the beginning of both buffers; GI
    // and probe-cloud queries continue using its original counts.
    const auto shared_nodes = static_cast<std::uint32_t>(nodes.size());
    const auto shared_vertices = static_cast<std::uint32_t>(triangles.size());

    std::vector<::vs::bvh::local_light> local_lights;
    local_lights.reserve(lights.size());

    for (const auto &light : lights)
        local_lights.push_back({{light.position_range[0], light.position_range[1], light.position_range[2]}, light.position_range[3]});

    const auto light_ranges = ::vs::bvh::append_light_local_trees(
        triangles, nodes, local_lights);
    const auto shadow_triangle_count = triangles.size() / 3u;

    for (const auto &range : light_ranges)
        if (range.first > nodes.size() ||
            range.count > nodes.size() - range.first)
            throw std::logic_error("Static light BVH range exceeds its node buffer");

    for (const auto &node : nodes)
    {
        const auto count = node.first_and_count & 15u;
        if (!count) continue;

        const auto first = node.first_and_count >> 4u;

        if (first > shadow_triangle_count ||
            count > shadow_triangle_count - first)
            throw std::logic_error("Static light BVH leaf exceeds its triangle buffer");
    }

    std::size_t empty_light_trees = 0;
    std::size_t global_light_trees = 0;
    std::size_t local_light_trees = 0;

    for (const auto &range : light_ranges)
    {
        if (!range.count) ++empty_light_trees;
        else if (!range.first)
            ++global_light_trees;
        else
            ++local_light_trees;
    }

    std::cout << "pc: static shadow trees, triangles=" << shared_vertices / 3u
              << ", nodes=" << nodes.size() << ", local="
              << local_light_trees << ", global=" << global_light_trees
              << ", empty=" << empty_light_trees << '\n';
    // Both the shared tree and local subtrees now reference immutable
    // triangles. Encode (origin, edge1, edge2) for all shadow queries.
    ::vs::bvh::store_triangle_edges(triangles);
    preparation.wait_idle();
    preparation.commit(1);

    auto neighborhoods = std::move(indexed.value());
    const bool new_shape =
        extent.width != data.grid_extent.width ||
        extent.height != data.grid_extent.height ||
        extent.depth != data.grid_extent.depth;
    data.grid_origin = origin;
    data.grid_spacing = spacing;
    data.grid_extent = extent;

    if (new_shape) {
        // synchronize_scene() is called after prior GPU frames are idle.
        for (std::uint32_t i = 0; i < data.frames.size(); ++i)
        {
            auto &frame = data.frames[i];
            create_grid(frame, extent);
            vk::DescriptorImageInfo gi{};
            gi.imageView = frame.grid_view;
            gi.imageLayout = vk::ImageLayout::eGeneral;
            data.gi_sets->write_image(i, 0, gi);
            gi.sampler = data.sampler;
            data.lighting_sets->write_image(i, 4, gi);

            if (!data.reflections.empty())
                data.lighting_sets->write_image(i + std::uint32_t(data.frames.size()), 4, gi);
        }
    }

    const auto storage = vk::BufferUsageFlagBits::eStorageBuffer;
    const auto memory = vk::MemoryPropertyFlagBits::eHostVisible |
                        vk::MemoryPropertyFlagBits::eHostCoherent;

    auto new_triangles = ::vs::gpu::create_buffer(
        std::max<std::size_t>(3 * sizeof(std::array<float, 4>),
                              triangles.size() * sizeof(std::array<float, 4>)),
        storage, memory);
    if (!triangles.empty())
        vk::buffer::write(new_triangles, triangles.data(),
                          triangles.size() * sizeof(std::array<float, 4>));
    auto new_nodes = ::vs::gpu::create_buffer(
        std::max<std::size_t>(sizeof(shadow_node),
                              nodes.size() * sizeof(shadow_node)),
        storage, memory);
    if (!nodes.empty())
        vk::buffer::write(new_nodes, nodes.data(), nodes.size() * sizeof(shadow_node));

    if (!lights.empty())
        vk::buffer::write(data.lights, lights.data(),
                          lights.size() * sizeof(point_light));
    if (source && !source->lights.empty()) {
        std::vector<::vs::gpu::light_shape> shapes;
        shapes.reserve(source->lights.size());

        for (const auto &l : source->lights)
            shapes.push_back(::vs::gpu::pack_shape(l));
        vk::buffer::write(data.light_shapes, shapes.data(),
                          shapes.size() * sizeof(::vs::gpu::light_shape));
    }

    data.triangles = std::move(new_triangles);
    data.nodes = std::move(new_nodes);

    static const bool disable_static_shadows = []
    {
        const char *value = std::getenv("VS_STATIC_SHADOWS");

        return value && *value == '0';
    }();

    if (disable_static_shadows) {
        const std::array<::vs::bvh::light_tree_range, 64> no_static_ranges{};
        vk::buffer::write(data.shadow_light_ranges, no_static_ranges.data(),
                          sizeof(no_static_ranges));
        static bool reported = false;

        if (!reported) {
            std::cout << "pc: static shadow queries disabled by diagnostic override\n";
            reported = true;
        }
    }
    else if (!light_ranges.empty())
        vk::buffer::write(data.shadow_light_ranges, light_ranges.data(),
                          light_ranges.size() * sizeof(::vs::bvh::light_tree_range));
    // Ordered candidate lists keep the original sum order for each cell.
    const auto offset_bytes = neighborhoods.offsets.size() * sizeof(std::uint32_t);
    const auto candidate_bytes = neighborhoods.indices.size() * sizeof(std::uint32_t);
    auto new_offsets = ::vs::gpu::create_buffer(
        std::max<std::size_t>(2 * sizeof(std::uint32_t), offset_bytes), storage, memory);
    auto new_candidates = ::vs::gpu::create_buffer(
        std::max<std::size_t>(sizeof(std::uint32_t), candidate_bytes), storage, memory);
    vk::buffer::write(new_offsets, neighborhoods.offsets.data(), offset_bytes);

    if (candidate_bytes)
        vk::buffer::write(new_candidates, neighborhoods.indices.data(), candidate_bytes);
    data.neighborhood_offsets = std::move(new_offsets);
    data.neighborhood_indices = std::move(new_candidates);
    data.node_count = shared_nodes;
    data.visual_node_count = static_cast<std::uint32_t>(nodes.size());
    data.point_count = static_cast<std::uint32_t>(points.size());
    data.vertex_count = shared_vertices;
    data.light_count = static_cast<std::uint32_t>(lights.size());

    for (std::uint32_t i = 0; i < data.frames.size(); ++i)
    {
        auto &frame = data.frames[i];
        frame.points = ::vs::gpu::create_buffer(
            std::max<std::size_t>(sizeof(probe_point), points.size() * sizeof(probe_point)),
            storage, memory);
        if (!points.empty())
            vk::buffer::write(frame.points, points.data(), points.size() * sizeof(probe_point));
        data.cloud_sets->write_buffer(i, 0, vk::buffer::descriptor(frame.points));
        data.cloud_sets->write_buffer(i, 1, vk::buffer::descriptor(data.triangles));
        data.gi_sets->write_buffer(i, 1, vk::buffer::descriptor(frame.points));
        data.gi_sets->write_buffer(i, 2, vk::buffer::descriptor(data.triangles));
        data.lighting_sets->write_buffer(i, 7, vk::buffer::descriptor(data.triangles));
        data.cloud_sets->write_buffer(i, 3, vk::buffer::descriptor(data.nodes));
        data.gi_sets->write_buffer(i, 3, vk::buffer::descriptor(data.nodes));
        data.gi_sets->write_buffer(i, 5, vk::buffer::descriptor(data.neighborhood_offsets));
        data.gi_sets->write_buffer(i, 6, vk::buffer::descriptor(data.neighborhood_indices));
        data.lighting_sets->write_buffer(i, 8, vk::buffer::descriptor(data.nodes));

        if (!data.reflections.empty()) {
            const auto secondary = i + std::uint32_t(data.frames.size());
            data.lighting_sets->write_buffer(secondary, 7,
                                             vk::buffer::descriptor(data.triangles));
            data.lighting_sets->write_buffer(secondary, 8,
                                             vk::buffer::descriptor(data.nodes));
        }
    }

    upload_transmission(source.get());
    data.sample_positions.clear();
    data.sample_positions.reserve(points.size());

    for (const auto &p : points)
        data.sample_positions.push_back({p.position[0], p.position[1], p.position[2]});
    // The caller has synchronized the GPU before static BVH replacement.
    data.dynamic_topology.clear();
    data.dynamic_order.clear();
    data.dynamic_group_signature = 0;
    data.dynamic_reference_cost = 0.f;
    data.dynamic_refit_age = 0;
    data.prepared_nodes.clear();
    data.prepared_vertices.clear();
    data.prepared_primitives = 0;
    data.dynamic_prepared = false;
    // Do not retain eligibility from the previous scene before new
    // animated geometry is prepared. All-one masks are conservative.
    data.dynamic_shadow_mask = {~0u, ~0u};

    for (auto &frame : data.frames)
    {
        frame.has_dynamic_bounds = false;
        frame.gi_interval = 0;
        frame.gi_cursor = 0;
        frame.gi_complete = false;
        frame.grid_initialized = false;
        frame.cloud_submitted = false;
        frame.dynamic_count = 0;
        frame.dynamic_has_spheres = false;
        frame.affected_cpu.clear();
        frame.changed_count = 0;
        frame.cloud_partial = false;
        const std::array<std::uint32_t, 4> empty{};
        vk::buffer::write(frame.dynamic_nodes, empty.data(), sizeof(empty));
    }

    data.snapshot = source;
    data.coverage_dirty = false;
    data.live_lights = source ? source->lights : std::vector<light>{};
    // The reflected view has its own conservative all-light lists.
    // In the single-planar-mirror path there are at most two
    // light emitters, so updating their screen tiles every frame
    // would be more expensive than using the exact full list.
    for (auto &reflected : data.reflections)
    {
        std::vector<std::uint32_t> tiles(
            std::size_t(data.tile_columns) * data.tile_rows * tile_mask_words, 0u);
        for (std::size_t tile = 0; tile < std::size_t(data.tile_columns) * data.tile_rows; ++tile)
        {
            const auto base = tile * tile_mask_words;

            for (std::uint32_t id = 0; id < data.live_lights.size() &&
                                       id < 64u;
                 ++id)
            {
                const auto &l = data.live_lights[id];

                if (l.radius > 0.f && l.strength > 0.f)
                    tiles[base + (id >> 5u)] |= 1u << (id & 31u);
            }
        }

        vk::buffer::write(reflected.tile_lights, tiles.data(),
                          tiles.size() * sizeof(std::uint32_t));
    }
    // The scene update can replace grid images and point buffers.
    // The caller already guaranteed all old GPU submissions are idle.
    for (auto &graph : data.frame_graphs)
        graph.clear_states();

    for (auto &frame : data.frames)
    {
        frame.dirty = true;
        frame.partial_grid = false;
    }
}

void prepare_dynamic_geometry(
    const std::vector<triangle> &faces,
    const std::vector<shadow_sphere> &spheres,
    std::span<const std::uint32_t> face_owners,
    std::span<const std::uint32_t> sphere_owners)
{
    using namespace vk::util::math;
    auto &data = get();

    if (!data.snapshot) return;
    dynamic_preparation.commit();

    // A/B fallback for unvalidated GPUs; all shadow hit tests stay exact.
    static const bool group_acceleration = []
    {
        const char *value = std::getenv("VS_DYNAMIC_BVH_GROUPS");

        return !value || *value != '0';
    }();

    const bool grouped = group_acceleration &&
                         (!face_owners.empty() || !sphere_owners.empty());

    if (grouped && (face_owners.size() != faces.size() ||
                    sphere_owners.size() != spheres.size()))
        throw std::invalid_argument(
            "Dynamic shadow owner counts do not match geometry");

    const std::uint64_t signature =
        grouped ? 1469598103934665603ull : 0ull;

    std::uint64_t group_signature = signature;

    if (grouped) {
        for (const auto owner : face_owners)
            group_signature = (group_signature ^ owner) * 1099511628211ull;

        for (const auto owner : sphere_owners)
            group_signature = (group_signature ^ owner) * 1099511628211ull;
    }

    const auto primitive_count = faces.size() + spheres.size();

    if (data.dynamic_order.size() != primitive_count ||
        data.dynamic_group_signature != group_signature)
    {
        if (data.dynamic_topology_pending) return;
        std::vector<std::array<float, 4>> initial;
        initial.reserve(primitive_count * 3u);

        for (std::uint32_t i = 0; i < faces.size(); ++i)
        {
            const auto &t = faces[i];
            initial.push_back({t.a.x, t.a.y, t.a.z, float(i)});
            initial.push_back({t.b.x, t.b.y, t.b.z, 0.f});
            initial.push_back({t.c.x, t.c.y, t.c.z, 0.f});
        }

        for (std::size_t i = 0; i < spheres.size(); ++i)
        {
            const auto &ball = spheres[i];

            const auto p = ball.center;
            const float r = std::max(ball.radius, .001f);
            const vec3 extent{r, r, r};

            const vec3 lo = p - extent, hi = p + extent;
            initial.push_back({lo.x, lo.y, lo.z, float(faces.size() + i)});
            initial.push_back({hi.x, hi.y, hi.z, 0.f});
            initial.push_back({p.x, p.y, p.z, 0.f});
        }
        // Eight-primitive leaves reduced exact shadow traversal time on M2.
        // Keep the four-primitive comparison override for repeatable A/B runs.
        static const std::size_t dynamic_max_leaf = []
        {
            const char *value = std::getenv("VS_DYNAMIC_BVH_LEAF");

            return value && std::strcmp(value, "4") == 0 ? 4u : 8u;
        }();

        const std::size_t max_leaf = dynamic_max_leaf;

        std::vector<std::uint32_t> owners;

        if (grouped) {
            owners.reserve(primitive_count);
            owners.insert(owners.end(), face_owners.begin(), face_owners.end());
            owners.insert(owners.end(), sphere_owners.begin(), sphere_owners.end());
        }

        if (!dynamic_preparation_started) {
            dynamic_preparation.start(1, 1);
            dynamic_preparation_started = true;
        }

        struct build_result
        {
            std::vector<::vs::bvh::stackless_node> nodes;
            std::vector<std::uint32_t> order;
            std::size_t owner_count = 0;
        };

        auto result = std::make_shared<build_result>();

        data.dynamic_topology_pending = true;
        data.pending_dynamic_signature = group_signature;
        data.pending_dynamic_primitives = primitive_count;

        const bool accepted = dynamic_preparation.submit(
            [&data, initial = std::move(initial), owners = std::move(owners),
             result, grouped, group_signature, primitive_count,
             max_leaf]() mutable -> ::vs::jobs::completion
            {
                if (grouped)
                    result->nodes = ::vs::bvh::build_grouped_stackless(
                        initial, owners, max_leaf / 2u, max_leaf);
                else
                    result->nodes = ::vs::bvh::build_stackless(
                        initial, max_leaf / 2u, max_leaf);
                if (grouped) {
                    std::sort(owners.begin(), owners.end());
                    result->owner_count = std::unique(
                                              owners.begin(), owners.end()) -
                                          owners.begin();
                }
                result->order.resize(primitive_count);
                for (std::size_t i = 0; i < primitive_count; ++i)
                    result->order[i] =
                        static_cast<std::uint32_t>(initial[i * 3u][3]);
                for (const auto &node : result->nodes)
                {
                    const auto count = node.first_and_count & 15u;
                    if (!count) continue;
                    const auto first = node.first_and_count >> 4u;
                    if (first > primitive_count || count > primitive_count - first)
                        throw std::logic_error(
                            "Dynamic shadow BVH leaf exceeds its primitive buffer");
                }
                return [&data, result, grouped, group_signature,
                        primitive_count, max_leaf]
                {
                    data.dynamic_topology_pending = false;
                    if (data.pending_dynamic_signature != group_signature ||
                        data.pending_dynamic_primitives != primitive_count)
                        return;
                    data.dynamic_topology = std::move(result->nodes);
                    data.dynamic_order = std::move(result->order);
                    data.dynamic_group_signature = group_signature;
                    data.dynamic_reference_cost = 0.f;
                    data.dynamic_refit_age = 0;
                    data.prepared_nodes.clear();
                    data.dynamic_prepared = false;
                    if (grouped)
                        std::cout << "pc: dynamic shadow BVH grouped, owners="
                                  << result->owner_count << ", primitives="
                                  << primitive_count << ", nodes="
                                  << data.dynamic_topology.size() << ", leaf="
                                  << max_leaf << '\n';
                    else if (primitive_count)
                        std::cout << "pc: dynamic shadow BVH ungrouped, primitives="
                                  << primitive_count << ", nodes="
                                  << data.dynamic_topology.size() << ", leaf="
                                  << max_leaf << '\n';
                };
            });
        if (!accepted) {
            data.dynamic_topology_pending = false;
            throw std::runtime_error("Dynamic shadow topology queue is full");
        }

        data.dynamic_prepared = false;

        return;
    }

    auto &vertices = data.prepared_vertices;
    vertices.clear();
    vertices.reserve(primitive_count * 3u);
    collision_box bounds{};

    const float inf = std::numeric_limits<float>::infinity();
    bounds.minimum = {inf, inf, inf};
    bounds.maximum = {-inf, -inf, -inf};

    const auto include = [&](vec3 p)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) ||
            !std::isfinite(p.z))
            throw std::runtime_error("pc: non-finite animated geometry");
        bounds.minimum = {
            std::min(bounds.minimum.x, p.x),
            std::min(bounds.minimum.y, p.y),
            std::min(bounds.minimum.z, p.z)
        };

        bounds.maximum = {
            std::max(bounds.maximum.x, p.x),
            std::max(bounds.maximum.y, p.y),
            std::max(bounds.maximum.z, p.z)
        };
    };

    for (const auto original : data.dynamic_order)
    {
        if (original < faces.size()) {
            const auto &face = faces[original];
            include(face.a);
            include(face.b);
            include(face.c);
            vertices.push_back({face.a.x, face.a.y, face.a.z, 0.f});

            const vec3 u = face.b - face.a, v = face.c - face.a;
            vertices.push_back({u.x, u.y, u.z, 0.f});
            vertices.push_back({v.x, v.y, v.z, 0.f});
        }
        else {
            const auto &sphere = spheres.at(original - faces.size());

            if (!std::isfinite(sphere.radius) || sphere.radius <= 0.f)
                throw std::runtime_error("pc: invalid animated sphere radius");
            const vec3 r{sphere.radius, sphere.radius, sphere.radius};
            include(sphere.center - r);
            include(sphere.center + r);
            vertices.push_back({sphere.center.x, sphere.center.y, sphere.center.z, -sphere.radius});
            vertices.push_back({0.f, 0.f, 0.f, 0.f});
            vertices.push_back({0.f, 0.f, 0.f, 0.f});
        }
    }

    auto &nodes = data.prepared_nodes;

    if (nodes.size() != data.dynamic_topology.size())
        nodes = data.dynamic_topology;

    float refit_cost = 0.f;

    for (std::size_t j = nodes.size(); j-- > 0;)
    {
        auto &node = nodes[j];
        vec3 lo{inf, inf, inf}, hi{-inf, -inf, -inf};

        const auto extend = [&](vec3 p)
        {
            lo = {std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = {std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
        };

        const std::uint32_t count = node.first_and_count & 15u;

        if (count) {
            const std::uint32_t first = node.first_and_count >> 4u;

            for (std::uint32_t k = 0; k < count; ++k)
            {
                const std::size_t at = (std::size_t(first) + k) * 3u;
                const auto &p = vertices.at(at);
                const vec3 a{p[0], p[1], p[2]};

                if (p[3] < 0.f) {
                    const float radius = -p[3];
                    const vec3 extent{radius, radius, radius};
                    extend(a - extent);
                    extend(a + extent);
                }
                else {
                    const auto &e1 = vertices.at(at + 1u);
                    const auto &e2 = vertices.at(at + 2u);
                    extend(a);
                    extend(a + vec3{e1[0], e1[1], e1[2]});
                    extend(a + vec3{e2[0], e2[1], e2[2]});
                }
            }
        }
        else {
            for (std::size_t child = j + 1u; child < node.next;
                 child = nodes[child].next)
            {
                const auto &n = nodes.at(child);
                extend({n.minimum[0], n.minimum[1], n.minimum[2]});
                extend({n.maximum[0], n.maximum[1], n.maximum[2]});
            }
        }

        if (std::isfinite(lo.x)) {
            node.minimum = {lo.x - .0001f, lo.y - .0001f, lo.z - .0001f};
            node.maximum = {hi.x + .0001f, hi.y + .0001f, hi.z + .0001f};

            if (!count) {
                const auto d = hi - lo;
                refit_cost += 2.f * (d.x * d.y + d.y * d.z + d.z * d.x);
            }
        }
    }
    // An initially good tree becomes expensive when skinned objects move
    // across its original spatial partitions. Refit is exact regardless,
    // but periodically replace a badly overlapping tree. A cooldown of
    // 120 frames keeps rebuilding outside the ordinary per-frame cost.
    if (data.dynamic_reference_cost == 0.f) {
        data.dynamic_reference_cost = refit_cost;
    }
    else if (++data.dynamic_refit_age >= 120u &&
             refit_cost > 2.f * data.dynamic_reference_cost &&
             primitive_count > 8u)
    {
        data.dynamic_topology.clear();
        data.dynamic_order.clear();
        data.prepared_nodes.clear();
        data.dynamic_reference_cost = 0.f;
        data.dynamic_refit_age = 0;
        prepare_dynamic_geometry(faces, spheres, face_owners, sphere_owners);

        return;
    }

    data.prepared_bounds = bounds;
    data.prepared_primitives = std::uint32_t(primitive_count);
    data.prepared_has_spheres = !spheres.empty();
    data.dynamic_prepared = true;
    // Every shadow segment lies inside its point-light influence sphere.
    // If no animated primitive bounds touch that sphere, a dynamic shadow
    // hit is mathematically impossible for this light. This mask is
    // conservative and used for main and reflected lighting alike.
    data.dynamic_shadow_mask = {0u, 0u};

    if (primitive_count)
        for (std::size_t i = 0; i < data.live_lights.size() && i < 64u; ++i)
        {
            const auto &light = data.live_lights[i];

            if (!light.enabled || light.strength <= 0.f || light.radius <= 0.f)
                continue;

            const float x = std::max({bounds.minimum.x - light.position.x, 0.f, light.position.x - bounds.maximum.x});
            const float y = std::max({bounds.minimum.y - light.position.y, 0.f, light.position.y - bounds.maximum.y});
            const float z = std::max({bounds.minimum.z - light.position.z, 0.f, light.position.z - bounds.maximum.z});
            const float radius = light.radius + 0.001f;

            if (x * x + y * y + z * z <= radius * radius)
                data.dynamic_shadow_mask[i >> 5u] |= 1u << (i & 31u);
        }
}

void upload_dynamic_geometry(std::uint32_t image)
{
    using namespace vk::util::math;
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto &data = get();

    if (!data.snapshot || !data.dynamic_prepared) return;
    auto &frame = data.frames.at(image);
    const auto &vertices = data.prepared_vertices;
    const auto &nodes = data.prepared_nodes;
    const auto &bounds = data.prepared_bounds;

    const auto primitive_count = data.prepared_primitives;
    frame.dynamic_count = primitive_count;
    frame.dynamic_has_spheres = data.prepared_has_spheres;

    if (vertices.size() != std::size_t(primitive_count) * 3u)
        throw std::logic_error("Dynamic shadow vertices do not match primitive count");

    if (vertices.size() > std::numeric_limits<std::uint32_t>::max() ||
        nodes.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("Dynamic shadow buffer exceeds shader count range");

    const auto storage = vk::BufferUsageFlagBits::eStorageBuffer;
    const auto host = vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent;

    const auto ensure = [&](::vs::gpu::owned_buffer &buffer, std::size_t bytes)
    {
        if (buffer.size >= bytes) return false;
        buffer = ::vs::gpu::create_buffer(
            std::max(bytes, std::size_t(buffer.size) * 2u), storage, host);
        return true;
    };

    const auto vertex_bytes = vertices.size() * sizeof(std::array<float, 4>);
    const bool vertices_moved = ensure(frame.dynamic_vertices,
                                       std::max<std::size_t>(vertex_bytes, 48));
    if (vertex_bytes)
        vk::buffer::write(frame.dynamic_vertices, vertices.data(), vertex_bytes);

    const auto node_bytes = 16u + nodes.size() * sizeof(shadow_node);
    const bool nodes_moved = ensure(frame.dynamic_nodes,
                                    std::max<std::size_t>(node_bytes, 16u + sizeof(shadow_node)));
    auto &node_packet = frame.dynamic_packet;
    node_packet.resize(node_bytes);
    const std::array<std::uint32_t, 4> counts{
        std::uint32_t(nodes.size()), std::uint32_t(vertices.size()), 0u, 0u
    };

    std::memcpy(node_packet.data(), counts.data(), sizeof(counts));

    if (!nodes.empty())
        std::memcpy(node_packet.data() + 16, nodes.data(),
                    nodes.size() * sizeof(shadow_node));
    vk::buffer::write(frame.dynamic_nodes, node_packet.data(), node_packet.size());

    if (vertices_moved || nodes_moved) {
        if (vertices_moved) {
            data.cloud_sets->write_buffer(image, 6,
                                          vk::buffer::descriptor(frame.dynamic_vertices));
            data.gi_sets->write_buffer(image, 10,
                                       vk::buffer::descriptor(frame.dynamic_vertices));
            data.lighting_sets->write_buffer(image, 16,
                                             vk::buffer::descriptor(frame.dynamic_vertices));
        }

        if (nodes_moved) {
            data.cloud_sets->write_buffer(image, 7,
                                          vk::buffer::descriptor(frame.dynamic_nodes));
            data.gi_sets->write_buffer(image, 11,
                                       vk::buffer::descriptor(frame.dynamic_nodes));
            data.lighting_sets->write_buffer(image, 17,
                                             vk::buffer::descriptor(frame.dynamic_nodes));
        }

        if (!data.reflections.empty()) {
            const auto reflected = image + std::uint32_t(data.frames.size());

            if (vertices_moved)
                data.lighting_sets->write_buffer(reflected, 16,
                                                 vk::buffer::descriptor(frame.dynamic_vertices));
            if (nodes_moved)
                data.lighting_sets->write_buffer(reflected, 17,
                                                 vk::buffer::descriptor(frame.dynamic_nodes));
        }
    }

    if (frame.dirty) return;

    const bool had = frame.has_dynamic_bounds;
    const auto previous = frame.dynamic_bounds;
    const bool has = primitive_count != 0;

    frame.changed_count = 0;

    if (!had && !has) return;
    // Dynamic direct shadows remain exact every frame. Only diffuse GI
    // is amortized, at most once per four acquired frames per image.
    if (!frame.dirty && had && has && ++frame.gi_interval < 4u)
        return;
    frame.gi_interval = 0;
    frame.dynamic_bounds = bounds;
    frame.has_dynamic_bounds = has;

    const auto component = [](vec3 v, int axis)
    {
        return axis == 0 ? v.x : axis == 1 ? v.y
                                           : v.z;
    };

    const auto intersects = [&](vec3 point, vec3 end, const collision_box &box)
    {
        const vec3 direction = end - point;
        float entry = 0.f, exit = 1.f;

        for (int axis = 0; axis < 3; ++axis)
        {
            const float d = component(direction, axis);
            const float min = component(box.minimum, axis) - .01f;
            const float max = component(box.maximum, axis) + .01f;
            const float p = component(point, axis);

            if (std::abs(d) < 1e-8f) {
                if (p < min || p > max) return false;
            }
            else {
                float a = (min - p) / d, b = (max - p) / d;

                if (a > b) std::swap(a, b);
                entry = std::max(entry, a);
                exit = std::min(exit, b);

                if (entry > exit) return false;
            }
        }

        return true;
    };

    auto &affected = frame.affected_cpu;
    affected.clear();
    affected.reserve(data.sample_positions.size() / 8u + 1u);
    vec3 dirty_min{inf, inf, inf}, dirty_max{-inf, -inf, -inf};

    const auto dirty = [&](vec3 p, float reach)
    {
        const vec3 offset{reach, reach, reach};

        const vec3 lo = p - offset, hi = p + offset;
        dirty_min = {std::min(dirty_min.x, lo.x), std::min(dirty_min.y, lo.y), std::min(dirty_min.z, lo.z)};
        dirty_max = {std::max(dirty_max.x, hi.x), std::max(dirty_max.y, hi.y), std::max(dirty_max.z, hi.z)};
    };

    if (had) {
        dirty(previous.minimum, bounce_reach);
        dirty(previous.maximum, bounce_reach);
    }

    if (has) {
        dirty(bounds.minimum, bounce_reach);
        dirty(bounds.maximum, bounce_reach);
    }

    for (std::uint32_t i = 0; i < data.sample_positions.size(); ++i)
    {
        const vec3 point = data.sample_positions[i];
        bool changed = false;

        for (const auto &light : data.live_lights)
        {
            if (!light.enabled || light.strength <= 0.f || light.radius <= 0.f ||
                length_squared(light.position - point) >= light.radius * light.radius)
                continue;

            if ((had && intersects(point, light.position, previous)) ||
                (has && intersects(point, light.position, bounds)))
            {
                changed = true;

                break;
            }
        }

        if (!changed) continue;
        affected.push_back(i);
        dirty(point, bounce_reach);
    }

    frame.changed_count = std::uint32_t(affected.size());

    const auto point_bytes = affected.size() * sizeof(std::uint32_t);
    const bool points_moved = ensure(frame.changed_points,
                                     std::max<std::size_t>(point_bytes, 4u));
    if (point_bytes)
        vk::buffer::write(frame.changed_points, affected.data(), point_bytes);

    if (points_moved)
        data.cloud_sets->write_buffer(image, 8,
                                      vk::buffer::descriptor(frame.changed_points));

    std::array<std::uint32_t, 3> first{}, last{};
    const std::array<std::uint32_t, 3> extent{
        data.grid_extent.width, data.grid_extent.height, data.grid_extent.depth
    };

    for (int axis = 0; axis < 3; ++axis)
    {
        const double begin = std::floor(
                                 (double(component(dirty_min, axis)) - component(data.grid_origin, axis)) /
                                 data.grid_spacing) -
                             1.0;

        const double end = std::ceil(
                               (double(component(dirty_max, axis)) - component(data.grid_origin, axis)) /
                               data.grid_spacing) +
                           2.0;
        first[axis] = std::uint32_t(std::clamp(begin, 0.0, double(extent[axis])));
        last[axis] = std::uint32_t(std::clamp(end, 0.0, double(extent[axis])));
    }

    if (first[0] >= last[0] || first[1] >= last[1] || first[2] >= last[2])
        return;

    if (!frame.dirty) {
        frame.dirty = true;
        frame.partial_grid = true;
        frame.cloud_partial = true;
        frame.cloud_submitted = false;
        frame.gi_cursor = 0;
        frame.gi_complete = false;
        frame.grid_first = first;
        frame.grid_end = last;
    }
    else if (frame.partial_grid) {
        for (int axis = 0; axis < 3; ++axis)
        {
            frame.grid_first[axis] = std::min(frame.grid_first[axis], first[axis]);
            frame.grid_end[axis] = std::max(frame.grid_end[axis], last[axis]);
        }
    }
}

shadow_geometry opaque_shadows()
{
    const auto &state = get();

    return {state.triangles.handle, state.nodes.handle, state.shadow_light_ranges.handle, state.transmission.handle, state.glass_vertices.handle, state.glass_nodes.handle, state.glass_filters.handle, state.light_shapes.handle};
}

inspection_resources inspect(std::uint32_t image)
{
    const auto &state = get();
    const auto &frame = state.frames.at(image);

    return {frame.grid_view, frame.points.handle, state.nodes.handle, state.grid_extent, {state.grid_origin.x, state.grid_origin.y, state.grid_origin.z, state.grid_spacing}, state.point_count, state.visual_node_count, state.node_count};
}

void update_dynamic_geometry(
    std::uint32_t image, const std::vector<triangle> &faces,
    const std::vector<shadow_sphere> &spheres,
    std::span<const std::uint32_t> face_owners,
    std::span<const std::uint32_t> sphere_owners)
{
    prepare_dynamic_geometry(faces, spheres, face_owners, sphere_owners);
    upload_dynamic_geometry(image);
}

void release()
{
    auto &data = get();
    dynamic_preparation.stop();
    dynamic_preparation_started = false;
    data.dynamic_topology_pending = false;

    const auto device = vk::device::get()->handle;

    for (auto &frame : data.frames)
    {
        if (frame.grid_view) vk::image_view::destroy(frame.grid_view);
        frame.grid_view = vk::ImageView{};
    }

    data.frames.clear();
    data.reflections.clear();
    data.reflection_targets.reset();
    data.frame_graphs.clear();
    data.cloud_program.reset();
    data.gi_program.reset();
    data.lighting_program.reset();
    data.opaque_lighting_program.reset();
    data.tone_program.reset();
    data.present_program.reset();
    data.presentation.reset();

    if (data.present_layout) device.destroyPipelineLayout(data.present_layout);
    data.cloud_sets.reset();
    data.gi_sets.reset();
    data.lighting_sets.reset();
    data.tone_sets.reset();
    data.present_sets.reset();
    data.targets.reset();
    data.lights.reset();
    data.light_shapes.reset();
    data.triangles.reset();
    data.nodes.reset();
    data.shadow_light_ranges.reset();
    data.transmission.reset();
    data.glass_vertices.reset();
    data.glass_nodes.reset();
    data.glass_filters.reset();
    data.neighborhood_offsets.reset();
    data.neighborhood_indices.reset();

    if (data.sampler) device.destroySampler(data.sampler);

    data = {};
}

vk::ImageView ldr_view(std::uint32_t image)
{
    return get().frames.at(image).ldr.view;
}

vk::ImageView hdr_view(std::uint32_t image)
{
    return get().frames.at(image).hdr.view;
}

vk::ImageView probe_view(std::uint32_t image)
{
    return get().frames.at(image).grid_view;
}

std::array<float, 4> probe_space()
{
    const auto &g = get();

    return {g.grid_origin.x, g.grid_origin.y, g.grid_origin.z, g.grid_spacing};
}

vk::RenderPass presentation_pass()
{
    return get().presentation->pass();
}

targets images(std::uint32_t image)
{
    const auto &frame = get().frames.at(image);

    return {frame.grid.handle, frame.points.handle, frame.hdr.image, frame.ldr.image};
}

void cloud(vk::CommandBuffer command, std::uint32_t image)
{
    get().region_shadows.upload(image);

    if (!get().frames.at(image).dirty ||
        get().frames.at(image).cloud_submitted ||
        !get().point_count) return;
    const auto &frame = get().frames.at(image);

    const auto count = frame.cloud_partial ? frame.changed_count : get().point_count;
    if (!count) return;
    const cloud_push push{count, get().vertex_count, get().light_count, get().node_count, frame.cloud_partial ? 1u : 0u};
    get().cloud_program->dispatch(command, (count + 63) / 64, 1, 1,
                                  {get().frames.at(image).cloud}, &push, sizeof(push));
}

void gi(vk::CommandBuffer command, std::uint32_t image)
{
    get().region_shadows.upload(image);
    auto &data = get();
    auto &frame = data.frames.at(image);

    if (!frame.dirty || frame.gi_complete) return;
    const std::array<std::uint32_t, 3> begin = frame.partial_grid ? frame.grid_first : std::array<std::uint32_t, 3>{};
    const std::array<std::uint32_t, 3> end = frame.partial_grid ? frame.grid_end : std::array<std::uint32_t, 3>{data.grid_extent.width, data.grid_extent.height, data.grid_extent.depth};

    if (end[0] <= begin[0] || end[1] <= begin[1] || end[2] <= begin[2]) {
        frame.gi_complete = true;

        return;
    }

    const std::array<std::uint32_t, 3> groups{
        (end[0] - begin[0] + 3u) / 4u,
        (end[1] - begin[1] + 3u) / 4u,
        (end[2] - begin[2] + 3u) / 4u
    };

    const auto dispatch = [&](std::uint32_t x, std::uint32_t y,
                              std::uint32_t z, std::uint32_t count_x,
                              std::uint32_t count_y, std::uint32_t count_z,
                              std::uint32_t clear = 0u)
    {
        const grid_push push{
            data.point_count, data.vertex_count, data.node_count, data.light_count, {data.grid_origin.x, data.grid_origin.y, data.grid_origin.z, data.grid_spacing}, {begin[0] + x * 4u, begin[1] + y * 4u, begin[2] + z * 4u, clear}, data.glass_node_count
        };

        data.gi_program->dispatch(command, count_x, count_y, count_z,
                                  {frame.gi}, &push, sizeof(push));
    };

    if (!frame.grid_initialized) {
        const grid_push push{
            data.point_count, data.vertex_count, data.node_count, data.light_count, {data.grid_origin.x, data.grid_origin.y, data.grid_origin.z, data.grid_spacing}, {0u, 0u, 0u, 1u}, data.glass_node_count
        };
        // Zero new 3D textures cheaply; full ray-traced GI is then
        // populated incrementally without one large first-frame dispatch.
        data.gi_program->dispatch(command,
                                  (data.grid_extent.width + 3u) / 4u,
                                  (data.grid_extent.height + 3u) / 4u,
                                  (data.grid_extent.depth + 3u) / 4u,
                                  {frame.gi}, &push, sizeof(push));
        vk::MemoryBarrier barrier{
            vk::AccessFlagBits::eShaderWrite,
            vk::AccessFlagBits::eShaderRead |
                vk::AccessFlagBits::eShaderWrite
        };

        command.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eComputeShader,
            {}, {barrier}, {}, {});
        frame.grid_initialized = true;
    }
    // Keep animated GI inexpensive while allowing initial scene lighting
    // to converge promptly. Both paths have fixed per-frame work bounds.
    // Moving crowds invalidate many probe neighborhoods at once.
    // Distribute the expensive refresh over additional frames without
    // changing the probe grid, GI weights, or final converged result.
    const bool dense_dynamic = frame.dynamic_count >= 900u;

    const std::uint32_t max_groups_per_frame =
        dense_dynamic ? (frame.partial_grid ? 1u : 4u) : (frame.partial_grid ? 4u : 12u);
    const std::uint32_t total = groups[0] * groups[1] * groups[2];
    std::uint32_t remaining = max_groups_per_frame;

    while (frame.gi_cursor < total && remaining)
    {
        const auto index = frame.gi_cursor;
        const auto x = index % groups[0];
        const auto y = (index / groups[0]) % groups[1];
        const auto z = index / (groups[0] * groups[1]);
        const auto count = std::min(remaining, groups[0] - x);
        dispatch(x, y, z, count, 1, 1);
        frame.gi_cursor += count;
        remaining -= count;
    }

    frame.gi_complete = frame.gi_cursor >= total;
}

void lighting(vk::CommandBuffer command, std::uint32_t image,
              const camera_frame &frame)
{
    // Fence-completed image only: no shared in-flight buffer mutation.
    get().region_shadows.upload(image);
    auto &data = get();
    auto &current = data.frames.at(image);
    // Conservatively rasterize each light's projected influence sphere into
    // screen tiles. Rays that cannot intersect the sphere cannot contribute.
    // The remaining light IDs retain the original server-defined order.
    std::fill(current.tile_cpu.begin(), current.tile_cpu.end(), 0u);

    if (data.snapshot) {
        using namespace vk::util::math;

        const auto extent = get().output.extent;
        const auto basis = ::vs::scene::axes(frame.yaw, frame.pitch);
        const auto &forward = basis.forward;
        const auto &right = basis.right;
        const auto &up = basis.up;

        const float tan_v = std::tan(frame.vertical_fov * 0.5f);
        const float tan_h = tan_v * float(extent.width) / float(extent.height);
        const auto projected = [](float value, float z, float radius, float tangent)
        {
            const float a = (value - radius) / (z - radius);
            const float b = (value - radius) / (z + radius);
            const float c = (value + radius) / (z - radius);
            const float d = (value + radius) / (z + radius);

            return std::pair{std::min(a, b) / tangent, std::max(c, d) / tangent};
        };

        for (std::uint32_t i = 0; i < data.snapshot->lights.size(); ++i)
        {
            const auto &light = data.snapshot->lights[i];

            if (!light.enabled || light.radius <= 0.f ||
                light.strength <= 0.f) continue;
            const vec3 delta = light.position - frame.position;
            const float z = dot(delta, forward), r = light.radius;
            // A sphere fully behind the camera plane cannot affect any
            // rasterized world or first-person receiver. The old fallback
            // assigned these lights to *every* tile, wasting shadow rays.
            if (z + r <= 0.f) continue;
            std::uint32_t x0 = 0, x1 = data.tile_columns;
            std::uint32_t y0 = 0, y1 = data.tile_rows;

            if (z > r && tan_v > 0.f && tan_h > 0.f) {
                const auto [left, right_edge] = projected(dot(delta, right), z, r, tan_h);
                const auto [bottom, top] = projected(dot(delta, up), z, r, tan_v);
                const auto lower = [](float ndc, std::uint32_t size,
                                      std::uint32_t count)
                {
                    return std::uint32_t(std::clamp(
                        int(std::floor((ndc * 0.5f + 0.5f) * float(size) / tile_size)) - 1,
                        0, int(count)));
                };

                const auto upper = [](float ndc, std::uint32_t size,
                                      std::uint32_t count)
                {
                    return std::uint32_t(std::clamp(
                        int(std::ceil((ndc * 0.5f + 0.5f) * float(size) / tile_size)) + 1,
                        0, int(count)));
                };

                x0 = lower(left, extent.width, data.tile_columns);
                x1 = upper(right_edge, extent.width, data.tile_columns);
                y0 = lower(-top, extent.height, data.tile_rows);
                y1 = upper(-bottom, extent.height, data.tile_rows);
            }

            for (std::uint32_t y = y0; y < y1; ++y)
                for (std::uint32_t x = x0; x < x1; ++x)
                {
                    const auto base = (std::size_t(y) * data.tile_columns + x) * tile_mask_words;
                    current.tile_cpu[base + (i >> 5u)] |= 1u << (i & 31u);
                }
        }
    }

    vk::buffer::write(current.tile_lights, current.tile_cpu.data(),
                      current.tile_cpu.size() * sizeof(std::uint32_t));
    // Static GLB geometry also dominates shadow-BVH traversal. The old
    // heuristic considered only animated primitives and therefore never
    // enabled the accelerated shader for the 87k-triangle mermaid scene.
    const std::size_t dynamic_primitives = current.dynamic_count;
    const std::size_t static_primitives = data.vertex_count / 3u;
    const std::uint32_t adaptive_rate =
        static_primitives >= 60000u || dynamic_primitives >= 900u ? 4u : static_primitives >= 12000u || dynamic_primitives >= 256u ? 2u
                                                                                                                                   : 1u;
    // An explicit A/B override isolates the edge-reconstruction and
    // shadow-ray costs without changing normal adaptive behavior.
    static const std::uint32_t forced_rate = []
    {
        const char *value = std::getenv("VS_LIGHTING_RATE");
        if (!value) return 0u;
        if (std::strcmp(value, "1") == 0) return 1u;
        if (std::strcmp(value, "2") == 0) return 2u;
        if (std::strcmp(value, "4") == 0) return 4u;

        return 0u;
    }();

    const std::uint32_t pixel_rate =
        forced_rate ? forced_rate : adaptive_rate;
    data.lighting_pixel_rate = pixel_rate;
    static std::uint32_t previous_rate = 0u;

    if (pixel_rate != previous_rate) {
        std::cout << "pc: lighting pixel rate " << pixel_rate
                  << "x (" << (forced_rate ? "forced" : "adaptive")
                  << "), static triangles=" << static_primitives
                  << ", dynamic primitives=" << dynamic_primitives << '\n';
        previous_rate = pixel_rate;
    }

    const bool opaque = data.opaque_lighting_program && pixel_rate == 4u &&
                        data.glass_node_count == 0u && data.region_shadows.size() == 0u &&
                        !current.dynamic_has_spheres;
    const auto &program = opaque ? *data.opaque_lighting_program : *data.lighting_program;

    if (data.opaque_lighting_program) {
        static std::optional<bool> previous;

        if (previous != opaque) {
            std::cout << "pc: opaque lighting specialization "
                      << (opaque ? "active" : "inactive") << '\n';
            previous = opaque;
        }
    }

    compute(command, program, current.lighting, frame, pixel_rate);
}

void tonemap(vk::CommandBuffer command, std::uint32_t image,
             const camera_frame &frame)
{
    const auto &current = get().frames.at(image);
    compute(command, *get().tone_program, current.tone, frame, 1u,
            tone_workgroup_size);
}

void present(vk::CommandBuffer command, std::uint32_t image,
             const overlay_callback &overlay)
{
    const auto extent = get().output.extent;

    vk::ClearValue clear{};
    clear.color = vk::ClearColorValue(std::array<float, 4>{0, 0, 0, 1});
    vk::RenderPassBeginInfo info{};
    info.renderPass = get().presentation->pass();
    info.framebuffer = get().presentation->framebuffer(image);
    info.renderArea = vk::Rect2D{{0, 0}, extent};
    info.clearValueCount = 1;
    info.pClearValues = &clear;

    ::vs::render::renderer graphics(command);
    graphics.begin_pass(info);
    graphics.viewport(
        vk::Viewport{0, 0, float(extent.width), float(extent.height), 0, 1},
        vk::Rect2D{{0, 0}, extent});
    ::vs::render::mesh triangle{};
    triangle.vertex_count = 3;
    ::vs::render::material surface{};
    surface.pipeline = get().present_program->handle();
    surface.layout = get().present_layout;
    surface.sets = {get().frames.at(image).presentation};
    graphics.draw(triangle, surface);

    if (overlay) overlay(command, extent);
    graphics.end_pass();
}

void record(vk::CommandBuffer command, std::uint32_t image,
            const camera_frame &camera, const overlay_callback &overlay)
{
    auto &state = get();

    if (!command || image >= state.frame_graphs.size())
        throw std::invalid_argument("::vs::pcloud::record invalid command or image index");

    const auto resources = images(image);

    const vk::ImageSubresourceRange color{
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1
    };

    const auto compute_stage = vk::PipelineStageFlagBits::eComputeShader;
    const auto shader_read = vk::AccessFlagBits::eShaderRead;
    const auto shader_write = vk::AccessFlagBits::eShaderWrite;

    auto &graph = state.frame_graphs[image];
    graph.clear_passes();
    try
    {
        graph.add("pc.cloud", {::vs::gpu::access::for_buffer(resources.points, compute_stage, shader_write, true)},
                  [image](vk::CommandBuffer cmd)
                  { cloud(cmd, image); });
        graph.add("pc.gi", {::vs::gpu::access::for_buffer(resources.points, compute_stage, shader_read, false), ::vs::gpu::access::for_image(resources.grid, color, vk::ImageLayout::eGeneral, compute_stage, shader_write, true)},
                  [image](vk::CommandBuffer cmd)
                  { gi(cmd, image); });
        graph.add("pc.lighting", {::vs::gpu::access::for_image(resources.grid, color, vk::ImageLayout::eGeneral, compute_stage, shader_read, false), ::vs::gpu::access::for_image(resources.hdr, color, vk::ImageLayout::eGeneral, compute_stage, shader_write, true)},
                  [image, camera](vk::CommandBuffer cmd)
                  {
                      lighting(cmd, image, camera);
                  });
        graph.add("pc.tonemap", {::vs::gpu::access::for_image(resources.hdr, color, vk::ImageLayout::eGeneral, compute_stage, shader_read, false), ::vs::gpu::access::for_image(resources.ldr, color, vk::ImageLayout::eGeneral, compute_stage, shader_write, true)},
                  [image, camera](vk::CommandBuffer cmd)
                  {
                      tonemap(cmd, image, camera);
                  });
        graph.add("pc.present", {::vs::gpu::access::for_image(resources.ldr, color, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits::eFragmentShader, shader_read, false)},
                  [image, &overlay](vk::CommandBuffer cmd)
                  {
                      present(cmd, image, overlay);
                  });
        graph.execute(command);
    }

    catch (...)
    {
        graph.clear_passes();
        graph.clear_states();
        throw;
    }

    graph.clear_passes();
}
} // namespace vs::pcloud
