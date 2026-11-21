#pragma once
// Single public entry point. All implementations are owned by vs/.
#include <vs/app/app.hpp>
#include <vs/assets/animation.hpp>
#include <vs/assets/gpu_assets.hpp>
#include <vs/assets/model.hpp>
#include <vs/bvh/gpu_regions.hpp>
#include <vs/bvh/regions.hpp>
#include <vs/bvh/stackless.hpp>
#include <vs/editor/editor.hpp>
#include <vs/editor/vulkan.hpp>
#include <vs/editor/widgets.hpp>
#include <vs/gpu/gpu.hpp>
#include <vs/gpu/residency.hpp>
#include <vs/jobs/pool.hpp>
#include <vs/lighting/transmission.hpp>
#include <vs/network/environment.hpp>
#include <vs/network/network.hpp>
#include <vs/network/region.hpp>
#include <vs/optics/layered.hpp>
#include <vs/pcloud/pcloud.hpp>
#include <vs/pcloud/region_adapter.hpp>
#include <vs/render/frame.hpp>
#include <vs/render/invalidation.hpp>
#include <vs/render/render.hpp>
#include <vs/scene/environment.hpp>
#include <vs/scene/region_runtime.hpp>
#include <vs/scene/region_stream.hpp>
#include <vs/scene/regions.hpp>
#include <vs/scene/scene.hpp>
#include <vs/spatial/index.hpp>
#include <vs/spatial/interest.hpp>
#include <vs/spatial/paged_index.hpp>
#include <vs/spatial/region_scheduler.hpp>
#include <vs/spatial/regions.hpp>
#include <vs/spatial/world.hpp>
#include <vs/visual/inspector.hpp>
namespace vs
{
using device = gpu::device;
using buffer = gpu::buffer;
using compute = gpu::compute;
using render_graph = gpu::graph;
using frame_scheduler = render::frame_scheduler;
using optics_compositor = optics::layered::compositor;
using editor_session = editor::session;
} // namespace vs
