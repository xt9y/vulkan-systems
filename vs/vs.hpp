#pragma once
// Single public entry point. All implementations are owned by vs/.
#include <vs/scene/scene.hpp>
#include <vs/assets/model.hpp>
#include <vs/assets/animation.hpp>
#include <vs/assets/gpu_assets.hpp>
#include <vs/app/app.hpp>
#include <vs/gpu/gpu.hpp>
#include <vs/render/render.hpp>
#include <vs/render/frame.hpp>
#include <vs/bvh/bvh.hpp>
#include <vs/bvh/hierarchy.hpp>
#include <vs/lumen/lumen.hpp>
#include <vs/lighting/transmission.hpp>
#include <vs/lighting/caustics.hpp>
#include <vs/pcloud/pcloud.hpp>
#include <vs/pmap/pmap.hpp>
#include <vs/optics/optics.hpp>
#include <vs/optics/layered.hpp>
#include <vs/spatial/index.hpp>
#include <vs/jobs/pool.hpp>
#include <vs/network/network.hpp>
#include <vs/editor/editor.hpp>
#include <vs/editor/widgets.hpp>
#include <vs/visual/inspector.hpp>
#include <vs/editor/vulkan.hpp>
namespace vs {
using device=gpu::device;
using buffer=gpu::buffer;
using compute=gpu::compute;
using render_graph=gpu::graph;
using frame_scheduler=render::frame_scheduler;
using photon_mapper=pmap::mapper;
using lumen_renderer=lumen::system;
using optics_compositor=optics::compositor;
using editor_session=editor::session;
}
