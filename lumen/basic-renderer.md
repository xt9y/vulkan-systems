# VulkanLumen — basic bounded renderer scope

This module is a **Lumen-style, software-BVH static-scene renderer**.
It is not Epic's production Lumen implementation and does not yet support
every scene type or advanced rendering feature.

## Stable baseline

- VulkanSystems: `lumen-baseline-perfect-20261108` at `32d0c85e1b63cf55bc2a984f6119e4cfaaeda443`.
- VulkanExperiment: `lumen-baseline-perfect-20261108` at `1c61c1a2aa0d669a9a30c2f65f3204d678489320`.
- Those branches retain the user-reported 60 FPS Cornell renderer.
- The bounded-scene implementation was developed on
  `lumen-basic-renderer-development`; do not call it performance-verified
  until the native M2 harness has completed.

## Implemented pipeline

1. **Static geometry:** host quads (2–256 planar cards) or a triangle mesh
   (1–255 triangles plus an area-light quad), validated/copied at creation.
   CPU BVH maps two triangle primitives per card.
2. **Adaptive atlas:** 4×4, 8×8 or 16×16 card packing; 64×64 material and
   128×128 lit/direct cards; invalid slots are cleared.
3. **Lighting:** 1–4 rectangular emitters at the end of the card array,
   direct-area shadows, Schlick/Smith/GGX direct specular and optional
   environment radiance for unoccluded rays.
4. **Indirect radiance:** 12×8×12 SH1 world-probe cache, static-scene
   bounds from the supplied geometry, amortized card relighting (1–8
   updated faces/frame), and software-BVH ray visibility.
5. **Screen gather:** 8-pixel probe spacing, 8×8 directional bins per
   screen probe, screen tracing with verified BVH fallback, temporal
   sample reuse, and geometry-aware 3×3 spatial filtering.
6. **Reflections:** rough specular reuses screen radiance; surfaces below
   roughness 0.4 trace one additional reflection ray per screen probe
   and blend with the rough result. Rect-light direct specular remains
   analytical, so emitter hits are not double-counted by reflection rays.
7. **Presentation/output:** HDR internal scene rasterization, composite,
   optional full-internal-resolution diffuse/reflection exports, and
   borrowed Vulkan image handles/views with atlas and card diagnostics.

## Host interface

```cpp
lumen::settings settings{};
settings.demo = lumen::demo_scene::none;
settings.surface_updates_per_frame = 2;
settings.environment_radiance = {0.0f, 0.0f, 0.0f};
settings.full_resolution_outputs = true;

auto scene = lumen::cornell_quads();
scene.faces[5].roughness = 0.2f;
renderer.create(pass, format, extent, settings, scene);

// Later, after waiting for the Vulkan device to become idle:
// renderer.replace_quads(scene);
```

For triangle meshes, populate `lumen::triangle_scene::triangles` and its
`emitter` quad and pass that to the corresponding `create()` overload.
Face normals are flat and derived from triangle winding. Use positive
emission only on the final 1–4 quad cards, or on the last emitter in a
`triangle_scene`. All other surfaces must be nonemissive.

For a scene containing `N` cards, the material atlas is
`columns×64` square and the lighting atlases are `columns×128` square.
The returned `outputs` report card count, emitter count and atlas extents.

## Remaining limitations — not represented as finished

- The `set_scene(scene)` raw GPU mesh-buffer interface and external
  `set_gbuffer` path are not implemented. The supported geometry path
  is the CPU static scene creation/rebuild API described above.
- This is bounded to 256 cards; it does not generate a compact multi-view
  card set for arbitrary high-polycount meshes. Triangles each consume
  one card, which is intentionally simple and memory-expensive.
- Static BVH rebuilds and shader specialization happen at create/replace.
  There is no per-frame skinned geometry, BLAS/TLAS refit, GPU-driven
  instance streaming or incremental material/scene invalidation.
- World probes have a fixed 12×8×12 topology. Custom-scene probe occupancy
  placement is conservative rather than a full geometry-aware solver.
- Rectangular emitter lighting uses cached visibility; the penumbrae can
  remain visibly quantized. There are no shadow maps or dedicated
  high-resolution spatially adaptive penumbra estimators.
- Smooth reflections use one ray per **screen probe**, not per pixel.
  The result is a low-resolution approximation rather than faithful
  high-frequency, multilayer or recursive mirror reflection.
- No texture/material graph sampling, transparent/translucent transport,
  volumetric GI, distance-field tracing or hardware ray-tracing backend.
- No native MoltenVK build or GPU measurement was executed from this chat.
  Shadow, reflection quality, and frame times require validation on M2.

## Regression validation

In `vulkan-experiment`, `LUMEN_SCENE=quads` is the default.
`triangles`, `two-lights`, `sky` and `smooth` are opt-in alternatives.
Run `c build run vulkan-experiment` after
`git submodule update --init --recursive`.

Check shader compilation, descriptor creation, first-frame output metadata,
card initialization, 60 FPS steady-state GPU cost, and visual GI/reflection
quality. The optional full-resolution export and smooth-reflection paths
add GPU work when enabled.
