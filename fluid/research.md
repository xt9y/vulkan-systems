# VulkanFluid: research-driven rebuild (2026-11-04)

Status: architecture and experimental plan, NOT a completed simulation.
The legacy blood + CPU SPH + stamp/surface-film stack is archived at
`archive/legacy-fluid-blood-v5-20261104`. Preserve `fluid/build.c`, the
`fld/` namespace, the VulkanEngine wrapper, and the c build static-library
approach. Do not reuse the old physical solver by renaming it.

## Objective

A reusable, material-independent, GPU-resident fluid renderer/simulator
optimised first for physically credible sub-millimetre through centimetre
drops, merging, surface adhesion, contact-line pinning, tiny rivulets,
thin streams, and smooth photorealistic render output. Larger connected
fluid bodies are a later scale regime, not the design anchor.

60 rendered FPS is an acceptance target on the M2 Air in explicitly
bounded scenes, never a claim of guaranteed throughput for unlimited
particle counts. Validate frame-time percentiles and numerical error
independently. The simulation and render rates are decoupled.

## Literature, prioritized

1. Timo Probst, Matthias Teschner, "Unified Pressure, Surface Tension
   and Friction for SPH Fluids", ACM TOG 44(1), Dec 2024.
   https://doi.org/10.1145/3708034
   Central small-drop solver design. Pressure, capillary energy and
   fluid-boundary Coulomb friction share one implicit solve. Test
   Laplace pressure (2 sigma / radius), contact angle, pinning on
   vertical/sloped solids, merging drops, wet trails. Beware: exact
   implementation and paper parameters may be expensive on M2.

2. Jan Bender, Dan Koschier, "Divergence-Free Smoothed Particle
   Hydrodynamics", SCA 2015.
   https://animation.rwth-aachen.de/publication/054/
   Two complementary pressure projections reduce density compression
   and velocity divergence. Compare pressure accuracy and convergence
   against GPU PBF before selecting the main incompressibility core.

3. Miles Macklin, Matthias Mueller, "Position Based Fluids", 2013.
   https://blog.mmacklin.com/project/pbf/
   GPU-friendly fallback/baseline, not by itself a plausible droplet
   adhesion or surface-tension model. Benchmark convergence/volume
   drift and identify missing force coupling.

4. Nadir Akinci, Gizem Akinci, Matthias Teschner, "Versatile Surface
   Tension and Adhesion for SPH Fluids", SIGGRAPH Asia 2013.
   https://cg.informatik.uni-freiburg.de/publications/2013_SIGGRAPHASIA_surfaceTensionAdhesion.pdf
   Symmetric liquid-liquid and liquid-solid forces, surface-area
   minimization and droplet cohesion. A baseline to compare with the
   newer unified solver; do not assume simple pairwise attraction
   always yields correct Laplace pressure across resolutions.

5. Orestis Vantzos et al., "Functional Thin Films on Surfaces",
   IEEE TVCG 2017.
   https://pubmed.ncbi.nlm.nih.gov/28113979/
   A mass-conserving mesh-surface PDE with a sparse implicit solve.
   Introduce it only if measured cases prove particle-only films
   inadequate. Do not treat UV charts as physical flow boundaries.

6. "Probing surface wetting across multiple force, length and time
   scales", Communications Physics, 2023.
   https://www.nature.com/articles/s42005-023-01268-z
   Advancing/receding contact angles and hysteresis control pinning.
   Static contact angle alone is not enough for a resting tiny drop.

7. Jihun Yu, Greg Turk, "Reconstructing Surfaces of Particle-Based
   Fluids Using Anisotropic Kernels", 2010.
   https://diglib.eg.org/items/2d966af2-5428-41ca-b90c-2cf76c1f4b53
   Local covariance-driven kernel shapes preserve filaments, thin
   sheets and disconnected drops without blobby sphere artifacts.

8. Wladimir van der Laan, Simon Green, Miguel Sainz, "Screen Space
   Fluid Rendering with Curvature Flow", I3D 2009.
   https://doi.org/10.1145/1507149.1507164
   Pixel-local rendering with smoothed depth and visual LOD. Combine
   with anisotropic surface reconstruction and guard discontinuities
   to avoid oversmoothing isolated tiny drops.

9. Yanrui Xu et al., "Anisotropic Screen Space Rendering for
   Particle-Based Fluid Simulation", Computers & Graphics 2023.
   https://www.sciencedirect.com/science/article/pii/S0097849322002308
   Stretch point sprites using local PCA and combine with
   discontinuity-aware screen-space surface filtering.

10. Florian Ferstl et al., "Narrow Band FLIP for Liquid Simulations",
    Computer Graphics Forum, 2016.
    https://www.cs.cit.tum.de/en/cg/research/publications/2016/narrow-band-flip-for-liquid-simulations/
    Later optional bulk-water path; particles near free surface and
    cheaper interior grid. Do NOT replace the small-drop solver with
    FLIP indiscriminately: sparse interfaces and capillary physics are
    more important for our primary target.

11. Chenfanfu Jiang et al., "The Affine Particle-In-Cell Method",
    ACM TOG, 2015.
    https://doi.org/10.1145/2766996
    Grid particle transport preserves affine velocity modes with
    lower dissipation than PIC, less noise than FLIP. Future bulk
    liquids only; independently benchmark against SPH.

12. "Optimization of flexible neighbors lists in Smoothed Particle
    Hydrodynamics on GPU", Advances in Engineering Software, 2024.
    https://www.sciencedirect.com/science/article/pii/S0965997824001182
    Memory layout and neighbor traversal can dominate kernel cost.

13. "Moving Towards Large-Scale Particle Based Fluid Simulation in
    Unity 3D", Applied Sciences, 2025.
    https://www.mdpi.com/2076-3417/15/17/9706
    GPU SoA, counting sort, exclusive scan and spatial hashing; reported
    benchmarks are not portable to Apple silicon without measurement.

14. Apple GPU tile-based deferred rendering optimization:
    https://developer.apple.com/documentation/metal/tailor-your-apps-for-apple-gpus-and-tile-based-deferred-rendering
    Avoid needless off-chip traffic; profile real MoltenVK/Vulkan memory
    usage and render pass scheduling instead of asserting theoretical
    Metal-specific optimizations are available through Vulkan.

## Physical regimes and mandatory behavior

- Capillary length ell_c=sqrt(sigma/(rho*g)) is roughly 2.7 mm for water
  in Earth gravity. Below this scale, surface tension can dominate the
  droplet silhouette; do not treat all droplets as ballistic spheres.
- Young-Laplace pressure for a spherical droplet: DeltaP=2*sigma/R.
  Validate simulated mean interior pressure at several resolutions.
- Bond number Bo=rho*g*L^2/sigma, Weber number We=rho*U^2*L/sigma,
  Ohnesorge Oh=mu/sqrt(rho*sigma*L). Select appropriate numerical
  treatment by regime, not just visual particle count.
- Wetting must have advancing AND receding contact angles, hysteresis,
  boundary friction, and a stick/slip transition, not a single
  user-defined dampening scalar.
- Liquid-solid collision uses the host's SDF/mesh query and conservative
  impulse exchange where two-way rigid coupling is enabled.
- Shear-dependent viscosity and cohesive forces must be coupled to the
  pressure solve for high-quality tiny-droplet behavior.
- Connected thin streams must not collapse into strings of isolated
  kernels. Sub-pixel droplets still carry physical volume and momentum.

## New pipeline (one contiguous GPU simulation, no CPU particle loop)

    materials + emit queue
      -> GPU compaction/free list
      -> cell key + counting histogram
      -> exclusive prefix scan + reorder (SoA)
      -> sorted cell offsets / compact neighbor iterator
      -> density + boundary contact field
      -> unified implicit pressure/tension/friction solve
      -> substep integration / collision / volume correction
      -> fluid surface classification and anisotropy (GPU)
      -> adaptive screen-space depth and thickness accumulation
      -> depth/normal reconstruction (edge-aware)
      -> Fresnel + Beer-Lambert absorption + refraction + specular
      -> scene composition

No per-frame CPU readback of particle positions, neighbor lists or
binning. CPU does only API submission, parameter update, instrumentation
readback with latency, and occasional resource management.

## GPU data and high-performance design

Structure of arrays for position+radius, velocity+phase, mass, density,
pressure, local surface state, neighbor cell key and sorted particle ID.
Persistent frame-safe storage buffers; separate live/free count and
capacity; ring-buffered host uploads; indirect GPU dispatch for active
particle count. Avoid per-particle heap allocations, host sorting and
whole-device synchronization.

For bounded uniform droplet scales start with one GPU integer-cell
histogram + prefix-scan and compact ranges. Measure before introducing
radix sort, sparse hashing, per-cell linked lists or multiresolution.
Fewer data-dependent atomics matter on Apple silicon. Later, adaptive
groups may be introduced with explicit conservative mass/momentum
transfer tests; do not simply change particle radius mid-solve.

DFSPH or unified SPH pressure/tension/friction requires multiple
iterations and barriers. Use GPU timestamp queries per stage and
evaluate occupancy, neighbor count, dispatch overhead, bandwidth and
solver convergence. Choose a fallback iteration budget only with
measured error thresholds and explicit reporting.

## Rendering quality

Two different cases must coexist:
(a) isolated microdrops rendered with analytic or very fine
    anisotropic contours retaining their round meniscus;
(b) nearby/connected particles reconstructed as a continuous optical
    surface. Use a surface classification criterion to avoid smoothing
    a tiny drop into nothing.

Render front depth, back depth/thickness and motion. Use spatial
curvature smoothing with depth discontinuity safeguards, not an
obligatory noisy/TAA path. Reconstruct normals, apply Fresnel reflection,
Beer-Lambert extinction exp(-sigma_a*distance), refraction against
the scene color/depth, and energy-aware specular lighting. Short-lived
caustic approximation is optional and separately budgeted.

The renderer should distinguish actual physics LOD from screen/pixel
quality; fluid dimensions are metric, not derived from UV pixels.
Use VulkanEngine-only API abstractions to support MoltenVK and future
Vulkan/D3D/Metal backends. Do not introduce a CUDA-only runtime.

## Benchmarks and acceptance gates

Scenes must be deterministic with recorded state and seed:
- One 0.25–1.0 mm droplet at rest in air (Laplace pressure, no jitter).
- A 1–3 mm drop on horizontal, 45 degree and near-vertical plates
  with varying advancing/receding angles (pinning threshold).
- Two sub-mm droplets merge (mass, momentum, shape recovery).
- Thin thread 0.5–2 mm in diameter on the verge of breakup
  (cohesion and filament preservation).
- A small continuous stream hits an inclined curved SDF surface
  (contact, rivulet formation, detachment).
- Droplet count scaling: 1K/10K/50K/100K+ with varied dispersion.
- Scene render scaling: 720p/900p/1440p, narrow-screen footprint
  and full-screen liquid, both still and moving cameras.

Log CPU update, frame interval median/p95/p99, GPU per-stage timings,
particles alive, neighbor visits, solver iterations, density error,
volume drift, capillary pressure error, dropped solver steps and
memory bandwidth pressure. Validate on an M2 Air at its actual GPU
frequency/power state and separately on discrete Vulkan.

Acceptance requires functional visual comparison to published
droplet/film experiments, deterministic numerical conservation
benchmarks, and >=60 rendered FPS p99 frame interval <=16.67ms
for an explicitly named bounded test scene. NO universal 60 FPS
guarantee regardless of complexity, liquid scale, resolution or
scene load.

## Implementation milestones

0. Preserve archive; isolate old code in a research branch; do not
   change the consumer game until a replacement builds and passes.
1. GPU SoA buffers, emission/compaction, histogram+prefix scan,
   deterministic microbenchmark and captured GPU timestamps.
2. GPU incompressible pressure solver (PBF baseline, DFSPH comparison).
3. Unified pressure, surface tension and boundary friction to match
   small-drop experiments; implement and test Laplace/contact-angle.
4. Anisotropic GPU surface reconstruction with analytic isolated
   drops, screen-space depth/thickness and physically based shading.
5. GPU collision SDF, contact hysteresis, coalescence and thin streams.
6. Perf profiling and bounded 60 FPS acceptance; only then switch
   the game to the new VulkanFluid and retire legacy code from main.
