#include <cbuild.h>

// Convenience target for consumers that use both the GPU and render modules.
// The individual module build.c files remain available for standalone use.
void build(C_Build* b)
{
    C_Target* systems = c_static_library(b, "VulkanSystems");
    c_sources(systems, "gpu/gpu/*.cpp");
    c_sources(systems, "render/render/*.cpp");
    c_include(systems, "gpu");
    c_include(systems, "render");
    c_include(systems, ".");
    c_flag(systems, "-std=c++17");

    C_Dependency* engine = c_git(b, "VulkanEngine",
        "https://github.com/xt9y/vulkan-engine.git",
        "2aa9b9c265995c76859952c6f32b8a75873d22f3");
    c_dep_cbuild(engine, "VulkanEngine", C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine, ".");
    c_use(systems, engine);
    c_default_target(b, systems);
}
