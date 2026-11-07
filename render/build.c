#include <cbuild.h>

void build(C_Build* b)
{
    C_Target* render=c_static_library(b,"VulkanRender");
    c_sources(render,"render/*.cpp");
    c_include(render,".");
    // VulkanRender reuses VulkanGPU headers, including gpu/image.hpp and gpu/graph.hpp.
    // The render subproject is built independently, so expose its sibling include root.
    c_include(render,"../gpu");
    c_flag(render,"-std=c++17");

    C_Dependency* engine=c_git(
        b,"VulkanEngine","https://github.com/xt9y/vulkan-engine.git",
        "2aa9b9c265995c76859952c6f32b8a75873d22f3");
    c_dep_cbuild(engine,"VulkanEngine",C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine,".");
    c_use(render,engine);
    c_default_target(b,render);
}
