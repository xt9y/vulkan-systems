#include <cbuild.h>

void build(C_Build* b)
{
    C_Target* transport=c_static_library(b,"VulkanTransport");
    c_sources(transport,"transport/*.cpp");
    c_include(transport,".");
    c_flag(transport,"-std=c++17");

    C_Dependency* engine=c_git(
        b,"VulkanEngine","https://github.com/xt9y/vulkan-engine.git",
        "2aa9b9c265995c76859952c6f32b8a75873d22f3");
    c_dep_cbuild(engine,"VulkanEngine",C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine,".");
    c_use(transport,engine);
    c_default_target(b,transport);
}
