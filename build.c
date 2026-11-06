#include <cbuild.h>
void build(C_Build* b) {
    C_Target* systems=c_static_library(b,"VulkanSystems");
    c_sources(systems,"gpu/gpu/*.cpp");
    c_sources(systems,"render/render/*.cpp");
    c_include(systems,"gpu");
    c_include(systems,"render");
    c_include(systems,".");
    c_flag(systems,"-std=c++17");

    C_Dependency* engine=c_git(b,"VulkanEngine","https://github.com/xt9y/vulkan-engine.git","2aa9b9c265995c76859952c6f32b8a75873d22f3");
    c_dep_cbuild(engine,"VulkanEngine",C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine,".");
    c_use(systems,engine);

    C_Target* spatial=c_static_library(b,"VulkanSpatial");
    c_sources(spatial,"spatial/spatial/*.cpp");
    c_include(spatial,"spatial");
    c_flag(spatial,"-std=c++17");
    c_use(spatial,engine);

    C_Target* jobs=c_static_library(b,"VulkanJobs");
    c_sources(jobs,"jobs/jobs/*.cpp");
    c_include(jobs,"jobs");
    c_flag(jobs,"-std=c++17");
#ifndef _WIN32
    c_link_system(jobs,"pthread");
#endif

    C_Target* network=c_static_library(b,"VulkanNetwork");
    c_sources(network,"network/network/*.cpp");
    c_include(network,"network");
    c_flag(network,"-std=c++17");
#ifdef _WIN32
    c_link_system(network,"ws2_32");
#endif
    c_default_target(b,systems);
}
