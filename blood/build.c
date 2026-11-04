#include <cbuild.h>

void build(C_Build *b)
{
    C_Target *blood = c_static_library(b, "VulkanBlood");
    c_sources(blood, "bld/core/*.cpp");
    c_sources(blood, "bld/material/*.cpp");
    c_sources(blood, "bld/surfaces/*.cpp");
    c_sources(blood, "bld/wounds/*.cpp");
    c_sources(blood, "bld/network/*.cpp");
    c_sources(blood, "bld/render/*.cpp");
    c_include(blood, ".");
    c_flag(blood, "-std=c++17");

    // Blood is a MATERIAL and impact/surface system, not another solver.
    // Use a named release ref: c_git only fetches selected refs, so historical
    // commit hashes may not be present in its shallow dependency checkout.
    C_Dependency *fluid = c_git(
        b, "VulkanFluid", "https://github.com/xt9y/vulkan-systems.git",
        "vulkan-fluids-v4");
    c_dep_subdir(fluid, "fluid");
    c_dep_cbuild(fluid, "VulkanFluid", C_TARGET_STATIC_LIBRARY);
    c_dep_include(fluid, ".");
    c_use(blood, fluid);

    C_Dependency *engine = c_git(
        b, "VulkanEngine", "https://github.com/xt9y/vulkan-engine.git",
        "2aa9b9c265995c76859952c6f32b8a75873d22f3");
    c_dep_cbuild(engine, "VulkanEngine", C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine, ".");
    c_use(blood, engine);

    C_Dependency *headers = c_git(
        b, "VulkanHeaders", "https://github.com/KhronosGroup/Vulkan-Headers.git",
        "9fbc2a782ee3f8a4db64e7d991e23da75e893f44");
    c_dep_header_only(headers);
    c_dep_include(headers, "include");
    c_use(blood, headers);

    c_default_target(b, blood);
}
