#include <cbuild.h>

void build(C_Build *b)
{
    C_Target *fluid = c_static_library(b, "VulkanFluid");
    c_sources(fluid, "fld/core/*.cpp");
    c_sources(fluid, "fld/math/*.cpp");
    c_sources(fluid, "fld/material/*.cpp");
    c_sources(fluid, "fld/simulation/*.cpp");
    c_sources(fluid, "fld/surface/*.cpp");
    c_sources(fluid, "fld/porous/*.cpp");
    c_sources(fluid, "fld/render/*.cpp");
    c_include(fluid, ".");
    c_flag(fluid, "-std=c++17");

    C_Dependency *engine = c_git(
        b, "VulkanEngine", "https://github.com/xt9y/vulkan-engine.git",
        "2aa9b9c265995c76859952c6f32b8a75873d22f3");
    c_dep_cbuild(engine, "VulkanEngine", C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine, ".");
    c_use(fluid, engine);

    C_Dependency *headers = c_git(
        b, "VulkanHeaders", "https://github.com/KhronosGroup/Vulkan-Headers.git",
        "9fbc2a782ee3f8a4db64e7d991e23da75e893f44");
    c_dep_header_only(headers);
    c_dep_include(headers, "include");
    c_use(fluid, headers);

    c_default_target(b, fluid);
}
