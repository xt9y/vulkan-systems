#include <cbuild.h>

void build(C_Build *b)
{
    C_Target *blood = c_static_library(b, "VulkanBlood");
    c_sources(blood, "bld/core/*.cpp");
    c_sources(blood, "bld/math/*.cpp");
    c_sources(blood, "bld/simulation/*.cpp");
    c_sources(blood, "bld/surfaces/*.cpp");
    c_sources(blood, "bld/network/*.cpp");
    c_sources(blood, "bld/render/*.cpp");
    c_include(blood, ".");
    c_flag(blood, "-std=c++17");

    C_Dependency *engine = c_git(
        b, "VulkanEngine", "https://github.com/xt9y/vulkan-engine.git", "main");
    c_dep_cbuild(engine, "VulkanEngine", C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine, ".");
    c_use(blood, engine);

    C_Dependency *headers = c_git(
        b, "VulkanHeaders", "https://github.com/KhronosGroup/Vulkan-Headers.git", "main");
    c_dep_header_only(headers);
    c_dep_include(headers, "include");
    c_use(blood, headers);

    c_default_target(b, blood);
}
