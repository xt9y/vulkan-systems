#include <cbuild.h>

void build(C_Build *b)
{
    C_Target *fluid=c_static_library(b,"VulkanFluid");
    c_sources(fluid,"fld/material/*.cpp");
    c_sources(fluid,"fld/continuum/*.cpp");
    c_sources(fluid,"fld/blood/*.cpp");
    // The old PBF particle implementation remains in Git history. The
    // active library uses only continuum volume/pressure state.
    c_include(fluid,"..");
    c_include(fluid,".");
    c_flag(fluid,"-std=c++17");

    C_Dependency *engine=c_git(
        b,"VulkanEngine","https://github.com/xt9y/vulkan-engine.git",
        "2aa9b9c265995c76859952c6f32b8a75873d22f3");
    c_dep_cbuild(engine,"VulkanEngine",C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine,".");
    c_use(fluid,engine);

    c_default_target(b,fluid);
}
