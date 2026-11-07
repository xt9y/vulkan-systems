#include <cbuild.h>

void build(C_Build* b)
{
    C_Target* target=c_static_library(b,"VulkanLumen");
    c_sources(target,"lumen/*.cpp");
    c_include(target,".");
    c_include(target,"../gpu");
    c_include(target,"../render");
    c_flag(target,"-std=c++17");

    C_Dependency* engine=c_git(
        b,"VulkanEngine","https://github.com/xt9y/vulkan-engine.git",
        "4b792b574727115d7478f414e11e9540e848404e");
    c_dep_cbuild(engine,"VulkanEngine",C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine,".");
    c_use(target,engine);
    c_default_target(b,target);
}
