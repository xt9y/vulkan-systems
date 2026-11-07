#include <cbuild.h>

void build(C_Build* b)
{
    // Entire editor is independent of CSG, gameplay, GLFW and renderer pipelines.
    C_Target* editor=c_static_library(b,"VulkanEditor");
    c_sources(editor,"editor/*.cpp");
    c_include(editor,".");
    c_flag(editor,"-std=c++17");

    // Vulkan is needed only for the optional opaque overlay adapter.
    C_Dependency* engine=c_git(
        b,"VulkanEngine","https://github.com/xt9y/vulkan-engine.git",
        "4b792b574727115d7478f414e11e9540e848404e");
    c_dep_cbuild(engine,"VulkanEngine",C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine,".");
    c_use(editor,engine);
    c_default_target(b,editor);
}
