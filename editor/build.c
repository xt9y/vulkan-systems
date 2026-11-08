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
        b,"VulkanWrapper","https://github.com/xt9y/vulkan-wrapper.git",
        "0d4e7e2ff59516ad9225015dffb22c5f487c485c");
    c_dep_cbuild(engine,"VulkanWrapper",C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine,".");
    c_use(editor,engine);

    // Optional font decoder. Glyph PNGs are supplied by the host's assets;
    // this static build never downloads or owns game content.
    C_Dependency* stb=c_git(
        b,"stb","https://github.com/nothings/stb.git",
        "d14d46f6b44456eb88beac58cba833a4135d4c89");
    c_dep_header_only(stb);
    c_dep_include(stb,".");
    c_use(editor,stb);
    c_default_target(b,editor);
}
