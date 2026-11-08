#include <cbuild.h>
void build(C_Build* b) {
    C_Target* target=c_static_library(b,"VulkanSpatial");
    c_sources(target,"spatial/*.cpp");
    c_include(target,".");
    c_flag(target,"-std=c++17");
    C_Dependency* engine=c_git(b,"VulkanWrapper","https://github.com/xt9y/vulkan-wrapper.git","0d4e7e2ff59516ad9225015dffb22c5f487c485c");
    c_dep_cbuild(engine,"VulkanWrapper",C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine,".");
    c_use(target,engine);
    c_default_target(b,target);
}
