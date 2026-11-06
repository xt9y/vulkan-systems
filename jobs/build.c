#include <cbuild.h>
void build(C_Build* b) {
    C_Target* target=c_static_library(b,"VulkanJobs");
    c_sources(target,"jobs/*.cpp");
    c_include(target,".");
    c_flag(target,"-std=c++17");
#ifndef _WIN32
    c_link_system(target,"pthread");
#endif
    c_default_target(b,target);
}
