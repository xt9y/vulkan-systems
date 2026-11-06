#include <cbuild.h>
void build(C_Build* b) {
    C_Target* target=c_static_library(b,"VulkanNetwork");
    c_sources(target,"network/*.cpp");
    c_include(target,".");
    c_flag(target,"-std=c++17");
#ifdef _WIN32
    c_link_system(target,"ws2_32");
#endif
    c_default_target(b,target);
}
