#include <cbuild.h>
void build(C_Build* b) {
    C_Dependency* crypto=c_git(b,"MbedTLS",
        "https://github.com/Mbed-TLS/mbedtls.git",
        "068ff080b369adfac81509f9b57b2afabaf82dc5");
    c_dep_cmake(crypto);
    c_dep_cmake_option(crypto,"-DUSE_SHARED_MBEDTLS_LIBRARY=OFF");
    c_dep_cmake_option(crypto,"-DUSE_STATIC_MBEDTLS_LIBRARY=ON");
    c_dep_cmake_option(crypto,"-DENABLE_TESTING=OFF");
    c_dep_cmake_option(crypto,"-DENABLE_PROGRAMS=OFF");
    c_dep_include(crypto,"include");
    c_dep_link(crypto,"mbedtls");
    c_dep_link(crypto,"mbedx509");
    c_dep_link(crypto,"mbedcrypto");
    C_Target* target=c_static_library(b,"VulkanNetwork");
    c_sources(target,"network/*.cpp");
    c_include(target,".");
    c_flag(target,"-std=c++17");
    c_use(target,crypto);
#ifdef _WIN32
    c_link_system(target,"ws2_32");
#endif
    c_default_target(b,target);
}
