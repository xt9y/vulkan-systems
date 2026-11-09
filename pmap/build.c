#include <cbuild.h>
#include <stdio.h>
#include <stdlib.h>
#include "integration.h"

void build(C_Build* b)
{
    const pm_build_paths shaders={"..",NULL,NULL,NULL};
    if(pm_build_compile_shaders(shaders,"build/shaders")!=0)exit(1);
    C_Dependency* wrapper=c_git(b,"VulkanWrapper",
        "https://github.com/xt9y/vulkan-wrapper.git",
        "69bae1717dbcb5df838f4fc6de78de50507ed4ac");
    c_dep_cbuild(wrapper,"VulkanWrapper",C_TARGET_STATIC_LIBRARY);
    c_dep_include(wrapper,".");
    C_Target* gpu=c_static_library(b,"VulkanGPU");
    c_sources(gpu,"../gpu/gpu/*.cpp");
    C_Target* bvh=c_static_library(b,"VulkanBVH");
    c_sources(bvh,"../bvh/bvh/*.cpp");
    C_Target* mapper=c_static_library(b,"VulkanPMap");
    c_sources(mapper,"pmap/*.cpp");
    C_Target* all[]={gpu,bvh,mapper};
    for(size_t i=0;i<sizeof(all)/sizeof(all[0]);++i)
    {
        c_include(all[i],".");
        c_include(all[i],"../gpu");
        c_include(all[i],"../bvh");
        c_flag(all[i],"-std=c++17");
        c_use(all[i],wrapper);
    }
    c_link_target(mapper,gpu);
    c_link_target(mapper,bvh);
    c_default_target(b,mapper);
}
