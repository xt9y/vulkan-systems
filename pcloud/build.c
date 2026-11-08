#include <cbuild.h>
#include <stdio.h>
#include "integration.h"

// Standalone build of pc:: inside vulkan-systems. The game uses the same
// source directly through its checked-in Git submodule instead.
void build(C_Build* b)
{
    // Standalone development also compiles shaders from the same source
    // used by VulkanWrapper consumers using pc_build_compile_shaders().
    const pc_build_paths paths={"..",NULL,NULL,NULL};
    if(pc_build_compile_shaders(paths,"build/shaders")!=0)exit(1);

    C_Dependency* wrapper=c_git(b,"VulkanWrapper",
        "https://github.com/xt9y/vulkan-wrapper.git",
        "69bae1717dbcb5df838f4fc6de78de50507ed4ac");
    c_dep_cbuild(wrapper,"VulkanWrapper",C_TARGET_STATIC_LIBRARY);
    c_dep_include(wrapper,".");

    C_Target* gpu=c_static_library(b,"VulkanGPU");
    c_sources(gpu,"../gpu/gpu/*.cpp");
    C_Target* render=c_static_library(b,"VulkanRender");
    c_sources(render,"../render/render/*.cpp");
    C_Target* bvh=c_static_library(b,"VulkanBVH");
    c_sources(bvh,"../bvh/bvh/*.cpp");
    C_Target* spatial=c_static_library(b,"VulkanSpatial");
    c_sources(spatial,"../spatial/spatial/*.cpp");
    C_Target* jobs=c_static_library(b,"VulkanJobs");
    c_sources(jobs,"../jobs/jobs/*.cpp");
    C_Target* pcloud=c_static_library(b,"VulkanPCloud");
    c_sources(pcloud,"pcloud/*.cpp");

    C_Target* targets[]={gpu,render,bvh,spatial,jobs,pcloud};
    for(unsigned int i=0;i<sizeof(targets)/sizeof(targets[0]);++i)
    {
        C_Target* target=targets[i];
        c_include(target,".");
        c_include(target,"../gpu");
        c_include(target,"../render");
        c_include(target,"../bvh");
        c_include(target,"../spatial");
        c_include(target,"../jobs");
        c_flag(target,"-std=c++17");
        c_use(target,wrapper);
    }
    c_link_target(render,gpu);
    c_link_target(pcloud,render);
    c_link_target(pcloud,gpu);
    c_link_target(pcloud,bvh);
    c_link_target(pcloud,spatial);
    c_link_target(pcloud,jobs);
#ifndef _WIN32
    c_link_system(jobs,"pthread");
#endif
    c_default_target(b,pcloud);
}
