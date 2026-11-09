#include <cbuild.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

static void compile_shaders(void)
{
#ifdef _WIN32
    _mkdir("build");
    _mkdir("build/shaders");
#else
    mkdir("build",0755);
    mkdir("build/shaders",0755);
#endif
    static const char* files[]={
        "cloud.comp","gi.comp","light.comp","tone.comp",
        "fullscreen.vert","present.frag"
    };
    for(size_t i=0;i<sizeof(files)/sizeof(files[0]);++i)
    {
        char src[256],dst[256],cmd[768];
        snprintf(src,sizeof(src),"pcloud/shaders/%s",files[i]);
        snprintf(dst,sizeof(dst),"build/shaders/pcloud_%s.spv",files[i]);
        struct stat input,output;
        if(stat(src,&input)!=0)
        {
            fprintf(stderr,"Missing PCloud shader: %s\n",src);
            exit(1);
        }
        if(stat(dst,&output)==0 && output.st_mtime>=input.st_mtime)continue;
        snprintf(cmd,sizeof(cmd),"glslangValidator -V \"%s\" -o \"%s\"",src,dst);
        if(system(cmd)!=0)
        {
            fprintf(stderr,"PCloud shader failed: %s\n",src);
            exit(1);
        }
    }
}

void build(C_Build* b)
{
    compile_shaders();
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
    for(size_t i=0;i<sizeof(targets)/sizeof(targets[0]);++i)
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
