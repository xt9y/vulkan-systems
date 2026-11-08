#ifndef VULKAN_PC_INTEGRATION_H
#define VULKAN_PC_INTEGRATION_H

// Include from an existing c-build build.c. Reuses the host VulkanWrapper
// target instead of fetching/building a second VulkanWrapper dependency.
#include <cbuild.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define PC_BUILD_MKDIR(path) _mkdir(path)
#else
#define PC_BUILD_MKDIR(path) mkdir(path,0755)
#endif

typedef struct pc_build_paths
{
    const char* systems;   // e.g. "third-party/vulkan-systems"
    const char* wrapper;   // e.g. "third-party/vulkan-wrapper"
    const char* headers;   // e.g. "third-party/vulkan-headers"
    const char* generated; // e.g. "build/generated"
} pc_build_paths;

typedef struct pc_build_targets
{
    C_Target* gpu;
    C_Target* render;
    C_Target* bvh;
    C_Target* spatial;
    C_Target* jobs;
    C_Target* pcloud;
} pc_build_targets;

static int pc_build_join(char* out,size_t size,const char* base,
                         const char* suffix)
{
    if(!base || !suffix)return -1;
    const int n=snprintf(out,size,"%s/%s",base,suffix);
    return n>=0 && (size_t)n<size ? 0 : -1;
}

static void pc_build_include_paths(C_Target* target,pc_build_paths paths)
{
    const char* components[]={"gpu","render","bvh","spatial","jobs","pcloud"};
    char dir[1024];
    if(!target || !paths.systems || !paths.wrapper ||
       !paths.headers || !paths.generated)
    {
        fprintf(stderr,"pc: invalid VulkanWrapper build paths\n");
        exit(1);
    }
    c_include(target,paths.wrapper);
    if(pc_build_join(dir,sizeof(dir),paths.wrapper,"vk"))exit(1);
    c_include(target,dir);
    if(pc_build_join(dir,sizeof(dir),paths.headers,"include"))exit(1);
    c_include(target,dir);
    c_include(target,paths.generated);
    for(size_t i=0;i<sizeof(components)/sizeof(components[0]);++i)
    {
        if(pc_build_join(dir,sizeof(dir),paths.systems,components[i]))exit(1);
        c_include(target,dir);
    }
}

static C_Target* pc_build_module(C_Build* b,const char* name,
                                 const char* dir,pc_build_paths paths,
                                 C_Target* wrapper)
{
    C_Target* target=c_static_library(b,name);
    char source[1024],base[1024];
    if(pc_build_join(base,sizeof(base),paths.systems,dir) ||
       pc_build_join(source,sizeof(source),base,"*.cpp"))
    {
        fprintf(stderr,"pc: systems path too long\n");
        exit(1);
    }
    // e.g. third-party/vulkan-systems/gpu/gpu/*.cpp
    if(snprintf(source,sizeof(source),"%s/%s/*.cpp",base,dir) >=
       (int)sizeof(source))exit(1);
    c_sources(target,source);
    pc_build_include_paths(target,paths);
    c_flag(target,"-std=c++17");
    if(wrapper)c_link_target(target,wrapper);
    return target;
}

// Register only when the host has not already created these named modules.
// The returned targets can be linked/reused by the rest of the host renderer.
static pc_build_targets pc_build_register(C_Build* b,C_Target* wrapper,
                                          pc_build_paths paths)
{
    pc_build_targets result;
    result.gpu=pc_build_module(b,"VulkanGPU","gpu",paths,wrapper);
    result.render=pc_build_module(b,"VulkanRender","render",paths,wrapper);
    result.bvh=pc_build_module(b,"VulkanBVH","bvh",paths,wrapper);
    result.spatial=pc_build_module(b,"VulkanSpatial","spatial",paths,wrapper);
    result.jobs=pc_build_module(b,"VulkanJobs","jobs",paths,NULL);
    result.pcloud=pc_build_module(b,"VulkanPCloud","pcloud",paths,wrapper);
    c_link_target(result.render,result.gpu);
    c_link_target(result.pcloud,result.gpu);
    c_link_target(result.pcloud,result.render);
    c_link_target(result.pcloud,result.bvh);
    c_link_target(result.pcloud,result.spatial);
    c_link_target(result.pcloud,result.jobs);
#ifndef _WIN32
    c_link_system(result.jobs,"pthread");
#endif
    return result;
}

static int pc_build_mkdirs(const char* path)
{
    char copy[1024];
    if(!path || strlen(path)>=sizeof(copy))return -1;
    strcpy(copy,path);
    char* start=copy;
#ifdef _WIN32
    if(copy[0] && copy[1]==':')start=copy+2;
#endif
    for(char* p=start+1;*p;++p)
    {
        if(*p!='/' && *p!='\\')continue;
        const char separator=*p;
        *p='\0';
        if(PC_BUILD_MKDIR(copy)!=0 && errno!=EEXIST)return -1;
        *p=separator;
    }
    return PC_BUILD_MKDIR(copy)==0 || errno==EEXIST ? 0 : -1;
}

// Build the six bundled shaders once and incrementally update changed files.
// glslangValidator must be on PATH. The default pc::configuration reads
// from "build/shaders/pcloud_<source>.spv".
static int pc_build_compile_shaders(pc_build_paths paths,const char* out_dir)
{
    static const char* names[]={
        "cloud.comp","gi.comp","light.comp","tone.comp",
        "fullscreen.vert","present.frag"
    };
    if(!out_dir || !paths.systems || pc_build_mkdirs(out_dir))
    {
        fprintf(stderr,"pc: cannot create shader output directory\n");
        return -1;
    }
    char shader_dir[1024];
    if(pc_build_join(shader_dir,sizeof(shader_dir),paths.systems,
                     "pcloud/pcloud/shaders"))
        return -1;
    for(size_t i=0;i<sizeof(names)/sizeof(names[0]);++i)
    {
        char source[1200],binary[1200],command[2600];
        if(pc_build_join(source,sizeof(source),shader_dir,names[i]))
            return -1;
        const int n=snprintf(binary,sizeof(binary),"%s/pcloud_%s.spv",
                             out_dir,names[i]);
        if(n<0 || (size_t)n>=sizeof(binary))return -1;
        struct stat input_info,output_info;
        if(stat(source,&input_info)!=0)
        {
            fprintf(stderr,"pc: missing shader %s\n",source);
            return -1;
        }
        if(stat(binary,&output_info)==0 &&
           output_info.st_mtime>=input_info.st_mtime)continue;
        if(strchr(source,'"') || strchr(binary,'"'))return -1;
        const int length=snprintf(command,sizeof(command),
              "glslangValidator -V \"%s\" -o \"%s\"",source,binary);
        if(length<0 || (size_t)length>=sizeof(command) || system(command)!=0)
        {
            fprintf(stderr,"pc: shader compilation failed for %s\n",source);
            return -1;
        }
    }
    return 0;
}

#endif
