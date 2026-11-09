#include <cbuild.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
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
        "photon_clear.comp","photon_emit.comp","photon_bin.comp",
        "photon_prefix.comp","photon_scatter.comp","photon_cache.comp",
        "photon_direct.comp","photon_direct_filter.comp","photon_gather.comp",
        "photon_temporal.comp","photon_denoise.comp",
        "fullscreen.vert","present.frag"
    };
    struct stat include;
    if(stat("shaders/photon_scene.glsl",&include)!=0)
    {
        fprintf(stderr,"Missing photon shader header\n");
        exit(1);
    }
    for(size_t i=0;i<sizeof(files)/sizeof(files[0]);++i)
    {
        char source[256],output[256],cmd[768];
        snprintf(source,sizeof(source),"shaders/%s",files[i]);
        snprintf(output,sizeof(output),"build/shaders/pmap_%s.spv",files[i]);
        struct stat src,dst;
        if(stat(source,&src)!=0)
        {
            fprintf(stderr,"Missing PMap shader: %s\n",source);
            exit(1);
        }
        if(stat(output,&dst)==0 && dst.st_mtime>=src.st_mtime &&
           dst.st_mtime>=include.st_mtime)continue;
        snprintf(cmd,sizeof(cmd),
            "glslangValidator -V -I\"shaders\" \"%s\" -o \"%s\"",
            source,output);
        if(system(cmd)!=0)
        {
            fprintf(stderr,"PMap shader failed: %s\n",source);
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
    C_Target* bvh=c_static_library(b,"VulkanBVH");
    c_sources(bvh,"../bvh/bvh/*.cpp");
    C_Target* mapper=c_static_library(b,"VulkanPMap");
    c_sources(mapper,"pmap/*.cpp");
    C_Target* targets[]={gpu,bvh,mapper};
    for(size_t i=0;i<sizeof(targets)/sizeof(targets[0]);++i)
    {
        c_include(targets[i],".");
        c_include(targets[i],"../gpu");
        c_include(targets[i],"../bvh");
        c_flag(targets[i],"-std=c++17");
        c_use(targets[i],wrapper);
    }
    c_link_target(mapper,gpu);
    c_link_target(mapper,bvh);
    c_default_target(b,mapper);
}
