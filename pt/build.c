#include <cbuild.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#endif

static void compile_shaders(void)
{
#ifdef _WIN32
    _mkdir("build");_mkdir("build/shaders");
#else
    mkdir("build",0755);mkdir("build/shaders",0755);
#endif
    static const char* shaders[]={
        "trace.comp","temporal.comp","denoise.comp",
        "fullscreen.vert","present.frag"
    };
    for(size_t i=0;i<sizeof(shaders)/sizeof(shaders[0]);++i)
    {
        char source[256],output[256],command[768];
        snprintf(source,sizeof(source),"shaders/%s",shaders[i]);
        snprintf(output,sizeof(output),"build/shaders/pt_%s.spv",shaders[i]);
        struct stat input,compiled;
        if(stat(source,&input)!=0)
        {
            fprintf(stderr,"Missing PT shader: %s\n",source);exit(1);
        }
        if(stat(output,&compiled)==0 && compiled.st_mtime>=input.st_mtime)
            continue;
        snprintf(command,sizeof(command),
                 "glslangValidator -V \"%s\" -o \"%s\"",source,output);
        if(system(command)!=0)
        {
            fprintf(stderr,"Failed to compile PT shader: %s\n",source);exit(1);
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
    C_Target* pt=c_static_library(b,"VulkanPT");
    c_sources(pt,"pt/*.cpp");
    C_Target* targets[]={gpu,pt};
    for(size_t i=0;i<2u;++i)
    {
        c_include(targets[i],".");
        c_include(targets[i],"../gpu");
        c_flag(targets[i],"-std=c++17");
        c_use(targets[i],wrapper);
    }
    c_link_target(pt,gpu);
    c_default_target(b,pt);
}
