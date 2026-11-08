#include <cbuild.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// The embedded runtime GLSL is source data in a header. Include its
// contents in the compiler command's cache key to avoid stale shaders.
static void fingerprint_lumen_headers(C_Target* target)
{
    static const char* paths[]={"lumen/shaders.hpp","lumen/lumen.hpp"};
    uint64_t hash=UINT64_C(14695981039346656037);
    for(size_t i=0;i<sizeof(paths)/sizeof(paths[0]);i++)
    {
        FILE* file=fopen(paths[i],"rb");
        if(!file)
        {
            fprintf(stderr,"Missing Lumen header: %s\\n",paths[i]);
            exit(1);
        }
        int byte;
        while((byte=fgetc(file))!=EOF)
        {
            hash^=(unsigned char)byte;
            hash*=UINT64_C(1099511628211);
        }
        if(ferror(file))
        {
            fprintf(stderr,"Failed reading Lumen header: %s\\n",paths[i]);
            fclose(file);
            exit(1);
        }
        fclose(file);
    }
    static char flag[96];
    snprintf(flag,sizeof(flag),"-DLUMEN_EMBEDDED_SHADER_HASH=%lluULL",
             (unsigned long long)hash);
    c_flag(target,flag);
}

void build(C_Build* b)
{
    C_Target* target=c_static_library(b,"VulkanLumen");
    c_sources(target,"lumen/*.cpp");
    fingerprint_lumen_headers(target);
    c_include(target,".");
    c_include(target,"../gpu");
    c_include(target,"../render");
    c_include(target,"../bvh"); // Reuse VulkanBVH std430 node and triangle types.
    c_flag(target,"-std=c++17");

    C_Dependency* engine=c_git(
        b,"VulkanWrapper","https://github.com/xt9y/vulkan-wrapper.git",
        "0d4e7e2ff59516ad9225015dffb22c5f487c485c");
    c_dep_cbuild(engine,"VulkanWrapper",C_TARGET_STATIC_LIBRARY);
    c_dep_include(engine,".");
    c_use(target,engine);
    c_default_target(b,target);
}
