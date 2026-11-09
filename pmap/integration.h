#ifndef VULKAN_PM_INTEGRATION_H
#define VULKAN_PM_INTEGRATION_H
// Include in an existing c-build build.c. Reuses the host VulkanWrapper.
// The shader compiler is glslangValidator (must be on PATH).
#include <cbuild.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define PM_MKDIR(path) _mkdir(path)
#else
#define PM_MKDIR(path) mkdir(path,0755)
#endif

typedef struct pm_build_paths
{
    const char* systems;
    const char* wrapper;
    const char* headers;
    const char* generated;
} pm_build_paths;
typedef struct pm_build_targets
{
    C_Target* gpu;
    C_Target* bvh;
    C_Target* pmap;
} pm_build_targets;

static int pm_path(char* buffer,size_t capacity,const char* a,const char* b)
{
    if(!buffer || !a || !b)return -1;
    const int written=snprintf(buffer,capacity,"%s/%s",a,b);
    return written<0 || (size_t)written>=capacity?-1:0;
}
static void pm_build_includes(C_Target* target,pm_build_paths paths)
{
    char buffer[1024];
    if(!target || !paths.systems || !paths.wrapper ||
       !paths.headers || !paths.generated)
    {
        fprintf(stderr,"pm: invalid include paths\n");
        exit(1);
    }
    c_include(target,paths.wrapper);
    if(pm_path(buffer,sizeof(buffer),paths.wrapper,"vk"))exit(1);
    c_include(target,buffer);
    if(pm_path(buffer,sizeof(buffer),paths.headers,"include"))exit(1);
    c_include(target,buffer);
    c_include(target,paths.generated);
    const char* modules[]={"gpu","bvh","pmap"};
    for(size_t i=0;i<sizeof(modules)/sizeof(modules[0]);++i)
    {
        if(pm_path(buffer,sizeof(buffer),paths.systems,modules[i]))exit(1);
        c_include(target,buffer);
    }
}
static C_Target* pm_build_module(C_Build* b,const char* name,
                                 const char* module,pm_build_paths paths,
                                 C_Target* wrapper)
{
    char filename[1024],folder[1024];
    if(pm_path(folder,sizeof(folder),paths.systems,module))exit(1);
    if(snprintf(filename,sizeof(filename),"%s/%s/*.cpp",folder,module)>=
       (int)sizeof(filename))exit(1);
    C_Target* target=c_static_library(b,name);
    c_sources(target,filename);
    pm_build_includes(target,paths);
    c_flag(target,"-std=c++17");
    if(wrapper)c_link_target(target,wrapper);
    return target;
}
// Reuse existing GPU/BVH static targets and the *same* VulkanWrapper.
static pm_build_targets pm_build_attach(C_Build* b,C_Target* wrapper,
                                       pm_build_paths paths,
                                       pm_build_targets existing)
{
    if(!b || !wrapper || !existing.gpu || !existing.bvh || existing.pmap)
    {
        fprintf(stderr,"pm: attach requires existing wrapper, gpu and bvh\n");
        exit(1);
    }
    existing.pmap=pm_build_module(b,"VulkanPMap","pmap",paths,wrapper);
    c_link_target(existing.pmap,existing.gpu);
    c_link_target(existing.pmap,existing.bvh);
    return existing;
}
static pm_build_targets pm_build_register(C_Build* b,C_Target* wrapper,
                                         pm_build_paths paths)
{
    pm_build_targets modules={0};
    modules.gpu=pm_build_module(b,"VulkanGPU","gpu",paths,wrapper);
    modules.bvh=pm_build_module(b,"VulkanBVH","bvh",paths,wrapper);
    return pm_build_attach(b,wrapper,paths,modules);
}
static void pm_build_link(C_Target* consumer,pm_build_targets targets,
                          C_Target* wrapper)
{
    if(!consumer || !wrapper || !targets.pmap || !targets.gpu || !targets.bvh)
    {
        fprintf(stderr,"pm: incomplete linker targets\n");
        exit(1);
    }
    c_link_target(consumer,targets.pmap);
    c_link_target(consumer,targets.gpu);
    c_link_target(consumer,targets.bvh);
    c_link_target(consumer,wrapper);
}
static int pm_mkdirs(const char* path)
{
    char copy[1024];
    if(!path || strlen(path)>=sizeof(copy))return -1;
    strcpy(copy,path);
    for(char* p=copy+1;*p;++p)
    {
        if(*p!='/' && *p!='\\')continue;
        const char delimiter=*p;
        *p='\0';
        if(PM_MKDIR(copy)!=0 && errno!=EEXIST)return -1;
        *p=delimiter;
    }
    return PM_MKDIR(copy)==0 || errno==EEXIST?0:-1;
}
static int pm_build_compile_shaders(pm_build_paths paths,const char* output_dir)
{
    static const char* files[]={
        "photon_clear.comp","photon_emit.comp","photon_bin.comp",
        "photon_prefix.comp","photon_scatter.comp","photon_cache.comp",
        "photon_direct.comp","photon_direct_filter.comp","photon_gather.comp",
        "photon_temporal.comp","photon_denoise.comp",
        "fullscreen.vert","present.frag"
    };
    char directory[1024],include_file[1024];
    if(!output_dir || !paths.systems || pm_mkdirs(output_dir) ||
       pm_path(directory,sizeof(directory),paths.systems,"pmap/shaders") ||
       pm_path(include_file,sizeof(include_file),directory,"photon_scene.glsl"))
    {
        fprintf(stderr,"pm: invalid shader directory\n");
        return -1;
    }
    struct stat include_stat;
    if(stat(include_file,&include_stat)!=0)
    {
        fprintf(stderr,"pm: missing shared shader header %s\n",include_file);
        return -1;
    }
    for(size_t i=0;i<sizeof(files)/sizeof(files[0]);++i)
    {
        char source[1200],destination[1200],command[2900];
        if(pm_path(source,sizeof(source),directory,files[i]))return -1;
        const int length=snprintf(destination,sizeof(destination),
                                 "%s/pmap_%s.spv",output_dir,files[i]);
        if(length<0 || (size_t)length>=sizeof(destination))return -1;
        struct stat src,dst;
        if(stat(source,&src)!=0)return -1;
        if(stat(destination,&dst)==0 &&
           dst.st_mtime>=src.st_mtime &&
           dst.st_mtime>=include_stat.st_mtime)continue;
        if(strchr(directory,'"') || strchr(source,'"') ||
           strchr(destination,'"'))return -1;
        const int n=snprintf(command,sizeof(command),
            "glslangValidator -V -I\"%s\" -o \"%s\" \"%s\"",
            directory,destination,source);
        if(n<0 || (size_t)n>=sizeof(command) || system(command)!=0)
        {
            fprintf(stderr,"pm: shader failed: %s\n",source);
            return -1;
        }
    }
    return 0;
}
#endif
