#include <cbuild.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define VS_GETCWD _getcwd
#else
#include <unistd.h>
#define VS_GETCWD getcwd
#endif

// One target, all systems. Every compilation unit lives under vs/*/.
static void include(C_Target* target) {
    c_include(target,".");
    c_flag(target,"-std=c++20");
}
static void shaders(void) {
#ifdef _WIN32
    _mkdir("build");_mkdir("build/shaders");
#else
    mkdir("build",0755);mkdir("build/shaders",0755);
#endif
    const char* files[]={
        "vs/gpu/shaders/cull.comp",
        "vs/gpu/shaders/virtual_cull.comp",
        "vs/gpu/shaders/hiz.comp",
        "vs/pcloud/shaders/cloud.comp","vs/pcloud/shaders/gi.comp",
        "vs/pcloud/shaders/light.comp","vs/pcloud/shaders/tone.comp",
        "vs/pcloud/shaders/fullscreen.vert","vs/pcloud/shaders/present.frag",
        "vs/visual/shaders/inspect.vert","vs/visual/shaders/inspect.frag",
        "vs/optics/shaders/legacy_compose.frag",
        "vs/optics/shaders/legacy_compose_mirror.frag",
        "vs/optics/shaders/legacy_compose_planar.frag",
        "vs/optics/shaders/legacy_compose_glass.frag",
        "examples/shaders/particles.comp",
        "examples/shaders/triangle.vert",
        "examples/shaders/triangle.frag",
    };
    const char* includes[]={
        "vs/pcloud/shaders/lighting_shared.glsl",
        "vs/pcloud/shaders/dynamic_shadow.glsl",
        "vs/lighting/shaders/shadow_cache.glsl",
        "vs/bvh/shaders/region_shadow.glsl",
        "vs/lighting/shaders/transmission.glsl",
        "vs/lighting/shaders/segment.glsl",
        "vs/lighting/shaders/light_shape.glsl",
        "vs/optics/shaders/legacy_compose_core.glsl",
        "vs/optics/shaders/legacy_raytrace.glsl",
        "vs/optics/shaders/legacy_glass_layer.glsl"
    };
    for(size_t i=0;i<sizeof(files)/sizeof(*files);++i) {
        const char* file=files[i],*name=strrchr(file,'/');
        name=name?name+1:file;
        const char* base=strncmp(file,"examples/",9)==0?"example/shaders":file+3;
        char module[32]={},destination[512],command[2048];
        const char* slash=strchr(base,'/');
        if(!slash||(size_t)(slash-base)>=sizeof(module))exit(1);
        memcpy(module,base,(size_t)(slash-base));
        snprintf(destination,sizeof(destination),"build/shaders/%s_%s.spv",module,name);
        struct stat src,dst,include_info;
        if(stat(file,&src)) {fprintf(stderr,"Missing VS shader %s\n",file);exit(1);}
        int cached=!stat(destination,&dst) && dst.st_mtime>=src.st_mtime;
        for(size_t j=0;j<sizeof(includes)/sizeof(*includes);++j)
            if(cached && !stat(includes[j],&include_info) &&
               dst.st_mtime<include_info.st_mtime)cached=0;
        if(cached)continue;
        snprintf(command,sizeof(command),
            "glslangValidator -V -Ivs/pcloud/shaders "
            "-Ivs/lighting/shaders "
            "-Ivs/bvh/shaders "
            "-Ivs/optics/shaders \"%s\" -o \"%s\"",file,destination);
        if(system(command)) {fprintf(stderr,"Failed compiling %s\n",file);exit(1);}
    }
}
static void prepare_mbedtls(C_Dependency* crypto) {
    char cwd[4096],option[4608];
    if(!VS_GETCWD(cwd,sizeof(cwd)))exit(1);
    for(char* p=cwd;*p;++p)if(*p=='\\')*p='/';
    const int n=snprintf(option,sizeof(option),
        "-DCMAKE_PROJECT_INCLUDE=%s/vs/network/mbedtls_prepare.cmake",cwd);
    if(n<0||(size_t)n>=sizeof(option))exit(1);
    c_dep_cmake_option(crypto,option);
}
void build(C_Build* b) {
    // Examples must work with a normal build; allow CPU-only use to opt out.
    if(!getenv("VS_SKIP_SHADERS"))shaders();
    C_Dependency* wrapper=c_git(b,"VulkanWrapper",
        "https://github.com/xt9y-org/vulkan-wrapper.git",
        "69bae1717dbcb5df838f4fc6de78de50507ed4ac");
    c_dep_cbuild(wrapper,"VulkanWrapper",C_TARGET_STATIC_LIBRARY);
    c_dep_include(wrapper,".");

    C_Dependency* fastgltf=c_git(b,"fastgltf",
        "https://github.com/spnda/fastgltf.git",
        "0b800b4a584704f9e9add67c79c4c8732e552787");
    c_dep_cmake(fastgltf);
    const char* fast_flags[]={"-DBUILD_SHARED_LIBS=OFF","-DFASTGLTF_ENABLE_INSTALL=ON",
        "-DCMAKE_DISABLE_FIND_PACKAGE_simdjson=ON",
        "-DCMAKE_DISABLE_FIND_PACKAGE_simdutf=ON"};
    for(size_t i=0;i<4;++i)c_dep_cmake_option(fastgltf,fast_flags[i]);
    c_dep_include(fastgltf,"include");
    const char* fast_libs[]={"fastgltf","fastgltf_simdjson","fastgltf_simdutf"};
    for(size_t i=0;i<3;++i)c_dep_link(fastgltf,fast_libs[i]);

    C_Dependency* ozz=c_git(b,"ozz",
        "https://github.com/guillaumeblanc/ozz-animation.git",
        "744eb9d99f606eda849acb0b1204f7a3dc20bca1");
    c_dep_cmake(ozz);
    const char* ozz_flags[]={"-DBUILD_SHARED_LIBS=OFF","-Dozz_build_tools=OFF",
        "-Dozz_build_samples=OFF","-Dozz_build_howtos=OFF",
        "-Dozz_build_tests=OFF","-Dozz_build_postfix=OFF"};
    for(size_t i=0;i<6;++i)c_dep_cmake_option(ozz,ozz_flags[i]);
    c_dep_include(ozz,"include");
    const char* ozz_libs[]={"ozz_animation_offline","ozz_animation","ozz_geometry","ozz_base"};
    for(size_t i=0;i<4;++i)c_dep_link(ozz,ozz_libs[i]);

    C_Dependency* stb=c_git(b,"stb","https://github.com/nothings/stb.git",
        "d14d46f6b44456eb88beac58cba833a4135d4c89");
    c_dep_header_only(stb);c_dep_include(stb,".");

    C_Dependency* crypto=c_git(b,"MbedTLS",
        "https://github.com/Mbed-TLS/mbedtls.git",
        "068ff080b369adfac81509f9b57b2afabaf82dc5");
    c_dep_cmake(crypto);
    prepare_mbedtls(crypto);
    const char* tls_flags[]={"-DUSE_SHARED_MBEDTLS_LIBRARY=OFF",
        "-DUSE_STATIC_MBEDTLS_LIBRARY=ON",
        "-DENABLE_TESTING=OFF","-DENABLE_PROGRAMS=OFF"};
    for(size_t i=0;i<4;++i)c_dep_cmake_option(crypto,tls_flags[i]);
    c_dep_include(crypto,"include");
    const char* tls_libs[]={"mbedtls","mbedx509","mbedcrypto"};
    for(size_t i=0;i<3;++i)c_dep_link(crypto,tls_libs[i]);

    C_Dependency* glfw=c_git(b,"GLFW","https://github.com/glfw/glfw.git","3.4");
    c_dep_cmake(glfw);
    c_dep_cmake_option(glfw,"-DGLFW_BUILD_EXAMPLES=OFF");
    c_dep_cmake_option(glfw,"-DGLFW_BUILD_TESTS=OFF");
    c_dep_cmake_option(glfw,"-DGLFW_BUILD_DOCS=OFF");
#ifdef __linux__
    c_dep_cmake_option(glfw,"-DGLFW_BUILD_WAYLAND=OFF");
#endif
    c_dep_include(glfw,"include");c_dep_link(glfw,"glfw3");

    C_Target* library=c_static_library(b,"vs");
    c_sources(library,"vs/*/*.cpp");
    include(library);
    c_use(library,wrapper);c_use(library,fastgltf);c_use(library,ozz);
    c_use(library,stb);c_use(library,crypto);c_use(library,glfw);
#ifdef _WIN32
    c_link_system(library,"ws2_32");
#else
    c_link_system(library,"pthread");
#endif

    // One self-contained executable per subsystem. All reuse the VS library.
    const char* demos[]={
        "app","assets","scene","gpu","render","bvh","pcloud",
        "spatial","jobs","network","editor"
    };
    for(size_t i=0;i<sizeof(demos)/sizeof(*demos);++i) {
        char target_name[80],source[128];
        snprintf(target_name,sizeof(target_name),"vs-%s",demos[i]);
        snprintf(source,sizeof(source),"examples/%s.cpp",demos[i]);
        C_Target* example=c_executable(b,target_name);
        c_sources(example,source);
        include(example);
        c_link_target(example,library);
        c_use(example,wrapper);c_use(example,fastgltf);c_use(example,ozz);
        c_use(example,stb);c_use(example,crypto);c_use(example,glfw);
#ifdef __APPLE__
        const char* frameworks[]={"Cocoa","IOKit","CoreFoundation","QuartzCore"};
        for(size_t j=0;j<4;++j)c_framework(example,frameworks[j]);
        c_link_system(example,"c++");
#elif defined(__linux__)
        const char* libs[]={"X11","Xrandr","Xi","Xcursor","Xinerama",
            "dl","pthread","m","stdc++"};
        for(size_t j=0;j<9;++j)c_link_system(example,libs[j]);
#else
        const char* libs[]={"ws2_32","user32","gdi32","shell32"};
        for(size_t j=0;j<4;++j)c_link_system(example,libs[j]);
#endif
    }
    c_default_target(b,library);
}
