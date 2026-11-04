#pragma once
// Headless test-only math shim. Production builds always use VulkanEngine
// math; this header merely allows deterministic liquid-physics tests to run
// in CI without granting the Actions runner private repository credentials.
#include <cmath>
#include <cstddef>
namespace vk::util::math
{
    struct vec2 { float x=0,y=0; };
    struct vec3 { float x=0,y=0,z=0; };
    struct vec4 { float x=0,y=0,z=0,w=0; };
    inline vec3 operator+(vec3 a,vec3 b) {return {a.x+b.x,a.y+b.y,a.z+b.z};}
    inline vec3 operator-(vec3 a,vec3 b) {return {a.x-b.x,a.y-b.y,a.z-b.z};}
    inline vec3 operator-(vec3 a) {return {-a.x,-a.y,-a.z};}
    inline vec3 operator*(vec3 a,float s) {return {a.x*s,a.y*s,a.z*s};}
    inline vec3 operator*(float s,vec3 a) {return a*s;}
    inline vec3 operator/(vec3 a,float s) {return {a.x/s,a.y/s,a.z/s};}
    inline float dot(vec3 a,vec3 b) {return a.x*b.x+a.y*b.y+a.z*b.z;}
    inline float length_squared(vec3 a) {return dot(a,a);}
    inline float length(vec3 a) {return std::sqrt(length_squared(a));}
    inline vec3 cross(vec3 a,vec3 b)
    {return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
    struct mat4
    {
        float m[16]{};
        static mat4 identity()
        {
            mat4 value;
            for(int i=0;i<4;++i)value.m[i*4+i]=1;
            return value;
        }
        float &operator()(int row,int column) {return m[row*4+column];}
        float operator()(int row,int column) const {return m[row*4+column];}
    };
    inline vec4 operator*(const mat4 &m,vec4 v)
    {
        vec4 out{};
        const float a[4]{v.x,v.y,v.z,v.w};
        float result[4]{};
        for(int row=0;row<4;++row)
            for(int col=0;col<4;++col)result[row]+=m(row,col)*a[col];
        return {result[0],result[1],result[2],result[3]};
    }
    inline mat4 operator*(const mat4 &a,const mat4 &b)
    {
        mat4 c{};
        for(int row=0;row<4;++row)
            for(int col=0;col<4;++col)
                for(int k=0;k<4;++k)
                    c(row,col)+=a(row,k)*b(k,col);
        return c;
    }
}
