#include "film.hpp"
#include "../material/rheology.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace fld::surface
{
    namespace
    {
        float dot3(vec3 a,vec3 b)
        {
            return a.x*b.x+a.y*b.y+a.z*b.z;
        }
        float norm(vec3 p)
        {
            return std::sqrt(std::max(0.f,dot3(p,p)));
        }
        vec3 unit(vec3 p,vec3 fallback)
        {
            const float n=norm(p);
            return n>1e-10f?p/n:fallback;
        }
        vec3 world_direction(const grid &f,vec3 local)
        {
            const auto &m=f.transform;
            return {m(0,0)*local.x+m(0,1)*local.y+m(0,2)*local.z,
                    m(1,0)*local.x+m(1,1)*local.y+m(1,2)*local.z,
                    m(2,0)*local.x+m(2,1)*local.y+m(2,2)*local.z};
        }
        float squared(float x){return x*x;}
        std::size_t index(const grid &g,std::uint32_t x,std::uint32_t y)
        {
            return std::size_t(y)*g.width+x;
        }
    }

    float cell_area(const grid &f)
    {
        if(!f.width || !f.height)return 0.f;
        return std::max(1e-10f,f.metres_per_uv.x*f.metres_per_uv.y/
                        (float(f.width)*float(f.height)));
    }

    void init(grid &f,std::uint32_t w,std::uint32_t h,vec2 metric)
    {
        if(w<4 || h<4 || w>2048 || h>2048 ||
           !(metric.x>0.f && metric.y>0.f) ||
           !std::isfinite(metric.x) || !std::isfinite(metric.y))
            throw std::invalid_argument("Invalid surface film chart");
        f={};
        f.width=w;f.height=h;f.metres_per_uv=metric;
        f.cells.resize(std::size_t(w)*h);
    }

    void set_frame(grid &f,vec3 normal,mat4 transform)
    {
        f.normal=unit(normal,{0,1,0});
        const vec3 down{0,-1,0};
        const vec3 projected=down-f.normal*dot3(down,f.normal);
        f.tangent_v=unit(projected,unit(vec3{1,0,0}-
            f.normal*dot3(vec3{1,0,0},f.normal),{0,0,1}));
        f.tangent_u=unit(vk::util::math::cross(f.tangent_v,f.normal),
                         {1,0,0});
        f.transform=transform;
    }

    void set_chart(grid &f,const std::vector<chart_vertex> &vertices,
                   const std::vector<std::uint32_t> &indices)
    {
        if(f.cells.empty())return;
        f.geometry.assign(f.cells.size(),{});
        for(std::size_t i=0;i+2<indices.size();i+=3)
        {
            if(indices[i]>=vertices.size() || indices[i+1]>=vertices.size() ||
               indices[i+2]>=vertices.size())continue;
            const auto &a=vertices[indices[i]];
            const auto &b=vertices[indices[i+1]];
            const auto &c=vertices[indices[i+2]];
            const float du1=b.uv.x-a.uv.x,dv1=b.uv.y-a.uv.y;
            const float du2=c.uv.x-a.uv.x,dv2=c.uv.y-a.uv.y;
            const float det=du1*dv2-du2*dv1;
            if(std::abs(det)<1e-9f)continue;
            const auto p1=b.position-a.position,p2=c.position-a.position;
            const vec3 tu=unit((p1*dv2-p2*dv1)/det,{1,0,0});
            const vec3 tv=unit((p2*du1-p1*du2)/det,{0,0,1});
            const float minx=std::min({a.uv.x,b.uv.x,c.uv.x});
            const float maxx=std::max({a.uv.x,b.uv.x,c.uv.x});
            const float miny=std::min({a.uv.y,b.uv.y,c.uv.y});
            const float maxy=std::max({a.uv.y,b.uv.y,c.uv.y});
            const int x0=std::max(0,int(std::floor(minx*f.width)));
            const int x1=std::min(int(f.width)-1,int(std::ceil(maxx*f.width)));
            const int y0=std::max(0,int(std::floor(miny*f.height)));
            const int y1=std::min(int(f.height)-1,int(std::ceil(maxy*f.height)));
            for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x)
            {
                const float u=(float(x)+.5f)/f.width-a.uv.x;
                const float v=(float(y)+.5f)/f.height-a.uv.y;
                const float t=(u*dv2-v*du2)/det;
                const float s=(v*du1-u*dv1)/det;
                if(t<-.001f || s<-.001f || t+s>1.001f)continue;
                auto &point=f.geometry[index(f,std::uint32_t(x),std::uint32_t(y))];
                point.position=a.position*(1.f-t-s)+b.position*t+c.position*s;
                point.normal=unit(a.normal*(1.f-t-s)+b.normal*t+c.normal*s,
                                  {0,1,0});
                point.tangent_u=tu;point.tangent_v=tv;
                point.valid=true;
            }
        }
        // Link touching regions even when their UV coordinates are far
        // apart. Geometry is sampled per chart texel, so make a sparse
        // spatial hash of boundary texels and connect nearby counterparts.
        // This is a conservative contact topology, not an artist-made blur.
        f.seam_links.clear();
        const float du=f.metres_per_uv.x/f.width;
        const float dv=f.metres_per_uv.y/f.height;
        const float proximity=std::max(.0002f,2.8f*std::sqrt(du*du+dv*dv));
        const float inv=1.f/proximity;
        struct key
        {
            int x,y,z;
            bool operator==(const key &b) const
            {return x==b.x && y==b.y && z==b.z;}
        };
        struct hash
        {
            std::size_t operator()(key p) const
            {
                std::size_t h=14695981039346656037ull;
                h=(h^std::uint32_t(p.x))*1099511628211ull;
                h=(h^std::uint32_t(p.y))*1099511628211ull;
                h=(h^std::uint32_t(p.z))*1099511628211ull;
                return h;
            }
        };
        const auto cell_key=[&](vec3 p)->key
        {return {int(std::floor(p.x*inv)),int(std::floor(p.y*inv)),
                 int(std::floor(p.z*inv))};};
        std::unordered_map<key,std::vector<std::uint32_t>,hash> buckets;
        std::vector<std::uint32_t> boundary;
        for(std::uint32_t y=0;y<f.height;++y)
            for(std::uint32_t x=0;x<f.width;++x)
            {
                const auto id=std::uint32_t(index(f,x,y));
                if(!f.geometry[id].valid)continue;
                const bool edge=x==0 || y==0 || x+1==f.width ||
                                y+1==f.height ||
                    !f.geometry[index(f,x-std::uint32_t(x>0),y)].valid ||
                    !f.geometry[index(f,x+std::uint32_t(x+1<f.width),y)].valid ||
                    !f.geometry[index(f,x,y-std::uint32_t(y>0))].valid ||
                    !f.geometry[index(f,x,y+std::uint32_t(y+1<f.height))].valid;
                if(!edge)continue;
                boundary.push_back(id);
                buckets[cell_key(f.geometry[id].position)].push_back(id);
            }
        std::unordered_set<std::uint64_t> seen;
        for(const auto a:boundary)
        {
            const auto ka=cell_key(f.geometry[a].position);
            float best=proximity*proximity;
            std::uint32_t target=~std::uint32_t{0};
            for(int dz=-1;dz<=1;++dz)
                for(int dy=-1;dy<=1;++dy)
                    for(int dx=-1;dx<=1;++dx)
                    {
                        const auto it=buckets.find({ka.x+dx,ka.y+dy,ka.z+dz});
                        if(it==buckets.end())continue;
                        for(const auto b:it->second)
                        {
                            const auto ax=a%f.width,ay=a/f.width;
                            const auto bx=b%f.width,by=b/f.width;
                            if(std::abs(int(ax)-int(bx))<=2 &&
                               std::abs(int(ay)-int(by))<=2)continue;
                            const auto &A=f.geometry[a],&B=f.geometry[b];
                            if(dot3(A.normal,B.normal)<.65f)continue;
                            const auto d=A.position-B.position;
                            const float distance2=dot3(d,d);
                            if(distance2<best)
                            {
                                best=distance2;
                                target=b;
                            }
                        }
                    }
            if(target==~std::uint32_t{0})continue;
            const auto lo=std::min(a,target),hi=std::max(a,target);
            const auto pair=(std::uint64_t(lo)<<32u)|hi;
            if(seen.insert(pair).second)f.seam_links.emplace_back(lo,hi);
        }
        ++f.generation;
    }

    void deposit(grid &f,vec2 uv,float radius,float amount)
    {
        if(f.cells.empty() || !std::isfinite(amount) || amount<=0.f ||
           !std::isfinite(uv.x) || !std::isfinite(uv.y))return;
        radius=std::max(radius,.0001f);
        const float rx=std::max(1.f,radius/f.metres_per_uv.x*f.width);
        const float ry=std::max(1.f,radius/f.metres_per_uv.y*f.height);
        const float cx=uv.x*f.width-.5f,cy=uv.y*f.height-.5f;
        const int x0=std::max(0,int(std::floor(cx-2.f*rx)));
        const int x1=std::min(int(f.width)-1,int(std::ceil(cx+2.f*rx)));
        const int y0=std::max(0,int(std::floor(cy-2.f*ry)));
        const int y1=std::min(int(f.height)-1,int(std::ceil(cy+2.f*ry)));
        float sum=0.f;
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x)
        {
            const auto k=index(f,x,y);
            if(!f.geometry.empty() && !f.geometry[k].valid)continue;
            const float r=squared((x-cx)/rx)+squared((y-cy)/ry);
            if(r<4.f)sum+=std::exp(-2.f*r);
        }
        if(sum<=0.f)
        {
            // A hit may fall between UV islands. Use the closest actual
            // surface texel, not an invalid cell where flow cannot advance.
            int x=std::clamp(int(cx+.5f),0,int(f.width)-1);
            int y=std::clamp(int(cy+.5f),0,int(f.height)-1);
            if(!f.geometry.empty() && !f.geometry[index(f,x,y)].valid)
            {
                float nearest=std::numeric_limits<float>::infinity();
                for(std::uint32_t j=0;j<f.height;++j)
                    for(std::uint32_t i=0;i<f.width;++i)
                    {
                        if(!f.geometry[index(f,i,j)].valid)continue;
                        const float d=squared(float(i)-cx)+squared(float(j)-cy);
                        if(d<nearest)
                        {
                            nearest=d;
                            x=int(i);y=int(j);
                        }
                    }
            }
            f.cells[index(f,x,y)].mobile+=amount;
        }
        else
        {
            for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x)
            {
                const auto k=index(f,x,y);
                if(!f.geometry.empty() && !f.geometry[k].valid)continue;
                const float r=squared((x-cx)/rx)+squared((y-cy)/ry);
                if(r<4.f)
                    f.cells[k].mobile+=amount*std::exp(-2.f*r)/sum;
            }
        }
        f.supplied_m3+=amount;
        ++f.generation;
    }

    void update(grid &f,float dt,const material &fluid,
                const porous::properties &substrate,vec3 gravity)
    {
        if(f.cells.empty() || !std::isfinite(dt) || dt<=0.f)return;
        const float area=cell_area(f);
        const float dx=f.metres_per_uv.x/f.width;
        const float dy=f.metres_per_uv.y/f.height;
        const int substeps=std::clamp(int(std::ceil(dt/.008f)),1,20);
        const float step=std::min(dt,.16f)/float(substeps);
        std::vector<float> mobile_delta(f.cells.size());
        std::vector<float> pore_delta(f.cells.size());
        bool changed=false;
        for(int sub=0;sub<substeps;++sub)
        {
            std::fill(mobile_delta.begin(),mobile_delta.end(),0.f);
            std::fill(pore_delta.begin(),pore_delta.end(),0.f);
            for(std::uint32_t y=0;y<f.height;++y)
                for(std::uint32_t x=0;x<f.width;++x)
                {
                    const auto i=index(f,x,y);
                    const auto &a=f.cells[i];
                    if(a.mobile<=0.f && a.stored<=0.f)continue;
                    for(int axis=0;axis<2;++axis)
                    {
                        const auto nx=x+(axis==0),ny=y+(axis==1);
                        if(nx>=f.width || ny>=f.height)continue;
                        const auto j=index(f,nx,ny);
                        const auto &b=f.cells[j];
                        if(!f.geometry.empty() &&
                           (!f.geometry[i].valid || !f.geometry[j].valid))
                            continue;
                        const float dist=axis==0?dx:dy;
                        const float width=axis==0?dy:dx;
                        const vec3 tangent=(!f.geometry.empty() ?
                            (axis==0?f.geometry[i].tangent_u:
                                      f.geometry[i].tangent_v):
                            (axis==0?f.tangent_u:f.tangent_v));
                        const vec3 world_tangent=unit(world_direction(f,tangent),
                                                       tangent);
                        const float ga=dot3(gravity,world_tangent);
                        const float ha=a.mobile/area,hb=b.mobile/area;
                        const float h=.5f*(ha+hb);
                        if(h>0.f)
                        {
                            // Lubrication approximation of tangential flow:
                            // q=h^3/(3 mu) * (rho*g_t - grad(p_capillary)).
                            // Shear-thinning rheology changes mu locally.
                            const float shear=std::abs(ga)*h/
                                std::max(.001f,fluid.viscosity_zero);
                            const float mu=std::max(.0001f,
                                rheology::viscosity(fluid,shear));
                            const float capillary=
                                fluid.surface_tension*(hb-ha)/(dist*dist);
                            const float drive=fluid.density*ga-capillary;
                            const float q=std::pow(std::min(h,.03f),3.f)*
                                drive/(3.f*mu)*width*step;
                            // No face may remove more than a fraction of its
                            // donor's original volume. Four faces stay stable.
                            const float amount=q>=0.f ?
                                std::min(q,a.mobile*.22f) :
                                -std::min(-q,b.mobile*.22f);
                            mobile_delta[i]-=amount;
                            mobile_delta[j]+=amount;
                        }
                        const float D=axis==0?substrate.wick_u_m2_s:
                                               substrate.wick_v_m2_s;
                        const float amount=porous::capillary_flux(
                            a.stored,b.stored,area,dist,step,D,
                            std::max(0.f,substrate.capacity_m)*area);
                        pore_delta[i]-=amount;
                        pore_delta[j]+=amount;
                    }
                }
            for(std::size_t i=0;i<f.cells.size();++i)
            {
                auto &c=f.cells[i];
                const float old=c.mobile;
                c.mobile=std::max(0.f,c.mobile+mobile_delta[i]);
                c.stored=std::max(0.f,c.stored+pore_delta[i]);
                porous::absorb(c.mobile,c.stored,area,step,substrate);
                if(std::abs(c.mobile-old)>1e-15f)changed=true;
            }
        }
        if(changed)++f.generation;
    }

    double volume(const grid &f)
    {
        double total=0;
        for(const auto &c:f.cells)
            total+=double(c.mobile)+c.stored+c.residue;
        return total;
    }

    vec3 world_position(const grid &f,std::uint32_t x,std::uint32_t y)
    {
        if(x>=f.width || y>=f.height)return {};
        vec3 p{};
        const auto i=index(f,x,y);
        if(!f.geometry.empty() && f.geometry[i].valid)
            p=f.geometry[i].position;
        else
            p=f.tangent_u*((float(x)+.5f)/f.width-.5f)*f.metres_per_uv.x+
              f.tangent_v*((float(y)+.5f)/f.height-.5f)*f.metres_per_uv.y;
        const auto &m=f.transform;
        return {m(0,0)*p.x+m(0,1)*p.y+m(0,2)*p.z+m(0,3),
                m(1,0)*p.x+m(1,1)*p.y+m(1,2)*p.z+m(1,3),
                m(2,0)*p.x+m(2,1)*p.y+m(2,2)*p.z+m(2,3)};
    }

    bool take(grid &f,std::uint32_t x,std::uint32_t y,float amount)
    {
        if(x>=f.width || y>=f.height || !std::isfinite(amount) ||
           amount<=0.f)return false;
        auto &c=f.cells[index(f,x,y)];
        if(c.mobile<amount)return false;
        c.mobile-=amount;
        f.detached_m3+=amount;
        ++f.generation;
        return true;
    }
}
