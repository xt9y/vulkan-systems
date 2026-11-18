#pragma once
#include <vs/scene/regions.hpp>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace vs::scene::region_codec {
namespace detail {
template<bool Reading> class archive {
public:
    static constexpr bool reading=Reading;
    explicit archive(std::span<const std::uint8_t> bytes={})
        :input_(bytes){}
    template<class T> void value(T& item) {
        if constexpr(std::is_same_v<T,float>) {
            std::uint32_t bits=Reading?0u:std::bit_cast<std::uint32_t>(item);
            value(bits);
            if constexpr(Reading)item=std::bit_cast<float>(bits);
        } else {
            static_assert(std::is_integral_v<T>);
            using U=std::make_unsigned_t<T>;
            U bits=Reading?U{}:static_cast<U>(item);
            for(std::size_t i=0;i<sizeof(T);++i) {
                if constexpr(Reading) {
                    if(position_>=input_.size())
                        throw std::runtime_error("Truncated region payload");
                    bits|=U(input_[position_++])<<(i*8u);
                } else {
                    output_.push_back(std::uint8_t(bits>>(i*8u)));
                }
            }
            if constexpr(Reading)item=static_cast<T>(bits);
        }
    }
    std::vector<std::uint8_t> finish() && {
        if constexpr(Reading) {
            if(position_!=input_.size())
                throw std::runtime_error("Region payload has trailing bytes");
            return {};
        } else return std::move(output_);
    }
private:
    std::span<const std::uint8_t> input_;
    std::vector<std::uint8_t> output_;
    std::size_t position_=0;
};
template<class A> void vec(A& a,::vs::scene::vec3& v) {
    a.value(v.x);a.value(v.y);a.value(v.z);
}
template<class A> void quat(A& a,::vk::util::math::quat& q) {
    a.value(q.x);a.value(q.y);a.value(q.z);a.value(q.w);
}
template<class A> void instance(A& a,::vs::scene::instance& i) {
    a.value(i.id);a.value(i.mesh_id);a.value(i.animation);
    a.value(i.animation_tick);a.value(i.overlay_animation);
    a.value(i.overlay_root);a.value(i.overlay_weight);
    vec(a,i.pose.position);vec(a,i.pose.scale);quat(a,i.pose.rotation);
    vec(a,i.tint);a.value(i.pattern);a.value(i.editor_group);
    a.value(i.optical_strength);a.value(i.optical_mattness);
    vec(a,i.rotation_reference);
}
template<class A> void surface(A& a,::vs::scene::surface& s) {
    vec(a,s.origin);vec(a,s.u);vec(a,s.v);vec(a,s.normal);
    vec(a,s.albedo);vec(a,s.emission);
    std::uint32_t triangle=s.triangle?1u:0u;
    a.value(triangle);
    if constexpr(A::reading) {
        if(triangle>1)throw std::runtime_error("Invalid region triangle flag");
        s.triangle=triangle!=0u;
    }
    a.value(s.optical_type);a.value(s.optical_strength);
    a.value(s.roughness);a.value(s.ior);a.value(s.pattern);
    for(auto& pair:s.uv)for(auto& value:pair)a.value(value);
    for(auto& normal:s.vertex_normals)vec(a,normal);
    a.value(s.base_color_texture);a.value(s.metallic_roughness_texture);
    a.value(s.normal_texture);a.value(s.occlusion_texture);
    a.value(s.emissive_texture);a.value(s.alpha_cutoff);
    a.value(s.base_alpha);
    std::uint32_t probe=s.probe_receiver?1u:0u;
    std::uint32_t shadow=s.shadow_caster?1u:0u;
    a.value(probe);a.value(shadow);
    if constexpr(A::reading) {
        if(probe>1 || shadow>1)
            throw std::runtime_error("Invalid region visibility flags");
        s.probe_receiver=probe!=0;s.shadow_caster=shadow!=0;
    }
}
template<class A> void collision(A& a,::vs::scene::collision_box& box) {
    vec(a,box.minimum);vec(a,box.maximum);
}
template<class A> void light(A& a,::vs::scene::point_light& l) {
    vec(a,l.position);a.value(l.radius);
    vec(a,l.color);a.value(l.strength);
    std::uint32_t enabled=l.enabled?1u:0u;
    std::uint32_t kind=static_cast<std::uint32_t>(l.kind);
    a.value(enabled);a.value(kind);
    vec(a,l.direction);a.value(l.cone_degrees);
    if constexpr(A::reading) {
        if(enabled>1 || kind>2)
            throw std::runtime_error("Invalid region light");
        l.enabled=enabled!=0;l.kind=::vs::scene::light_kind(kind);
    }
}
template<class A,class T,class Visit>
void list(A& a,std::vector<T>& items,std::uint32_t maximum,Visit visit) {
    if constexpr(!A::reading) {
        if(items.size()>maximum || items.size()>std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("Region element budget exceeded");
    }
    std::uint32_t count=static_cast<std::uint32_t>(items.size());
    a.value(count);
    if(count>maximum)throw std::runtime_error("Region element budget exceeded");
    if constexpr(A::reading)items.resize(count);
    for(auto& item:items)visit(a,item);
}
template<class A> void fragment(A& a,::vs::scene::region_fragment& f) {
    std::uint32_t magic=0x46525356u; // VSRF, little endian
    std::uint32_t version=1u;
    a.value(magic);a.value(version);
    if(magic!=0x46525356u || version!=1u)
        throw std::runtime_error("Unsupported region fragment encoding");
    list(a,f.instances,4096u,instance<A>);
    list(a,f.surfaces,8192u,surface<A>);
    list(a,f.collision_boxes,8192u,collision<A>);
    list(a,f.lights,256u,light<A>);
}
} // namespace detail
inline std::vector<std::uint8_t> encode(const region_fragment& source) {
    auto fragment=source;
    detail::archive<false> writer;
    detail::fragment(writer,fragment);
    auto bytes=std::move(writer).finish();
    if(bytes.size()>8u*1024u*1024u)
        throw std::runtime_error("Region exceeds the 8 MiB stream budget");
    return bytes;
}
inline std::shared_ptr<const region_fragment> decode(
    std::span<const std::uint8_t> bytes) {
    if(bytes.size()>8u*1024u*1024u)
        throw std::runtime_error("Oversized region fragment");
    auto fragment=std::make_shared<region_fragment>();
    detail::archive<true> reader(bytes);
    detail::fragment(reader,*fragment);
    std::move(reader).finish();
    return fragment;
}
} // namespace vs::scene::region_codec
