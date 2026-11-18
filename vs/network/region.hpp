#pragma once
#include <vs/spatial/world.hpp>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace vs::network::region {
// A bounded application-level message transported by network::session/DTLS.
// Replace, remove, and manifest are separate from model/texture asset blobs.
enum class operation : std::uint8_t {replace=1,remove=2,manifest=3};
struct message {
    operation action=operation::replace;
    ::vs::spatial::region_id id{};
    std::uint64_t revision=0;
    std::vector<std::uint8_t> payload;
};
inline constexpr std::size_t header_bytes=48;
inline constexpr std::size_t max_payload=8u*1024u*1024u;
namespace detail {
inline void write(std::uint8_t* p,std::uint64_t value) {
    for(int i=0;i<8;++i)p[i]=std::uint8_t(value>>(8*i));
}
inline std::uint64_t read(const std::uint8_t* p) {
    std::uint64_t value=0;
    for(int i=0;i<8;++i)value|=std::uint64_t(p[i])<<(8*i);
    return value;
}
inline std::uint32_t hash(std::span<const std::uint8_t> data) {
    std::uint32_t v=2166136261u;
    for(auto b:data)v=(v^b)*16777619u;
    return v;
}
}
inline std::vector<std::uint8_t> encode(const message& m) {
    if(m.revision==0 || m.payload.size()>max_payload ||
       (m.action==operation::remove && !m.payload.empty()) ||
       (m.action!=operation::replace && m.action!=operation::remove &&
        m.action!=operation::manifest))
        throw std::invalid_argument("Invalid region network message");
    std::vector<std::uint8_t> out(header_bytes+m.payload.size());
    out[0]='V';out[1]='S';out[2]='R';out[3]='1';
    out[4]=static_cast<std::uint8_t>(m.action);
    detail::write(out.data()+8,std::bit_cast<std::uint64_t>(m.id.x));
    detail::write(out.data()+16,std::bit_cast<std::uint64_t>(m.id.y));
    detail::write(out.data()+24,std::bit_cast<std::uint64_t>(m.id.z));
    detail::write(out.data()+32,m.revision);
    detail::write(out.data()+40,
        (std::uint64_t(detail::hash(m.payload))<<32)|m.payload.size());
    if(!m.payload.empty())
        std::memcpy(out.data()+header_bytes,m.payload.data(),m.payload.size());
    return out;
}
inline std::optional<message> decode(std::span<const std::uint8_t> bytes) {
    if(bytes.size()<header_bytes || bytes.size()>header_bytes+max_payload ||
       bytes[0]!='V'||bytes[1]!='S'||bytes[2]!='R'||bytes[3]!='1' ||
       bytes[5]||bytes[6]||bytes[7]||bytes[4]<1||bytes[4]>3)
        return std::nullopt;
    message m;
    m.action=static_cast<operation>(bytes[4]);
    m.id={std::bit_cast<std::int64_t>(detail::read(bytes.data()+8)),
          std::bit_cast<std::int64_t>(detail::read(bytes.data()+16)),
          std::bit_cast<std::int64_t>(detail::read(bytes.data()+24))};
    m.revision=detail::read(bytes.data()+32);
    const auto checksum=detail::read(bytes.data()+40);
    const auto length=static_cast<std::uint32_t>(checksum);
    if(!m.revision || length!=bytes.size()-header_bytes ||
       (m.action==operation::remove && length))return std::nullopt;
    m.payload.assign(bytes.begin()+header_bytes,bytes.end());
    if(detail::hash(m.payload)!=std::uint32_t(checksum>>32))return std::nullopt;
    return m;
}
} // namespace vs::network::region
