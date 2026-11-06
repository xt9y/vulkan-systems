#include "stream.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace net::stream {
namespace {
    constexpr std::size_t header_size=28;
    constexpr std::uint8_t data_flag=1, ack_flag=2;
    std::uint32_t read32(const std::uint8_t* p) {
        return (std::uint32_t(p[0])<<24)|(std::uint32_t(p[1])<<16)|
               (std::uint32_t(p[2])<<8)|std::uint32_t(p[3]);
    }
    void write32(std::uint8_t* p,std::uint32_t v) {
        p[0]=std::uint8_t(v>>24);p[1]=std::uint8_t(v>>16);
        p[2]=std::uint8_t(v>>8);p[3]=std::uint8_t(v);
    }
    std::uint32_t hash(const std::uint8_t* p,std::size_t n) {
        std::uint32_t h=2166136261u;
        for(std::size_t i=0;i<n;++i) h=(h^p[i])*16777619u;
        return h;
    }
    bytes packet(std::uint8_t type,std::uint32_t id,std::uint32_t size,
                 std::uint32_t part,std::uint32_t count,
                 const std::uint8_t* data=nullptr,std::size_t length=0) {
        bytes result(header_size+length,0);
        result[0]='V'; result[1]='S'; result[2]='N'; result[3]='1';
        result[4]=type;
        write32(result.data()+8,id);
        write32(result.data()+12,size);
        write32(result.data()+16,part);
        write32(result.data()+20,count);
        write32(result.data()+24,data ? hash(data,length) : 0u);
        if(length)std::memcpy(result.data()+header_size,data,length);
        return result;
    }
}
channel::channel(settings limits):limits_(limits) {
    if(limits_.mtu<header_size+64 || limits_.mtu>1472 ||
       !limits_.max_message || limits_.max_message>std::numeric_limits<std::uint32_t>::max() ||
       limits_.max_queued<limits_.max_message || !limits_.max_messages ||
       !limits_.window || limits_.retransmit.count()<=0 ||
       limits_.receive_timeout.count()<=0)
        throw std::invalid_argument("invalid reliable stream settings");
}
std::uint32_t channel::queue(bytes payload) {
    if(payload.empty() || payload.size()>limits_.max_message ||
       payload.size()>limits_.max_queued-std::min(queued_bytes_,limits_.max_queued) ||
       outgoing_.size()>=limits_.max_messages)
        return 0;
    const auto cap=limits_.mtu-header_size;
    const auto parts=(payload.size()+cap-1)/cap;
    if(parts>std::numeric_limits<std::uint32_t>::max())return 0;
    const std::uint32_t id=next_id_++;
    if(!next_id_)++next_id_;
    outgoing value{};
    value.id=id;
    value.remaining=parts;
    value.acked.resize(parts,0);
    value.last_sent.resize(parts);
    value.payload=std::move(payload);
    queued_bytes_+=value.payload.size();
    outgoing_.push_back(std::move(value));
    return id;
}
bool channel::accept(const void* input,std::size_t size,clock::time_point now) {
    if(!input || size<header_size || size>limits_.mtu)return false;
    const auto* p=static_cast<const std::uint8_t*>(input);
    if(p[0]!='V'||p[1]!='S'||p[2]!='N'||p[3]!='1'||
       (p[4]!=data_flag && p[4]!=ack_flag) || p[5] || p[6] || p[7])
        return false;
    const auto id=read32(p+8), total=read32(p+12),
               part=read32(p+16), count=read32(p+20), check=read32(p+24);
    if(!id || !total || total>limits_.max_message)return false;
    const auto cap=limits_.mtu-header_size;
    if(!count || count!=(std::size_t(total)+cap-1)/cap || part>=count)return false;
    if(p[4]==ack_flag) {
        if(size!=header_size || check)return false;
        for(auto it=outgoing_.begin();it!=outgoing_.end();++it) {
            if(it->id!=id || it->payload.size()!=total || it->acked.size()!=count)continue;
            if(!it->acked[part]) {
                it->acked[part]=1;
                if(!--it->remaining) {
                    queued_bytes_-=it->payload.size();
                    outgoing_.erase(it);
                }
            }
            return true;
        }
        return false;
    }
    const auto offset=std::size_t(part)*cap;
    const auto length=std::min<std::size_t>(cap,total-offset);
    if(size!=header_size+length || check!=hash(p+header_size,length))return false;
    // Acknowledge even repeated fragments of completed messages.
    if(std::find(history_.begin(),history_.end(),id)!=history_.end()) {
        if(acks_.size()<8192)acks_.push_back(packet(ack_flag,id,total,part,count));
        return true;
    }
    auto it=incoming_.find(id);
    if(it==incoming_.end()) {
        if(incoming_.size()>=limits_.max_messages ||
           std::size_t(total)>limits_.max_queued-
               std::min(receive_bytes_,limits_.max_queued))
            return false;
        incoming record{};
        record.payload.resize(total);
        record.seen.resize(count,0);
        record.remaining=count;
        record.updated=now;
        receive_bytes_+=total;
        it=incoming_.emplace(id,std::move(record)).first;
    }
    auto& record=it->second;
    if(record.payload.size()!=total || record.seen.size()!=count)return false;
    record.updated=now;
    if(!record.seen[part]) {
        if(record.remaining==1 && record.payload.size()>
           limits_.max_queued-std::min(completed_bytes_,limits_.max_queued))
            return false;
        std::memcpy(record.payload.data()+offset,p+header_size,length);
        record.seen[part]=1;
        --record.remaining;
    }
    if(acks_.size()<8192)acks_.push_back(packet(ack_flag,id,total,part,count));
    if(!record.remaining) {
        completed_bytes_+=total;
        completed_.push_back(std::move(record.payload));
        receive_bytes_-=total;
        incoming_.erase(it);
        history_.push_back(id);
        if(history_.size()>256)history_.pop_front();
    }
    return true;
}
std::vector<bytes> channel::flush(std::size_t budget,clock::time_point now) {
    for(auto it=incoming_.begin();it!=incoming_.end();) {
        if(now-it->second.updated>limits_.receive_timeout) {
            receive_bytes_-=it->second.payload.size();
            it=incoming_.erase(it);
        } else ++it;
    }
    std::vector<bytes> result;
    while(!acks_.empty() && budget>=acks_.front().size()) {
        budget-=acks_.front().size();
        result.push_back(std::move(acks_.front()));
        acks_.pop_front();
    }
    const auto cap=limits_.mtu-header_size;
    for(bool progress=true;progress && budget>=header_size+1;) {
        progress=false;
        for(auto& message:outgoing_) {
            if(budget<header_size+1)break;
            std::size_t in_flight=0;
            for(std::size_t i=0;i<message.acked.size();++i)
                if(!message.acked[i] && message.last_sent[i]!=clock::time_point{})
                    ++in_flight;
            for(std::size_t i=0;i<message.acked.size();++i) {
                if(message.acked[i])continue;
                const bool sent=message.last_sent[i]!=clock::time_point{};
                if(sent && now-message.last_sent[i]<limits_.retransmit)continue;
                if(!sent && in_flight>=limits_.window)continue;
                const auto offset=i*cap;
                const auto length=std::min(cap,message.payload.size()-offset);
                if(budget<header_size+length)continue;
                auto data=packet(data_flag,message.id,
                    std::uint32_t(message.payload.size()),std::uint32_t(i),
                    std::uint32_t(message.acked.size()),
                    message.payload.data()+offset,length);
                budget-=data.size();
                result.push_back(std::move(data));
                message.last_sent[i]=now;
                if(!sent)++in_flight;
                progress=true;
                break; // fair scheduling between streams
            }
        }
    }
    return result;
}
bool channel::pop(bytes& out) {
    if(completed_.empty())return false;
    completed_bytes_-=completed_.front().size();
    out=std::move(completed_.front());
    completed_.pop_front();
    return true;
}
void channel::clear() noexcept {
    outgoing_.clear(); incoming_.clear(); history_.clear(); acks_.clear();
    completed_.clear(); queued_bytes_=receive_bytes_=completed_bytes_=0; next_id_=1;
}
}
