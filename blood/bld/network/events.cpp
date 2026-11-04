#include "events.hpp"
#include "../core/system.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace bld::network
{
    namespace
    {
        state *data=nullptr;
        void write32(std::uint8_t *p,std::uint32_t v)
        {
            for(int i=0;i<4;++i)p[i]=std::uint8_t(v>>(i*8));
        }
        void write64(std::uint8_t *p,std::uint64_t v)
        {
            for(int i=0;i<8;++i)p[i]=std::uint8_t(v>>(i*8));
        }
        std::uint32_t read32(const std::uint8_t *p)
        {
            std::uint32_t v=0;
            for(int i=0;i<4;++i)v|=std::uint32_t(p[i])<<(i*8);
            return v;
        }
        std::uint64_t read64(const std::uint8_t *p)
        {
            std::uint64_t v=0;
            for(int i=0;i<8;++i)v|=std::uint64_t(p[i])<<(i*8);
            return v;
        }
        void write_float(std::uint8_t *p,float v)
        {
            std::uint32_t bits;
            std::memcpy(&bits,&v,sizeof(v));
            write32(p,bits);
        }
        float read_float(const std::uint8_t *p)
        {
            const auto bits=read32(p);
            float v;
            std::memcpy(&v,&bits,sizeof(v));
            return v;
        }
        std::uint32_t checksum(const std::uint8_t *p,std::size_t length)
        {
            std::uint32_t hash=2166136261u;
            for(std::size_t i=0;i<length;++i)
                hash=(hash ^ p[i])*16777619u;
            return hash;
        }
    }

    state *get() { return data; }

    void init(std::size_t replay_window)
    {
        if(data)throw std::logic_error("Blood network already initialized");
        data=new state{};
        data->window=std::clamp<std::size_t>(replay_window,64,65536);
    }

    packet send(const impact &e)
    {
        packet p{};
        p[0]='B';p[1]='L';p[2]='D';p[3]='2';
        p[4]=2;
        p[5]=e.uv_valid?1:0;
        write64(p.data()+8,e.id);
        write32(p.data()+16,e.source);
        write32(p.data()+20,e.target);
        write32(p.data()+24,e.tick);
        write32(p.data()+28,e.seed);
        write_float(p.data()+32,e.position.x);
        write_float(p.data()+36,e.position.y);
        write_float(p.data()+40,e.position.z);
        write_float(p.data()+44,e.normal.x);
        write_float(p.data()+48,e.normal.y);
        write_float(p.data()+52,e.normal.z);
        write_float(p.data()+56,e.direction.x);
        write_float(p.data()+60,e.direction.y);
        write_float(p.data()+64,e.direction.z);
        write_float(p.data()+68,e.uv.x);
        write_float(p.data()+72,e.uv.y);
        write_float(p.data()+76,e.amount_ml);
        write_float(p.data()+80,e.velocity);
        write32(p.data()+84,checksum(p.data(),84));
        return p;
    }

    std::optional<impact> decode(const void *bytes,std::size_t size)
    {
        if(!bytes || size!=packet{}.size())return std::nullopt;
        const auto *p=static_cast<const std::uint8_t *>(bytes);
        if(p[0]!='B'||p[1]!='L'||p[2]!='D'||p[3]!='2'||
           p[4]!=2||p[6]||p[7]||(p[5]&~1u)||
           checksum(p,84)!=read32(p+84))return std::nullopt;
        impact e{};
        e.uv_valid=p[5]!=0;
        e.id=read64(p+8);
        e.source=read32(p+16);
        e.target=read32(p+20);
        e.tick=read32(p+24);
        e.seed=read32(p+28);
        e.position={read_float(p+32),read_float(p+36),read_float(p+40)};
        e.normal={read_float(p+44),read_float(p+48),read_float(p+52)};
        e.direction={read_float(p+56),read_float(p+60),read_float(p+64)};
        e.uv={read_float(p+68),read_float(p+72)};
        e.amount_ml=read_float(p+76);
        e.velocity=read_float(p+80);
        if(!e.id || !e.target || !std::isfinite(e.amount_ml) ||
           !std::isfinite(e.velocity) || e.amount_ml<=0.f ||
           e.amount_ml>30.f || e.velocity<0.f || e.velocity>25.f ||
           !std::isfinite(e.position.x) || !std::isfinite(e.position.y) ||
           !std::isfinite(e.position.z) ||
           !std::isfinite(e.normal.x) || !std::isfinite(e.normal.y) ||
           !std::isfinite(e.normal.z) ||
           !std::isfinite(e.direction.x) || !std::isfinite(e.direction.y) ||
           !std::isfinite(e.direction.z) ||
           !std::isfinite(e.uv.x) || !std::isfinite(e.uv.y) ||
           (e.uv_valid && (e.uv.x<0.f||e.uv.x>1.f||
                           e.uv.y<0.f||e.uv.y>1.f)))
            return std::nullopt;
        return e;
    }

    bool receive(const void *bytes,std::size_t size)
    {
        const auto event=decode(bytes,size);
        if(!event || !data)return false;
        std::lock_guard<std::mutex> guard(data->mutex);
        const std::uint64_t key=
            event->id ^ (std::uint64_t(event->source)*0x9e3779b97f4a7c15ull);
        if(!data->seen.insert(key).second)return false;
        if(data->history.size()>=data->window)
        {
            data->seen.erase(data->history.front());
            data->history.pop_front();
        }
        data->history.push_back(key);
        if(data->queue.size()>=data->window)data->queue.pop_front();
        data->queue.push_back(*event);
        return true;
    }

    void dispatch()
    {
        if(!data)return;
        std::deque<impact> ready;
        {
            std::lock_guard<std::mutex> guard(data->mutex);
            ready.swap(data->queue);
        }
        for(const auto &item:ready)system::apply(item);
    }

    void reset()
    {
        if(!data)return;
        std::lock_guard<std::mutex> guard(data->mutex);
        data->history.clear();
        data->seen.clear();
        data->queue.clear();
    }

    void shutdown()
    {
        delete data;
        data=nullptr;
    }
}
