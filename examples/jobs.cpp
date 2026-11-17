#include <vs/jobs/pool.hpp>
#include <cstdint>
#include <iostream>
#include <vector>

// Many independent CPU jobs, with completion callbacks run on the main thread.
int main() {
    constexpr std::size_t chunks=256;
    std::vector<std::uint64_t> sums(chunks);
    vs::jobs::pool workers;
    workers.start(0,chunks);
    for(std::size_t id=0;id<chunks;++id) {
        const bool queued=workers.submit([&,id]() -> vs::jobs::completion {
            std::uint64_t sum=0;
            for(std::uint64_t x=id*10000;x<(id+1)*10000;++x)sum+=x*x;
            return [&,id,sum] {sums[id]=sum;};
        });
        if(!queued)return 1;
    }
    workers.wait_idle();
    while(workers.commit(64)) {}
    workers.stop();
    std::uint64_t checksum=0;
    for(const auto v:sums)checksum+=v;
    std::cout<<"jobs="<<chunks<<" checksum="<<checksum<<'\n';
}
