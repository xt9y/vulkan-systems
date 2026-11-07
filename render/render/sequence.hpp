#pragma once
#include <gpu/graph.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace render
{
    // Reusable ordered render/compute pass dispatcher, independent of scene,
    // camera, G-buffer layout, lighting algorithms and swapchain policy.
    class sequence
    {
    public:
        struct pass
        {
            std::string name;
            std::vector<gpu::access> resources;
            gpu::graph::callback record;
        };

        void resize(std::size_t frames)
        {
            graphs_.clear();
            graphs_.resize(frames);
        }

        void reset() noexcept {graphs_.clear();}

        // The caller must declare image/buffer accesses that require barriers.
        // Existing render-pass external dependencies may handle attachments.
        void record(vk::CommandBuffer command,std::size_t image,
                    std::vector<pass> passes)
        {
            if(!command || image>=graphs_.size())
                throw std::invalid_argument("Invalid render sequence");
            auto& graph=graphs_[image];
            graph.clear_passes();
            try
            {
                for(auto& p:passes)
                    graph.add(std::move(p.name),std::move(p.resources),
                              std::move(p.record));
                graph.execute(command);
            }
            catch(...)
            {
                graph.clear_passes();
                throw;
            }
            graph.clear_passes();
        }
    private:
        std::vector<gpu::graph> graphs_;
    };
}
