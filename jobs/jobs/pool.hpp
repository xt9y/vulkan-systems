#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace jobs {
    // Execute work on threads, return optional callbacks to the caller thread.
    using completion = std::function<void()>;
    using task = std::function<completion()>;

    class pool {
    public:
        pool() = default;
        ~pool() { stop(); }
        pool(const pool&) = delete;
        pool& operator=(const pool&) = delete;
        void start(std::size_t threads = 0, std::size_t max_queued = 1024);
        bool submit(task work);
        std::size_t commit(std::size_t budget = 64);
        void wait_idle();
        void stop() noexcept;
        std::size_t pending() const;
    private:
        void run();
        mutable std::mutex mutex_;
        std::condition_variable wake_, idle_;
        std::deque<task> pending_;
        std::deque<completion> completed_;
        std::vector<std::thread> workers_;
        std::exception_ptr error_;
        std::size_t active_ = 0;
        std::size_t max_queued_ = 0;
        bool running_ = false, stopping_ = false;
    };
}
