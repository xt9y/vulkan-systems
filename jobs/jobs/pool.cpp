#include "pool.hpp"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace jobs {
    void pool::start(std::size_t count, std::size_t max_queued) {
        if(!max_queued) throw std::invalid_argument("queue limit must be positive");
        if(!count) count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if(running_) throw std::logic_error("job pool already running");
            running_=true; stopping_=false; error_={}; max_queued_=max_queued;
        }
        try {
            for(std::size_t i=0; i<count; ++i)
                workers_.emplace_back([this]{run();});
        } catch(...) { stop(); throw; }
    }

    void pool::run() {
        for(;;) {
            task work;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock,[this]{return stopping_ || !pending_.empty();});
                if(stopping_) return;
                work=std::move(pending_.front());
                pending_.pop_front(); ++active_;
            }
            completion done;
            std::exception_ptr error;
            try { done=work(); }
            catch(...) { error=std::current_exception(); }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if(error && !error_) {
                    error_=error;
                    pending_.clear();
                }
                if(done && !error_) completed_.push_back(std::move(done));
                --active_;
                if(pending_.empty() && active_==0) idle_.notify_all();
            }
        }
    }

    bool pool::submit(task work) {
        if(!work) return false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if(!running_ || stopping_ || error_ || pending_.size()>=max_queued_) return false;
            pending_.push_back(std::move(work));
        }
        wake_.notify_one();
        return true;
    }

    std::size_t pool::commit(std::size_t budget) {
        std::deque<completion> work;
        std::exception_ptr error;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            error=error_;
            while(budget-- && !completed_.empty()) {
                work.push_back(std::move(completed_.front()));
                completed_.pop_front();
            }
        }
        if(error) std::rethrow_exception(error);
        for(auto& complete:work) complete();
        return work.size();
    }

    void pool::wait_idle() {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_.wait(lock,[this]{return pending_.empty() && active_==0;});
        if(error_) std::rethrow_exception(error_);
    }

    void pool::stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if(!running_ && workers_.empty()) return;
            stopping_=true; pending_.clear();
        }
        wake_.notify_all();
        idle_.notify_all();
        for(auto& thread:workers_) if(thread.joinable()) thread.join();
        workers_.clear();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_.clear(); error_={}; running_=false; stopping_=false;
            active_=0; max_queued_=0;
        }
    }

    std::size_t pool::pending() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_.size()+active_;
    }
}
