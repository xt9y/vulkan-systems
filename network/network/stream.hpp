#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <vector>

namespace net::stream {
    using bytes = std::vector<std::uint8_t>;
    using clock = std::chrono::steady_clock;

    struct settings {
        std::size_t mtu = 1200;
        std::size_t max_message = 64 * 1024 * 1024;
        std::size_t max_queued = 128 * 1024 * 1024;
        std::size_t max_messages = 16;
        std::size_t window = 128;
        std::chrono::milliseconds retransmit{100};
        std::chrono::seconds receive_timeout{10};
    };

    // One channel per remote endpoint. Loss/reordering/duplicates handled by
    // fragment acknowledgments. Wire protocol is independent of application data.
    class channel {
    public:
        explicit channel(settings limits = {});
        // Returns zero if the local send window or memory budget is exceeded.
        std::uint32_t queue(bytes payload);
        // Accept one packet previously received from this channel's peer.
        bool accept(const void* data, std::size_t size, clock::time_point now);
        // Returns datagrams to send with the caller-owned UDP socket.
        std::vector<bytes> flush(std::size_t byte_budget, clock::time_point now);
        bool pop(bytes& completed);
        std::size_t ready_size() const noexcept { return completed_.empty()?0:completed_.front().size(); }
        std::size_t pending_bytes() const noexcept { return queued_bytes_; }
        void clear() noexcept;
    private:
        struct outgoing {
            std::uint32_t id = 0;
            bytes payload;
            std::vector<std::uint8_t> acked;
            std::vector<clock::time_point> last_sent;
            std::size_t remaining = 0;
        };
        struct incoming {
            bytes payload;
            std::vector<std::uint8_t> seen;
            std::size_t remaining = 0;
            clock::time_point updated{};
        };
        settings limits_;
        std::uint32_t next_id_ = 1;
        std::size_t queued_bytes_ = 0;
        std::size_t receive_bytes_ = 0;
        std::size_t completed_bytes_ = 0;
        std::deque<outgoing> outgoing_;
        std::unordered_map<std::uint32_t, incoming> incoming_;
        std::deque<std::uint32_t> history_;
        std::deque<bytes> acks_;
        std::deque<bytes> completed_;
    };
}
