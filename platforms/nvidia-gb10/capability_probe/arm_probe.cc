#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

struct alignas(64) Slot {
  std::atomic<std::uint64_t> sequence{0};
  std::uint64_t payload{0};
};

int main(int argc, char** argv) {
  const std::uint64_t iterations = argc > 1 ? std::stoull(argv[1]) : 1000000;
  Slot slot;
  std::atomic<bool> failed{false};
  std::atomic<std::uint64_t> handoffs{0};
  std::thread producer([&] {
    for (std::uint64_t value = 1; value <= iterations; ++value) {
      const std::uint64_t empty_sequence = 2 * (value - 1);
      while (slot.sequence.load(std::memory_order_acquire) != empty_sequence) {
        std::this_thread::yield();
      }
      slot.payload = value ^ 0x9e3779b97f4a7c15ULL;
      slot.sequence.store(empty_sequence + 1, std::memory_order_release);
      handoffs.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::thread consumer([&] {
    for (std::uint64_t value = 1; value <= iterations; ++value) {
      const std::uint64_t full_sequence = 2 * value - 1;
      while (slot.sequence.load(std::memory_order_acquire) != full_sequence) {
        std::this_thread::yield();
      }
      if (slot.payload != (value ^ 0x9e3779b97f4a7c15ULL)) failed.store(true);
      slot.sequence.store(full_sequence + 1, std::memory_order_release);
      handoffs.fetch_add(1, std::memory_order_relaxed);
    }
  });
  producer.join();
  consumer.join();
  if (handoffs.load(std::memory_order_relaxed) != 2 * iterations) failed.store(true);
  std::cout << "{\"iterations\":" << iterations << ",\"passed\":"
            << (failed.load() ? "false" : "true") << "}\n";
  return failed.load() ? 2 : 0;
}
