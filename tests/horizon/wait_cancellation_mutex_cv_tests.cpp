// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/process_wide_key.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <stop_token>
#include <thread>

namespace {
using namespace std::chrono_literals;
using armigo::horizon::ProcessWideKey;

constexpr std::uint32_t waiter_bit = 0x40000000u;
std::atomic<unsigned> failures{0};

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);             \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

[[noreturn]] void bail(unsigned line, const char *what) {
  std::fprintf(stderr, "FAIL %u: %s; aborting so no worker can hang the test\n",
               line, what);
  std::fflush(stderr);
  std::abort();
}

template <typename Ready> void require_settles(unsigned line, Ready &&ready) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!ready()) {
    if (std::chrono::steady_clock::now() >= deadline)
      bail(line, "condition timed out");
    std::this_thread::yield();
  }
}

#define REQUIRE_SETTLES(condition)                                             \
  require_settles(__LINE__, [&] { return (condition); })

template <typename Result>
void require_ready(unsigned line, std::future<Result> &result) {
  if (result.wait_for(5s) != std::future_status::ready)
    bail(line, "worker timed out");
}

#define REQUIRE_READY(result) require_ready(__LINE__, result)

struct GuestMemory {
  static constexpr std::uint64_t base = 0x7200000000;
  static constexpr std::uint64_t stride = sizeof(std::uint32_t);
  std::array<std::atomic<std::uint32_t>, 4> words{};
  std::mutex gate_mutex;
  std::condition_variable gate_changed;
  std::uint64_t paused_write_address = 0;
  bool write_paused = false;
  bool resume_write = false;

  static bool index(std::uint64_t address, std::size_t &result) noexcept {
    if (address < base || (address - base) % stride != 0)
      return false;
    result = static_cast<std::size_t>((address - base) / stride);
    return result < 4;
  }

  static bool read(void *context, std::uint64_t address,
                   std::uint32_t &value) noexcept {
    auto &self = *static_cast<GuestMemory *>(context);
    std::size_t at = 0;
    if (!index(address, at))
      return false;
    value = self.words[at].load(std::memory_order_acquire);
    return true;
  }

  static bool write(void *context, std::uint64_t address,
                    std::uint32_t value) noexcept {
    auto &self = *static_cast<GuestMemory *>(context);
    std::size_t at = 0;
    if (!index(address, at))
      return false;
    self.words[at].store(value, std::memory_order_release);
    std::unique_lock lock(self.gate_mutex);
    if (self.paused_write_address == address) {
      self.write_paused = true;
      self.gate_changed.notify_all();
      self.gate_changed.wait(lock, [&] { return self.resume_write; });
    }
    return true;
  }

  static ProcessWideKey::ExchangeResult
  compare_exchange(void *context, std::uint64_t address,
                   std::uint32_t &expected, std::uint32_t desired) noexcept {
    auto &self = *static_cast<GuestMemory *>(context);
    std::size_t at = 0;
    if (!index(address, at))
      return ProcessWideKey::ExchangeResult::invalid_address;
    if (self.words[at].compare_exchange_strong(expected, desired,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire))
      return ProcessWideKey::ExchangeResult::exchanged;
    return ProcessWideKey::ExchangeResult::mismatch;
  }

  static constexpr std::uint64_t address(std::size_t at) {
    return base + at * stride;
  }

  std::uint32_t load(std::size_t at) const {
    return words[at].load(std::memory_order_acquire);
  }

  void store(std::size_t at, std::uint32_t value) {
    words[at].store(value, std::memory_order_release);
  }

  void pause_write(std::uint64_t address) {
    const std::lock_guard lock(gate_mutex);
    paused_write_address = address;
  }

  bool wait_for_write_pause() {
    std::unique_lock lock(gate_mutex);
    return gate_changed.wait_for(lock, 5s, [&] { return write_paused; });
  }

  void release_write() {
    {
      const std::lock_guard lock(gate_mutex);
      resume_write = true;
    }
    gate_changed.notify_all();
  }
};

struct State {
  GuestMemory memory;
  ProcessWideKey keys;

  ProcessWideKey::GuestWords callbacks() {
    return {GuestMemory::read, GuestMemory::write, &memory,
            GuestMemory::compare_exchange};
  }
};

void pre_cancelled_condition_wait_releases_owner() {
  State state;
  const auto mutex = GuestMemory::address(0);
  const auto key = GuestMemory::address(1);
  constexpr std::uint32_t owner = 0x1111;
  state.memory.store(0, owner);
  std::stop_source cancellation;
  CHECK(cancellation.request_stop());
  CHECK(state.keys.wait(state.callbacks(), mutex, key, owner, -1,
                        cancellation.get_token()) ==
        ProcessWideKey::WaitOutcome::cancelled);
  CHECK(state.memory.load(0) == 0);
  CHECK(state.keys.waiters(key) == 0);
  CHECK(state.keys.mutex_waiters(mutex) == 0);
}

void zero_timeout_deregisters_before_a_signal_can_select() {
  State state;
  const auto mutex = GuestMemory::address(0);
  const auto key = GuestMemory::address(1);
  constexpr std::uint32_t tag = 0x1919;
  state.memory.store(0, tag);
  state.memory.pause_write(key);
  auto waiter = std::async(std::launch::async, [&] {
    return state.keys.wait(state.callbacks(), mutex, key, tag, 0);
  });
  if (!state.memory.wait_for_write_pause())
    bail(__LINE__, "condition-word arm did not pause");
  auto signaler = std::async(std::launch::async, [&] {
    return state.keys.signal(state.callbacks(), key, 1);
  });
  CHECK(signaler.wait_for(20ms) == std::future_status::timeout);
  state.memory.release_write();
  REQUIRE_READY(waiter);
  CHECK(waiter.get() == ProcessWideKey::WaitOutcome::timed_out);
  REQUIRE_READY(signaler);
  CHECK(signaler.get() == ProcessWideKey::SignalOutcome::signaled);
  CHECK(state.keys.waiters(key) == 0);
  CHECK(state.keys.mutex_waiters(mutex) == 0);
  CHECK(state.memory.load(0) == 0);
}

void cancellation_removes_condition_waiter_after_release() {
  State state;
  const auto mutex = GuestMemory::address(0);
  const auto key = GuestMemory::address(1);
  constexpr std::uint32_t tag = 0x2222;
  state.memory.store(0, tag);
  std::stop_source cancellation;
  auto waiter = std::async(std::launch::async, [&] {
    return state.keys.wait(state.callbacks(), mutex, key, tag, -1,
                           cancellation.get_token());
  });
  REQUIRE_SETTLES(state.keys.waiters(key) == 1);
  CHECK(state.memory.load(0) == 0);
  CHECK(cancellation.request_stop());
  REQUIRE_READY(waiter);
  CHECK(waiter.get() == ProcessWideKey::WaitOutcome::cancelled);
  CHECK(state.keys.waiters(key) == 0);
  CHECK(state.keys.mutex_waiters(mutex) == 0);
  CHECK(state.memory.load(0) == 0);
}

void cancellation_removes_ordinary_mutex_waiter_without_stealing_owner() {
  State state;
  const auto mutex = GuestMemory::address(0);
  constexpr std::uint32_t owner = 0x3333;
  state.memory.store(0, owner | waiter_bit);
  std::stop_source cancellation;
  auto waiter = std::async(std::launch::async, [&] {
    return state.keys.lock(state.callbacks(), mutex, owner, 0x4444,
                           cancellation.get_token());
  });
  REQUIRE_SETTLES(state.keys.mutex_waiters(mutex) == 1);
  CHECK(cancellation.request_stop());
  REQUIRE_READY(waiter);
  CHECK(waiter.get() == ProcessWideKey::LockOutcome::cancelled);
  CHECK(state.keys.mutex_waiters(mutex) == 0);
  CHECK(state.memory.load(0) == (owner | waiter_bit));
  CHECK(state.keys.unlock(state.callbacks(), mutex));
  CHECK(state.memory.load(0) == 0);
}

void cancellation_removes_pending_condition_handoff() {
  State state;
  const auto mutex = GuestMemory::address(0);
  const auto key = GuestMemory::address(1);
  constexpr std::uint32_t waiter_tag = 0x5555;
  constexpr std::uint32_t owner = 0x6666;
  state.memory.store(0, waiter_tag);
  std::stop_source cancellation;
  auto waiter = std::async(std::launch::async, [&] {
    return state.keys.wait(state.callbacks(), mutex, key, waiter_tag, -1,
                           cancellation.get_token());
  });
  REQUIRE_SETTLES(state.keys.waiters(key) == 1);
  state.memory.store(0, owner);
  CHECK(state.keys.signal(state.callbacks(), key, 1) ==
        ProcessWideKey::SignalOutcome::signaled);
  REQUIRE_SETTLES(state.keys.mutex_waiters(mutex) == 1);
  CHECK(state.memory.load(0) == (owner | waiter_bit));
  CHECK(cancellation.request_stop());
  REQUIRE_READY(waiter);
  CHECK(waiter.get() == ProcessWideKey::WaitOutcome::cancelled);
  CHECK(state.keys.waiters(key) == 0);
  CHECK(state.keys.mutex_waiters(mutex) == 0);
  CHECK(state.memory.load(0) == (owner | waiter_bit));
  CHECK(state.keys.unlock(state.callbacks(), mutex));
  CHECK(state.memory.load(0) == 0);
}

void grant_and_cancellation_have_one_ownership_winner() {
  for (unsigned iteration = 0; iteration != 200; ++iteration) {
    State state;
    const auto mutex = GuestMemory::address(0);
    const auto key = GuestMemory::address(1);
    constexpr std::uint32_t tag = 0x7777;
    state.memory.store(0, tag);
    std::stop_source cancellation;
    auto waiter = std::async(std::launch::async, [&] {
      return state.keys.wait(state.callbacks(), mutex, key, tag, -1,
                             cancellation.get_token());
    });
    REQUIRE_SETTLES(state.keys.waiters(key) == 1);

    std::thread signaler([&] {
      CHECK(state.keys.signal(state.callbacks(), key, 1) ==
            ProcessWideKey::SignalOutcome::signaled);
    });
    std::thread canceller([&] { cancellation.request_stop(); });
    signaler.join();
    canceller.join();
    REQUIRE_READY(waiter);
    const auto outcome = waiter.get();
    CHECK((outcome == ProcessWideKey::WaitOutcome::signaled &&
           state.memory.load(0) == tag) ||
          (outcome == ProcessWideKey::WaitOutcome::cancelled &&
           state.memory.load(0) == 0));
    CHECK(state.keys.waiters(key) == 0);
    CHECK(state.keys.mutex_waiters(mutex) == 0);
  }
}
} // namespace

int main() {
  pre_cancelled_condition_wait_releases_owner();
  zero_timeout_deregisters_before_a_signal_can_select();
  cancellation_removes_condition_waiter_after_release();
  cancellation_removes_ordinary_mutex_waiter_without_stealing_owner();
  cancellation_removes_pending_condition_handoff();
  grant_and_cancellation_have_one_ownership_winner();
  std::fprintf(stderr, "Wait cancellation mutex/CV failures: %u\n",
               failures.load());
  return failures.load() == 0 ? 0 : 1;
}
