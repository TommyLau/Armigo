// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/arbitration.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stop_token>
#include <thread>

namespace {
using namespace std::chrono_literals;
using armigo::horizon::ArbitrateOutcome;
using armigo::horizon::Arbitration;

constexpr std::uint64_t address = 0x7100000000;
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

void pre_cancelled_wait_skips_transaction() {
  Arbitration arbitration;
  std::stop_source cancellation;
  CHECK(cancellation.request_stop());
  std::atomic<unsigned> checks{0};
  const auto outcome = arbitration.wait_transaction(
      address, -1,
      [](void *context) noexcept {
        static_cast<std::atomic<unsigned> *>(context)->fetch_add(
            1, std::memory_order_relaxed);
        return ArbitrateOutcome::succeeded;
      },
      &checks, cancellation.get_token());
  CHECK(outcome == ArbitrateOutcome::cancelled);
  CHECK(checks.load(std::memory_order_relaxed) == 0);
  CHECK(arbitration.waiters(address) == 0);
}

void cancellation_removes_indefinite_waiter() {
  Arbitration arbitration;
  std::stop_source cancellation;
  auto waiter = std::async(std::launch::async, [&] {
    return arbitration.wait(address, 7, 7, -1, cancellation.get_token());
  });
  REQUIRE_SETTLES(arbitration.waiters(address) == 1);
  CHECK(cancellation.request_stop());
  REQUIRE_READY(waiter);
  CHECK(waiter.get() == ArbitrateOutcome::cancelled);
  CHECK(arbitration.waiters(address) == 0);
  CHECK(arbitration.signal(address, 1) == 0);
}

void signal_and_cancellation_have_one_winner() {
  for (unsigned iteration = 0; iteration != 200; ++iteration) {
    Arbitration arbitration;
    std::stop_source cancellation;
    auto waiter = std::async(std::launch::async, [&] {
      return arbitration.wait(address, 9, 9, -1, cancellation.get_token());
    });
    REQUIRE_SETTLES(arbitration.waiters(address) == 1);

    std::size_t signaled = 0;
    std::thread signaler([&] { signaled = arbitration.signal(address, 1); });
    std::thread canceller([&] { cancellation.request_stop(); });
    signaler.join();
    canceller.join();
    REQUIRE_READY(waiter);
    const auto outcome = waiter.get();
    CHECK((signaled == 1 && outcome == ArbitrateOutcome::succeeded) ||
          (signaled == 0 && outcome == ArbitrateOutcome::cancelled));
    CHECK(arbitration.waiters(address) == 0);
  }
}
} // namespace

int main() {
  pre_cancelled_wait_skips_transaction();
  cancellation_removes_indefinite_waiter();
  signal_and_cancellation_have_one_winner();
  std::fprintf(stderr, "Wait cancellation arbitration failures: %u\n",
               failures.load());
  return failures.load() == 0 ? 0 : 1;
}
