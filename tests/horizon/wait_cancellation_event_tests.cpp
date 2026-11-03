// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/horizon/event.h>
#include <armigo/horizon/startup_queries.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stop_token>
#include <thread>

namespace {
using namespace std::chrono_literals;
using namespace armigo::horizon;

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

template <typename Result>
void require_ready(unsigned line, std::future<Result> &result) {
  if (result.wait_for(5s) != std::future_status::ready)
    bail(line, "worker timed out");
}

#define REQUIRE_READY(result) require_ready(__LINE__, result)

std::shared_ptr<ProcessRecord> make_process() {
  ProcessConfiguration configuration;
  configuration.heap_address = 0x1000000000;
  configuration.alias_address = 0x2000000000;
  configuration.stack_address = 0x6000000000;
  configuration.allowed_core_mask = 1;
  const auto built = ProcessRecord::create(configuration, {});
  CHECK(built.ok());
  return built.process;
}

void event_wait_cancels_without_changing_signal_state() {
  HandleTable table(1);
  const auto event = std::make_shared<Event>();
  const std::array handles{table.insert(event).value()};
  std::stop_source cancellation;
  auto waiter = std::async(std::launch::async, [&] {
    return wait_event(table, handles, -1, cancellation.get_token());
  });
  CHECK(waiter.wait_for(20ms) == std::future_status::timeout);
  CHECK(cancellation.request_stop());
  REQUIRE_READY(waiter);
  const auto result = waiter.get();
  CHECK(result.disposition == EventDisposition::cancelled);
  CHECK(result.result == 0 && !result.index && result.detail != nullptr);
  CHECK(!event->is_signaled());
}

void ready_event_wins_over_pre_requested_stop() {
  HandleTable table(1);
  const auto event = std::make_shared<Event>(true);
  const std::array handles{table.insert(event).value()};
  std::stop_source cancellation;
  CHECK(cancellation.request_stop());
  const auto result = wait_event(table, handles, -1, cancellation.get_token());
  CHECK(result.disposition == EventDisposition::handled);
  CHECK(result.result == 0 && result.index == 0 && result.detail == nullptr);
  CHECK(event->is_signaled());
}

void event_signal_and_cancellation_have_one_result() {
  for (unsigned iteration = 0; iteration != 100; ++iteration) {
    HandleTable table(1);
    const auto event = std::make_shared<Event>();
    const std::array handles{table.insert(event).value()};
    std::stop_source cancellation;
    auto waiter = std::async(std::launch::async, [&] {
      return wait_event(table, handles, -1, cancellation.get_token());
    });
    CHECK(waiter.wait_for(1ms) == std::future_status::timeout);
    std::thread signaler([&] { event->signal(); });
    std::thread canceller([&] { cancellation.request_stop(); });
    signaler.join();
    canceller.join();
    REQUIRE_READY(waiter);
    const auto result = waiter.get();
    CHECK(result.disposition == EventDisposition::handled ||
          result.disposition == EventDisposition::cancelled);
    CHECK((result.disposition == EventDisposition::handled &&
           result.index == 0) ||
          (result.disposition == EventDisposition::cancelled && !result.index));
  }
}

void thread_exit_wait_cancels_without_marking_exit() {
  HandleTable table(1);
  const auto thread = std::make_shared<ThreadRecord>(make_process(), 7, 32);
  const std::array handles{table.insert(thread).value()};
  std::stop_source cancellation;
  auto waiter = std::async(std::launch::async, [&] {
    return wait_event(table, handles, -1, cancellation.get_token());
  });
  CHECK(waiter.wait_for(20ms) == std::future_status::timeout);
  CHECK(cancellation.request_stop());
  REQUIRE_READY(waiter);
  const auto result = waiter.get();
  CHECK(result.disposition == EventDisposition::cancelled);
  CHECK(!result.index && !thread->exited());
  thread->mark_exited();
  CHECK(thread->wait_for_exit(0, {}) == ThreadWaitOutcome::exited);
}

void exited_thread_wins_over_pre_requested_stop() {
  HandleTable table(1);
  const auto thread = std::make_shared<ThreadRecord>(make_process(), 8, 32);
  thread->mark_exited();
  const std::array handles{table.insert(thread).value()};
  std::stop_source cancellation;
  CHECK(cancellation.request_stop());
  const auto result = wait_event(table, handles, -1, cancellation.get_token());
  CHECK(result.disposition == EventDisposition::handled);
  CHECK(result.result == 0 && result.index == 0 && result.detail == nullptr);
}

void require_active(ThreadRecord& thread) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (!thread.synchronization_wait_active()) {
    if (std::chrono::steady_clock::now() >= deadline)
      bail(__LINE__, "synchronization wait did not register");
    std::this_thread::yield();
  }
}

void guest_pending_is_one_bit_and_poll_and_signal_preserve_it() {
  HandleTable table(2);
  ThreadRecord caller(make_process(), 9, 32);
  const auto event = std::make_shared<Event>(true);
  const std::array handles{table.insert(event).value()};
  caller.cancel_synchronization();
  caller.cancel_synchronization();
  const auto ready = wait_event(table, handles, -1, caller);
  CHECK(ready.result == 0 && ready.index == 0);
  CHECK(event->reset());
  const auto poll = wait_event(table, handles, 0, caller);
  CHECK(poll.result == 0xea01 && !poll.index);
  const auto cancelled = wait_event(table, handles, -1, caller);
  CHECK(cancelled.disposition == EventDisposition::handled);
  CHECK(cancelled.result == 0xec01 && !cancelled.index);
  CHECK(wait_event(table, handles, 1000000, caller).result == 0xea01);
}


void exited_thread_and_host_stop_preserve_guest_pending() {
  HandleTable table(1);
  ThreadRecord caller(make_process(), 19, 32);
  const auto target = std::make_shared<ThreadRecord>(make_process(), 20, 32);
  const std::array handles{table.insert(target).value()};
  target->mark_exited();
  caller.cancel_synchronization();
  CHECK(wait_event(table, handles, -1, caller).index == 0);
  std::stop_source host_stop;
  CHECK(host_stop.request_stop());
  CHECK(wait_event(table, {}, -1, caller, host_stop.get_token()).disposition ==
        EventDisposition::cancelled);
  CHECK(wait_event(table, {}, -1, caller).result == 0xec01);
}

void guest_cancel_wakes_event_and_does_not_stop_host() {
  HandleTable table(2);
  const auto caller = std::make_shared<ThreadRecord>(make_process(), 10, 32);
  const auto caller_handle = table.insert(caller).value();
  const auto event = std::make_shared<Event>();
  const std::array handles{table.insert(event).value()};
  std::stop_source host_stop;
  auto waiter = std::async(std::launch::async, [&] {
    return wait_event(table, handles, -1, *caller, host_stop.get_token());
  });
  require_active(*caller);
  CHECK(cancel_synchronization(table, caller_handle, *caller).result == 0);
  REQUIRE_READY(waiter);
  const auto result = waiter.get();
  CHECK(result.disposition == EventDisposition::handled);
  CHECK(result.result == 0xec01 && result.index == 0xffffffffu);
  CHECK(!host_stop.stop_requested() && !event->is_signaled());
  CHECK(!caller->synchronization_wait_active());
  CHECK(wait_event(table, handles, 1000000, *caller).result == 0xea01);
}

void guest_cancel_wakes_thread_join_and_empty_wait() {
  HandleTable table(1);
  ThreadRecord caller(make_process(), 11, 32);
  const auto target = std::make_shared<ThreadRecord>(make_process(), 12, 32);
  const std::array handles{table.insert(target).value()};
  auto join = std::async(std::launch::async, [&] {
    return wait_event(table, handles, -1, caller);
  });
  require_active(caller);
  caller.cancel_synchronization();
  REQUIRE_READY(join);
  CHECK(join.get().result == 0xec01 && !target->exited());
  auto empty = std::async(std::launch::async, [&] {
    return wait_event(table, {}, -1, caller);
  });
  require_active(caller);
  CHECK(cancel_synchronization(table, current_thread_handle, caller).result == 0);
  REQUIRE_READY(empty);
  CHECK(empty.get().result == 0xec01);
  CHECK(wait_event(table, {}, 0, caller).result == 0xea01);
  CHECK(wait_event(table, {}, 1000000, caller).result == 0xea01);
}

void guest_cancel_validates_before_consuming_pending() {
  HandleTable table(4);
  const auto process = make_process();
  ThreadRecord caller(process, 13, 32);
  const auto event = std::make_shared<Event>(true);
  const auto event_handle = table.insert(event).value();
  const auto process_handle = table.insert(process).value();
  const auto transfer_handle = table.insert(
      std::make_shared<TransferMemoryRecord>(AddressRange{0x20000000, 4096}, 0)).value();
  const std::array invalid{Handle{0}};
  const std::array wrong_type{transfer_handle};
  const std::array process_wait{process_handle};
  const std::array later_invalid{event_handle, Handle{0}};
  const std::array multiple{event_handle, event_handle};
  std::array<Handle, 65> excessive{};
  caller.cancel_synchronization();
  CHECK(wait_event(table, excessive, -1, caller).result == 0xee01);
  CHECK(wait_event(table, invalid, -1, caller).result == 0xe401);
  CHECK(wait_event(table, wrong_type, -1, caller).result == 0xe401);
  CHECK(wait_event(table, process_wait, -1, caller).disposition == EventDisposition::unsupported);
  CHECK(wait_event(table, later_invalid, -1, caller).result == 0xe401);
  // A signaled object completes a wait at once, ahead of the pending cancellation.
  const auto multiple_result = wait_event(table, multiple, -1, caller);
  CHECK(multiple_result.result == 0 && multiple_result.index == 0u);
  CHECK(cancel_synchronization(table, 0, caller).result == 0xe401);
  CHECK(cancel_synchronization(table, event_handle, caller).result == 0xe401);
  CHECK(cancel_synchronization(table, current_process_handle, caller).result == 0xe401);
  CHECK(wait_event(table, {}, -1, caller).result == 0xec01);
}

// A wait on several objects reports the index of the first ready one: at once
// when one is already signaled, or when one is signaled (even if reset at once)
// or exits later. It times out with index -1, and guest and host cancellation
// end it.
void waits_on_several_objects() {
  HandleTable table(8);
  const auto process = make_process();
  ThreadRecord caller(process, 21, 32);
  const auto a = std::make_shared<Event>(false);
  const auto b = std::make_shared<Event>(false);
  const auto worker = std::make_shared<ThreadRecord>(process, 22, 32);
  const auto a_handle = table.insert(a).value();
  const auto b_handle = table.insert(b).value();
  const auto worker_handle = table.insert(worker).value();
  const std::array pair{a_handle, b_handle};
  b->signal();
  auto result = wait_event(table, pair, -1, caller);
  CHECK(result.result == 0 && result.index == 1u);
  b->reset();
  CHECK(wait_event(table, pair, 0, caller).result == 0xea01);
  result = wait_event(table, pair, 1000000, caller);
  CHECK(result.result == 0xea01 && result.index == 0xffffffffu);
  // Each action below happens once the wait is registered with every object.
  const auto wait_then = [&](auto handles, auto action, std::stop_token stop = {}) {
    auto waiter = std::async(std::launch::async, [&, handles, stop] {
      return wait_event(table, handles, -1, caller, stop);
    });
    require_active(caller);
    action();
    REQUIRE_READY(waiter);
    return waiter.get();
  };
  result = wait_then(pair, [&] { a->signal(); });
  CHECK(result.result == 0 && result.index == 0u);
  a->reset();
  // A signal followed at once by a reset still completes the wait.
  result = wait_then(pair, [&] {
    b->signal();
    b->reset();
  });
  CHECK(result.result == 0 && result.index == 1u);
  // The object that woke the wait is reported, not a lower index signaled later.
  result = wait_then(pair, [&] {
    b->signal();
    a->signal();
  });
  CHECK(result.result == 0 && result.index == 1u);
  a->reset();
  b->reset();
  const std::array mixed{a_handle, worker_handle};
  result = wait_then(mixed, [&] { worker->mark_exited(); });
  CHECK(result.result == 0 && result.index == 1u);
  result = wait_then(pair, [&] { caller.cancel_synchronization(); });
  CHECK(result.result == 0xec01 && result.index == 0xffffffffu);
  std::stop_source host_stop;
  result = wait_then(pair, [&] { host_stop.request_stop(); }, host_stop.get_token());
  CHECK(result.disposition == EventDisposition::cancelled);
  // Every registration is gone: later signals touch no finished waiter.
  a->signal();
  b->signal();
  CHECK(wait_event(table, pair, 0, caller).index == 0u);
}

void guest_cancel_stays_pending_outside_cancellable_wait() {
  HandleTable table(1);
  ThreadRecord caller(make_process(), 14, 32);
  const auto target = std::make_shared<ThreadRecord>(make_process(), 15, 32);
  std::stop_source host_stop;
  auto unrelated_wait = std::async(std::launch::async, [&] {
    return target->wait_for_exit(-1, host_stop.get_token());
  });
  CHECK(unrelated_wait.wait_for(20ms) == std::future_status::timeout);
  CHECK(cancel_synchronization(table, current_thread_handle, caller).result == 0);
  CHECK(unrelated_wait.wait_for(20ms) == std::future_status::timeout);
  CHECK(!host_stop.stop_requested());
  target->mark_exited();
  REQUIRE_READY(unrelated_wait);
  CHECK(unrelated_wait.get() == ThreadWaitOutcome::exited);
  CHECK(wait_event(table, {}, -1, caller).result == 0xec01);
}

void guest_wait_pins_closed_handle_and_host_stop_remains_separate() {
  HandleTable table(1);
  ThreadRecord caller(make_process(), 16, 32);
  auto event = std::make_shared<Event>();
  const std::weak_ptr<Event> lifetime = event;
  const std::array handles{table.insert(event).value()};
  std::stop_source host_stop;
  auto waiter = std::async(std::launch::async, [&] {
    return wait_event(table, handles, -1, caller, host_stop.get_token());
  });
  require_active(caller);
  event.reset();
  CHECK(table.close(handles[0]));
  CHECK(!lifetime.expired());
  CHECK(host_stop.request_stop());
  REQUIRE_READY(waiter);
  CHECK(waiter.get().disposition == EventDisposition::cancelled);
  CHECK(lifetime.expired());
  CHECK(wait_event(table, {}, 1000000, caller).result == 0xea01);
}

void signal_before_guest_cancel_preserves_pending_even_after_reset() {
  for (unsigned iteration = 0; iteration != 50; ++iteration) {
    HandleTable table(1);
    ThreadRecord caller(make_process(), 17, 32);
    const auto event = std::make_shared<Event>();
    const std::array handles{table.insert(event).value()};
    auto waiter = std::async(std::launch::async, [&] {
      return wait_event(table, handles, -1, caller);
    });
    require_active(caller);
    event->signal();
    CHECK(event->reset());
    caller.cancel_synchronization();
    REQUIRE_READY(waiter);
    const auto result = waiter.get();
    CHECK(result.result == 0 && result.index == 0);
    CHECK(wait_event(table, {}, -1, caller).result == 0xec01);
  }
}

void guest_cancel_before_signal_wins_without_consuming_signal() {
  for (unsigned iteration = 0; iteration != 50; ++iteration) {
    HandleTable table(1);
    ThreadRecord caller(make_process(), 18, 32);
    const auto event = std::make_shared<Event>();
    const std::array handles{table.insert(event).value()};
    auto waiter = std::async(std::launch::async, [&] {
      return wait_event(table, handles, -1, caller);
    });
    require_active(caller);
    caller.cancel_synchronization();
    event->signal();
    REQUIRE_READY(waiter);
    CHECK(waiter.get().result == 0xec01);
    CHECK(wait_event(table, handles, 0, caller).index == 0);
    CHECK(event->reset());
    CHECK(wait_event(table, {}, 1000000, caller).result == 0xea01);
  }
}
} // namespace

int main() {
  guest_pending_is_one_bit_and_poll_and_signal_preserve_it();
  exited_thread_and_host_stop_preserve_guest_pending();
  guest_cancel_wakes_event_and_does_not_stop_host();
  guest_cancel_wakes_thread_join_and_empty_wait();
  guest_cancel_validates_before_consuming_pending();
  waits_on_several_objects();
  guest_cancel_stays_pending_outside_cancellable_wait();
  guest_wait_pins_closed_handle_and_host_stop_remains_separate();
  signal_before_guest_cancel_preserves_pending_even_after_reset();
  guest_cancel_before_signal_wins_without_consuming_signal();
  event_wait_cancels_without_changing_signal_state();
  ready_event_wins_over_pre_requested_stop();
  event_signal_and_cancellation_have_one_result();
  thread_exit_wait_cancels_without_marking_exit();
  exited_thread_wins_over_pre_requested_stop();
  std::fprintf(stderr, "Wait cancellation event failures: %u\n",
               failures.load());
  return failures.load() == 0 ? 0 : 1;
}
