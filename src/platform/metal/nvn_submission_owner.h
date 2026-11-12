// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#import <Metal/Metal.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace armigo::platform::metal::nvn_driver {

using Metal4Completion = void (^)(id<MTL4CommitFeedback>);
using Metal4FeedbackFilter = std::function<id<MTL4CommitFeedback>(std::uint64_t, id<MTL4CommitFeedback>)>;
using UploadPublication = std::function<void(std::function<void()>)>;

// Feedback can arrive on different CPU threads even though GPU submissions
// are ordered. Publish retirement only after every preceding callback ran.
class Metal4RetirementQueue {
public:
    std::uint64_t reserve(id<MTL4CommandBuffer> commands, Metal4Completion completion) {
        std::lock_guard lock(mutex_);
        const auto ticket = ++issued_;
        pending_.push_back({ticket, commands, [completion copy], nil, false});
        return ticket;
    }

    bool attach(std::uint64_t ticket, id<MTL4CommandBuffer> commands, Metal4Completion completion) {
        std::lock_guard lock(mutex_);
        for (auto& entry : pending_)
            if (entry.ticket == ticket && !entry.gpu_ready && !entry.ready) {
                entry.commands = commands;
                entry.completion = [completion copy];
                return true;
            }
        return false;
    }

    void complete(std::uint64_t ticket, id<MTL4CommitFeedback> feedback) {
        std::unique_lock lock(mutex_);
        for (auto& entry : pending_)
            if (entry.ticket == ticket) {
                entry.feedback = feedback;
                entry.ready = true;
                entry.gpu_ready = true;
                break;
            }
        advance_gpu_locked();
        if (draining_) return;
        draining_ = true;
        while (!pending_.empty() && pending_.front().ready) {
            auto entry = std::move(pending_.front());
            pending_.pop_front();
            const auto retired = entry.ticket;
            lock.unlock();
            if (entry.completion) entry.completion(entry.feedback);
            // The ticket also covers its retained command buffer and callback
            // captures. Release them before waking a waiter that may reuse them.
            entry = {};
            lock.lock();
            retired_ = retired;
            changed_.notify_all();
        }
        draining_ = false;
    }

    bool wait(std::uint64_t ticket, std::chrono::nanoseconds timeout) {
        std::unique_lock lock(mutex_);
        if (timeout == std::chrono::nanoseconds::max()) {
            changed_.wait(lock, [&] { return retired_ >= ticket; });
            return true;
        }
        return changed_.wait_for(lock, timeout, [&] { return retired_ >= ticket; });
    }

    // Hardware observation must progress even when a CPU retirement callback
    // is waiting for a resource mutex. CPU readback can wait on this frontier;
    // resource reuse that requires bookkeeping uses the retirement frontier.
    void observe(std::uint64_t ticket) {
        std::lock_guard lock(mutex_);
        for (auto& entry : pending_)
            if (entry.ticket == ticket) { entry.gpu_ready = true; break; }
        advance_gpu_locked();
    }
    bool wait_gpu(std::uint64_t ticket, std::chrono::nanoseconds timeout) {
        std::unique_lock lock(mutex_);
        if (timeout == std::chrono::nanoseconds::max()) {
            changed_.wait(lock, [&] { return gpu_completed_ >= ticket; });
            return true;
        }
        return changed_.wait_for(lock, timeout, [&] { return gpu_completed_ >= ticket; });
    }

    [[nodiscard]] std::uint64_t retired() const {
        std::lock_guard lock(mutex_);
        return retired_;
    }

    [[nodiscard]] std::uint64_t issued() const {
        std::lock_guard lock(mutex_);
        return issued_;
    }

private:
    struct Entry {
        std::uint64_t ticket = 0;
        id<MTL4CommandBuffer> commands = nil;
        Metal4Completion completion = nil;
        id<MTL4CommitFeedback> feedback = nil;
        bool ready = false;
        bool gpu_ready = false;
    };
    void advance_gpu_locked() {
        while (!pending_.empty() && gpu_completed_ + 1 >= pending_.front().ticket) {
            const auto offset = gpu_completed_ + 1 - pending_.front().ticket;
            if (offset >= pending_.size() || !pending_[offset].gpu_ready) break;
            ++gpu_completed_;
        }
        changed_.notify_all();
    }
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Entry> pending_;
    std::uint64_t issued_ = 0, retired_ = 0;
    std::uint64_t gpu_completed_ = 0;
    bool draining_ = false;
};

// All NVN upload, replay and native-presentation publication uses this owner.
// Encoding may happen on other threads; queue operations are one transaction.
class Metal4SubmissionOwner {
public:
    // Synthetic integration tests may adapt feedback after real GPU execution.
    Metal4SubmissionOwner(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                          Metal4FeedbackFilter feedback_filter = {})
        : device_(device), queue_(queue), feedback_filter_(std::move(feedback_filter)), retirement_(std::make_shared<Metal4RetirementQueue>()),
          callbacks_(dispatch_queue_create("armigo.nvn.gpu-retirement",
              dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0))) {}

    // Context construction supplies the queue. Transparent Metal device
    // adapters need not have the same Objective-C identity as queue.device.
    [[nodiscard]] bool valid() const { return device_ && queue_; }
    [[nodiscard]] id<MTLDevice> device() const { return device_; }
    [[nodiscard]] id<MTL4CommandQueue> queue() const { return queue_; }

    // Reserve when a commit enters the CPU recording stream, rather than
    // when its encoder finally executes. A ready later native present cannot
    // overtake this position while its upload/replay record is still buffered.
    std::uint64_t reserve() {
        if (!valid()) return 0;
        std::lock_guard lock(publication_);
        return reserve_locked();
    }

    bool publish_reserved(std::uint64_t ticket, id<MTL4CommandBuffer> commands,
                          Metal4Completion completion, Metal4Completion observed = nil,
                          id<MTLDrawable> drawable = nil, bool present_drawable = true) {
        std::lock_guard lock(publication_);
        for (auto& entry : pending_)
            if (entry.ticket == ticket && !entry.ready) {
                entry.commands = commands;
                entry.completion = [completion copy];
                entry.observed = [observed copy];
                entry.drawable = drawable;
                entry.present_drawable = present_drawable;
                entry.ready = true;
                drain_locked();
                return commands != nil;
            }
        return false;
    }

    bool cancel_reserved(std::uint64_t ticket) {
        std::lock_guard lock(publication_);
        for (auto& entry : pending_)
            if (entry.ticket == ticket && !entry.ready) {
                entry.ready = true;
                drain_locked();
                return true;
            }
        return false;
    }

    // observed is only for nonblocking GPU-outcome publication, such as setting
    // an upload's atomic status. It must not acquire a resource-owner lock.
    // completion runs in submission order and may perform CPU retirement.
    // A completion must not wait for its own or a later retirement ticket.
    std::uint64_t publish(id<MTL4CommandBuffer> commands, Metal4Completion completion,
                          Metal4Completion observed = nil, id<MTLDrawable> drawable = nil,
                          bool present_drawable = true) {
        if (!valid() || !commands) return 0;
        std::lock_guard publication(publication_);
        const auto ticket = reserve_locked();
        auto& entry = pending_.back();
        entry.commands = commands;
        entry.completion = [completion copy];
        entry.observed = [observed copy];
        entry.drawable = drawable;
        entry.present_drawable = present_drawable;
        entry.ready = true;
        drain_locked();
        return ticket;
    }

    bool wait(std::uint64_t ticket, std::chrono::nanoseconds timeout) {
        return retirement_->wait(ticket, timeout);
    }
    bool wait_gpu(std::uint64_t ticket, std::chrono::nanoseconds timeout) {
        return retirement_->wait_gpu(ticket, timeout);
    }
    [[nodiscard]] std::uint64_t retired() const { return retirement_->retired(); }
    [[nodiscard]] std::uint64_t issued() const { return retirement_->issued(); }
    [[nodiscard]] std::uint64_t published() const {
        std::lock_guard lock(publication_);
        return published_;
    }

private:
    struct Publication {
        std::uint64_t ticket = 0;
        id<MTL4CommandBuffer> commands = nil;
        Metal4Completion completion = nil, observed = nil;
        id<MTLDrawable> drawable = nil;
        bool ready = false;
        bool present_drawable = true;
    };
    std::uint64_t reserve_locked() {
        const auto ticket = retirement_->reserve(nil, nil);
        pending_.push_back({ticket});
        return ticket;
    }
    void drain_locked() {
        while (!pending_.empty() && pending_.front().ready) {
            auto entry = std::move(pending_.front());
            pending_.pop_front();
            const auto ticket = entry.ticket;
            const auto retirement = retirement_;
            const auto callbacks = callbacks_;
            retirement->attach(ticket, entry.commands, entry.completion);
            if (!entry.commands) {
                retirement->observe(ticket);
                dispatch_async(callbacks, ^{ retirement->complete(ticket, nil); });
            } else {
                id<MTL4CommandQueue> queue = queue_;
                Metal4Completion observed = entry.observed;
                const auto filter = feedback_filter_;
                auto* options = [MTL4CommitOptions new];
                [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
                    static_cast<void>(queue);
                    id<MTL4CommitFeedback> effective = filter ? filter(ticket, feedback) : feedback;
                    if (observed) observed(effective);
                    retirement->observe(ticket);
                    dispatch_async(callbacks, ^{ retirement->complete(ticket, effective); });
                }];
                if (entry.drawable) [queue_ waitForDrawable:entry.drawable];
                id<MTL4CommandBuffer> submitted[]{entry.commands};
                [queue_ commit:submitted count:1 options:options];
                if (entry.drawable) {
                    [queue_ signalDrawable:entry.drawable];
                    if (entry.present_drawable) [entry.drawable present];
                }
            }
            published_ = ticket;
        }
    }
    id<MTLDevice> device_ = nil;
    id<MTL4CommandQueue> queue_ = nil;
    Metal4FeedbackFilter feedback_filter_;
    std::shared_ptr<Metal4RetirementQueue> retirement_;
    dispatch_queue_t callbacks_;
    mutable std::mutex publication_;
    std::deque<Publication> pending_;
    std::uint64_t published_ = 0;
};

// A native frame reserves its place while the NVN worker captures the actual
// allocation. The VI/UI thread supplies the drawable commands later. Dropping
// a frame must release the unfilled position even if metadata remains retained.
class Metal4NativeSubmission {
public:
    Metal4NativeSubmission(std::shared_ptr<Metal4SubmissionOwner> owner, std::uint64_t ticket,
                           std::function<std::string()> producer_error = {},
                           std::shared_ptr<const void> allocation = {},
                           std::function<void()> release_read = {})
        : owner_(std::move(owner)), ticket_(ticket), producer_error_(std::move(producer_error)),
          allocation_(std::move(allocation)), release_read_(std::move(release_read)) {}
    ~Metal4NativeSubmission() { cancel(); }
    std::uint64_t publish(id<MTL4CommandBuffer> commands, Metal4Completion completion,
                          id<MTLDrawable> drawable = nil, bool defer_presentation = false) {
        std::lock_guard lock(mutex_);
        if (!owner_ || cancelled_) return 0;
        const auto release = ticket_ ? release_read_ : std::function<void()>{};
        const auto allocation = allocation_;
        Metal4Completion retired = ^(id<MTL4CommitFeedback> feedback) {
            static_cast<void>(allocation);
            if (completion) completion(feedback);
            if (release) release();
        };
        if (!ticket_) return owner_->publish(commands, retired, nil, drawable, !defer_presentation);
        return owner_->publish_reserved(ticket_, commands, retired, nil, drawable, !defer_presentation) ? ticket_ : 0;
    }
    // Decoded pixels own their input and use the presenter's queue. A fresh
    // native ticket could sit behind the next frame's unfilled reservation
    // while that frame waits for this synchronous consumer to return.
    bool prepare_decoded_submission() {
        std::lock_guard lock(mutex_);
        if (!owner_ || cancelled_) return false;
        if (ticket_ && owner_->cancel_reserved(ticket_) && release_read_) release_read_();
        return true;
    }
    // The queue has already signaled this drawable at the consumer position.
    // Visibility waits until ordered retirement validated the producer result.
    bool present_deferred_drawable(id<MTLDrawable> drawable) {
        std::lock_guard lock(mutex_);
        if (cancelled_ || !drawable) return false;
        [drawable present];
        return true;
    }
    [[nodiscard]] bool cancelled() {
        std::lock_guard lock(mutex_);
        return cancelled_;
    }
    void cancel() {
        std::lock_guard lock(mutex_);
        if (!ticket_) return;
        cancelled_ = true;
        if (owner_ && owner_->cancel_reserved(ticket_) && release_read_) release_read_();
    }
    bool wait_gpu(std::uint64_t ticket, std::chrono::nanoseconds timeout) {
        return owner_ && owner_->wait_gpu(ticket, timeout);
    }
    bool wait(std::uint64_t ticket, std::chrono::nanoseconds timeout) {
        return owner_ && owner_->wait(ticket, timeout);
    }
    bool wait_retirement(std::chrono::nanoseconds timeout) {
        return owner_ && owner_->wait(ticket_, timeout);
    }
    [[nodiscard]] std::string producer_error() const { return producer_error_ ? producer_error_() : std::string{}; }
private:
    std::mutex mutex_;
    bool cancelled_ = false;
    std::shared_ptr<Metal4SubmissionOwner> owner_;
    std::uint64_t ticket_ = 0;
    std::function<std::string()> producer_error_;
    std::shared_ptr<const void> allocation_;
    std::function<void()> release_read_;
};

} // namespace armigo::platform::metal::nvn_driver
