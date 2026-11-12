// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic coverage for the windowed presenter when
// its layer has nothing to draw into: a frame presented to a layer with no
// drawable-sized surface is dropped and still completes instead of ending the
// run. A layer that returns no drawable (a minimized or hidden window) takes
// the same path; a layer outside a window never runs out of drawables, so
// that case is not reproducible here. The layer is never attached to a window.

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <mach/mach.h>
#include <thread>

#include <armigo/platform/presenter.h>
#include "../../src/platform/metal/nvn_submission_owner.h"
#include "../metal/commit_feedback.h"
#include <atomic>
#include <future>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <variant>

static std::atomic<unsigned> native_visible{0}, native_early_visible{0};
static std::atomic<bool> native_checked{false};

@interface NativeDrawableProbe : NSProxy
@property(nonatomic,strong) id<CAMetalDrawable> target;
@end
@implementation NativeDrawableProbe
- (NSMethodSignature*)methodSignatureForSelector:(SEL)selector { return [(NSObject*)self.target methodSignatureForSelector:selector]; }
- (void)forwardInvocation:(NSInvocation*)invocation { [invocation invokeWithTarget:self.target]; }
- (void)present { if(!native_checked.load()) ++native_early_visible; ++native_visible; [self.target present]; }
@end
@interface NativeLayerProbe : CAMetalLayer
@end
@implementation NativeLayerProbe
- (id<CAMetalDrawable>)nextDrawable {
    auto target=[super nextDrawable]; if(!target)return nil;
    auto probe=[NativeDrawableProbe alloc]; probe.target=target;
    return (id<CAMetalDrawable>)probe;
}
@end
@interface NativeQueueProbe : NSProxy
@property(nonatomic,strong) id<MTL4CommandQueue> target;
@property(nonatomic) NSUInteger sequence, waits, commits, signals, waitSequence, commitSequence, signalSequence;
@property(nonatomic,copy) void (^onCommit)(void);
@end
@implementation NativeQueueProbe
- (NSMethodSignature*)methodSignatureForSelector:(SEL)selector { return [(NSObject*)self.target methodSignatureForSelector:selector]; }
- (void)forwardInvocation:(NSInvocation*)invocation { [invocation invokeWithTarget:self.target]; }
- (void)waitForDrawable:(id<MTLDrawable>)drawable {
    self.waits += 1; self.waitSequence = ++self.sequence;
    [self.target waitForDrawable:((NativeDrawableProbe*)drawable).target];
}
- (void)commit:(const id<MTL4CommandBuffer>*)commands count:(NSUInteger)count options:(MTL4CommitOptions*)options {
    self.commits += 1; self.commitSequence = ++self.sequence;
    if (self.onCommit) self.onCommit();
    [self.target commit:commands count:count options:options];
}
- (void)signalDrawable:(id<MTLDrawable>)drawable {
    self.signals += 1; self.signalSequence = ++self.sequence;
    [self.target signalDrawable:((NativeDrawableProbe*)drawable).target];
}
@end

namespace armigo::platform {
bool presenter_test_set_queue(Presenter& presenter, id<MTL4CommandQueue> queue);
std::shared_ptr<Presenter> create_layer_presenter(const PresenterOptions& options, CAMetalLayer* layer,
                                                  PresentationFailure* failure_out);
} // namespace armigo::platform

namespace {

using namespace armigo::platform;
namespace graphics = armigo::graphics;

unsigned failures = 0;

#define CHECK(...)                                                          \
    do {                                                                    \
        if (!(__VA_ARGS__)) {                                               \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #__VA_ARGS__); \
            ++failures;                                                     \
        }                                                                   \
    } while (false)

PresentationFrame solid_frame(std::uint64_t number) {
    graphics::Rgba8Image image;
    image.width = 16;
    image.height = 9;
    image.pixels.assign(std::size_t{image.width} * image.height * 4, std::byte{0x80});
    PresentationFrame frame;
    frame.queued.frame_number = number;
    frame.pixels = std::make_shared<const graphics::Rgba8Image>(std::move(image));
    frame.readback = false;
    return frame;
}

void check_dropped(Presenter& presenter, std::uint64_t number) {
    const PresentResult result =
        presenter.present(solid_frame(number), std::chrono::steady_clock::now() + std::chrono::seconds(10), {});
    if (const auto* failed = std::get_if<PresentationFailure>(&result))
        std::fprintf(stderr, "present failure: code=%d detail=%s\n", int(failed->code), failed->detail.c_str());
    const auto* presented = std::get_if<PresentedFrame>(&result);
    CHECK(presented);
    if (presented) {
        CHECK(presented->frame_number == number);
        CHECK(!presented->readback.has_value());
    }
}

std::uint64_t phys_footprint() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
        return 0;
    return info.phys_footprint;
}

// The runtime presents from a plain thread with no autorelease pool. A windowed
// present must drain what Metal and Core Animation autorelease for it (the
// drawable and encoders, which hold the frame's command allocator and surface
// state), or every frame's command memory stays alive for the thread's life.
void check_frames_release_resources(Presenter& presenter, CAMetalLayer* layer) {
    layer.drawableSize = CGSizeMake(64, 36);
    std::uint64_t before = 0, after = 0;
    unsigned presented = 0;
    std::thread worker([&] {
        const auto present = [&](std::uint64_t number) {
            const PresentResult result = presenter.present(
                solid_frame(number), std::chrono::steady_clock::now() + std::chrono::seconds(10), {});
            if (std::holds_alternative<PresentedFrame>(result)) ++presented;
        };
        for (std::uint64_t number = 100; number != 120; ++number) present(number);
        before = phys_footprint();
        for (std::uint64_t number = 120; number != 420; ++number) present(number);
        after = phys_footprint();
    });
    worker.join();
    CHECK(presented == 320);
    CHECK(before != 0 && after < before + (std::uint64_t{64} << 20));
    if (after >= before + (std::uint64_t{64} << 20))
        std::fprintf(stderr, "footprint grew %llu MB over 300 frames\n",
                     static_cast<unsigned long long>((after - before) >> 20));
}

// Real GPU work is queued through the owner; the proxy observes only the
// visible present call and unwraps its real drawable for Metal validation.
void check_ordered_visibility(id<MTLDevice> device, unsigned mode, bool readback, bool dropped) {
    using namespace armigo::platform::metal::nvn_driver;
    native_visible=0; native_early_visible=0; native_checked=false;
    auto layer=[NativeLayerProbe layer]; layer.drawableSize=CGSizeMake(dropped?0:16,dropped?0:9);
    PresenterOptions options; options.mode=PresentationMode::windowed; options.width=16; options.height=9;
    PresentationFailure error;
    auto presenter=create_layer_presenter(options,layer,&error); CHECK(presenter!=nullptr); if(!presenter)return;
    auto proxy=[NativeQueueProbe alloc]; proxy.target=[device newMTL4CommandQueue];
    auto queue=(id<MTL4CommandQueue>)proxy;
    auto owner=std::make_shared<Metal4SubmissionOwner>(device,queue,
        [mode](std::uint64_t ticket,id<MTL4CommitFeedback> feedback)->id<MTL4CommitFeedback> {
            return mode==1 && ticket==1 ? failed_commit_feedback(feedback) : feedback;
        });
    auto gate=[device newSharedEvent]; [queue waitForEvent:gate value:1];
    auto producer_done=std::make_shared<std::atomic<bool>>(false), producer_failed=std::make_shared<std::atomic<bool>>(false);
    auto allocator=[device newCommandAllocator]; auto commands=[device newCommandBuffer];
    [commands beginCommandBufferWithAllocator:allocator]; [commands endCommandBuffer];
    owner->publish(commands, ^(id<MTL4CommitFeedback> feedback) {
        static_cast<void>(allocator); producer_failed->store(feedback.error!=nil); producer_done->store(true);
    });
    auto descriptor=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:16 height:9 mipmapped:NO];
    descriptor.storageMode=MTLStorageModeShared; descriptor.usage=MTLTextureUsageShaderRead;
    auto source=[device newTextureWithDescriptor:descriptor];
    std::array<std::byte,16*9*4> pixels{};
    [source replaceRegion:MTLRegionMake2D(0,0,16,9) mipmapLevel:0 withBytes:pixels.data() bytesPerRow:64];
    auto allocation=std::shared_ptr<const void>((__bridge_retained const void*)source,[](const void* p){CFRelease(p);});
    auto native=std::make_shared<Metal4NativeSubmission>(owner,owner->reserve(),
        [mode]{return mode==2?std::string{"Synthetic upload failure"}:std::string{};},allocation);
    PresentationFrame frame; frame.queued.frame_number=31; frame.queued.width=16; frame.queued.height=9;
    frame.queued.native_texture=allocation; frame.queued.native_submission=native; frame.readback=readback;
    frame.queued.native_producer_ready=[producer_done,producer_failed] {
        native_checked=true; return producer_done->load() && !producer_failed->load();
    };
    std::stop_source stop;
    auto result=std::async(std::launch::async,[&] { return presenter->present(frame,std::chrono::steady_clock::now()+std::chrono::seconds(5),stop.get_token()); });
    const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while(owner->published()<2 && std::chrono::steady_clock::now()<until) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(owner->published()>=2 && native_visible==0 && !producer_done->load());
    CHECK(result.wait_for(std::chrono::milliseconds(10))==std::future_status::timeout);
    if(mode==3) {
        stop.request_stop();
        CHECK(result.wait_for(std::chrono::seconds(1))==std::future_status::ready);
        CHECK(std::get<PresentationFailure>(result.get()).code==PresentationError::cancelled);
    }
    gate.signaledValue=1;
    if(mode!=3) {
        CHECK(result.wait_for(std::chrono::seconds(3))==std::future_status::ready);
        auto value=result.get();
        if(mode) CHECK(std::holds_alternative<PresentationFailure>(value));
        else {
            CHECK(std::holds_alternative<PresentedFrame>(value));
            if(readback && !dropped && std::holds_alternative<PresentedFrame>(value)) CHECK(std::get<PresentedFrame>(value).readback.has_value());
        }
    }
    CHECK(owner->wait(owner->issued(),std::chrono::seconds(3)));
    CHECK(std::holds_alternative<std::monostate>(presenter->drain(std::chrono::steady_clock::now()+std::chrono::seconds(3))));
    CHECK(native_early_visible==0 && native_visible==(mode==0 && !dropped?1u:0u));
}

// CPU-owned decoded pixels use the presenter's queue even when they retain
// native validation metadata. The proxy forwards every real Metal operation.
void check_decoded_native_queue_order(id<MTLDevice> device, unsigned mode) {
    using namespace armigo::platform::metal::nvn_driver;
    native_visible = 0; native_early_visible = 0; native_checked = false;
    auto layer = [NativeLayerProbe layer];
    layer.drawableSize = CGSizeMake(16, 9);
    PresenterOptions options; options.mode = PresentationMode::windowed; options.width = 16; options.height = 9;
    PresentationFailure error;
    auto presenter = create_layer_presenter(options, layer, &error);
    CHECK(presenter != nullptr);
    if (!presenter) return;
    auto queue = [NativeQueueProbe alloc]; queue.target = [device newMTL4CommandQueue];
    CHECK(presenter_test_set_queue(*presenter, (id<MTL4CommandQueue>)queue));
    auto owner = std::make_shared<Metal4SubmissionOwner>(device, [device newMTL4CommandQueue]);
    auto native = std::make_shared<Metal4NativeSubmission>(owner, owner->reserve(), [mode] {
        return mode == 2 ? std::string{"Synthetic decoded upload failure"} : std::string{};
    });
    auto frame = solid_frame(500 + mode);
    frame.readback = true;
    frame.queued.native_submission = native;
    frame.queued.native_producer_ready = [mode] { native_checked = true; return mode != 1; };
    if (mode == 3) queue.onCommit = ^{ native->cancel(); };
    auto result = presenter->present(frame, std::chrono::steady_clock::now() + std::chrono::seconds(5), {});
    std::fprintf(stderr, "decoded_native_queue_evidence mode=%u waits=%lu commits=%lu signals=%lu order=%lu/%lu/%lu visible=%u early=%u owner_issued=%llu\n",
        mode, static_cast<unsigned long>(queue.waits), static_cast<unsigned long>(queue.commits),
        static_cast<unsigned long>(queue.signals), static_cast<unsigned long>(queue.waitSequence),
        static_cast<unsigned long>(queue.commitSequence), static_cast<unsigned long>(queue.signalSequence),
        native_visible.load(), native_early_visible.load(), static_cast<unsigned long long>(owner->issued()));
    CHECK(queue.waits == 1 && queue.commits == 1 && queue.signals == 1 &&
          queue.waitSequence < queue.commitSequence && queue.commitSequence < queue.signalSequence);
    CHECK(owner->issued() == 1);
    CHECK(native_early_visible == 0 && native_visible == (mode == 0 ? 1u : 0u));
    if (mode) {
        CHECK(std::holds_alternative<PresentationFailure>(result));
        if (std::holds_alternative<PresentationFailure>(result)) {
            const auto& failure = std::get<PresentationFailure>(result);
            CHECK(failure.code == (mode == 3 ? PresentationError::cancelled : PresentationError::host_failure));
            if (mode == 2) CHECK(failure.detail == "Synthetic decoded upload failure");
        }
    } else {
        CHECK(std::holds_alternative<PresentedFrame>(result));
        if (std::holds_alternative<PresentedFrame>(result)) {
            const auto& completed = std::get<PresentedFrame>(result);
            CHECK(completed.frame_number == frame.queued.frame_number && completed.readback.has_value());
            if (completed.readback) CHECK(completed.readback->pixels == frame.pixels->pixels);
        }
    }
    CHECK(owner->wait(owner->issued(), std::chrono::seconds(3)));
    CHECK(std::holds_alternative<std::monostate>(presenter->drain(std::chrono::steady_clock::now() + std::chrono::seconds(3))));
}

} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::printf("SKIP: no Metal 4 device\n");
            return 0;
        }
        if (std::getenv("ARMIGO_DECODED_NATIVE_TESTS")) {
            for (unsigned mode = 0; mode != 4; ++mode) check_decoded_native_queue_order(device, mode);
            return failures ? 1 : 0;
        }
        for (unsigned mode = 0; mode != 4; ++mode) check_decoded_native_queue_order(device, mode);
        CAMetalLayer* layer = [CAMetalLayer layer];
        PresenterOptions options;
        options.mode = PresentationMode::windowed;
        options.width = 16;
        options.height = 9;
        PresentationFailure failure;
        std::shared_ptr<Presenter> presenter = create_layer_presenter(options, layer, &failure);
        if (!presenter) std::fprintf(stderr, "presenter: %s\n", failure.detail.c_str());
        CHECK(presenter);
        if (!presenter) return 1;

        // No drawable-sized surface (a collapsed or not yet laid out view).
        layer.drawableSize = CGSizeMake(0, 0);
        check_dropped(*presenter, 3);

        check_frames_release_resources(*presenter, layer);
        for(unsigned mode=0;mode!=4;++mode) for(bool readback:{false,true})
            check_ordered_visibility(device,mode,readback,false);
        check_ordered_visibility(device,0,false,true);
        check_ordered_visibility(device,1,false,true);
        check_ordered_visibility(device,3,false,true);

    }
    if (failures) {
        std::fprintf(stderr, "%u failure(s)\n", failures);
        return 1;
    }
    std::puts("presenter layer tests passed");
    return 0;
}
