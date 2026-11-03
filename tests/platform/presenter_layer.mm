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

#include <chrono>
#include <cstdio>
#include <memory>
#include <variant>

namespace armigo::platform {
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

} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4]) {
            std::printf("SKIP: no Metal 4 device\n");
            return 0;
        }
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

    }
    if (failures) {
        std::fprintf(stderr, "%u failure(s)\n", failures);
        return 1;
    }
    std::puts("presenter layer tests passed");
    return 0;
}
