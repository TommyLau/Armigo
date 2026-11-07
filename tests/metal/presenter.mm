// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic coverage for the headless Metal presenter
// contract: explicit failures, exact readback after real GPU completion,
// resource retention for timed-out or cancelled inflight work, and GPU
// consumption of unread native frames through reused presenter storage.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <armigo/platform/presenter.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <thread>
#include <variant>
#include <vector>

// Internal test controls defined in src/platform/metal/presenter.mm.
namespace armigo::platform {
void presenter_test_set_feedback_delay_ns(std::uint64_t nanoseconds);
std::size_t presenter_test_inflight_count(const Presenter& presenter);
std::uint64_t presenter_test_presented_count(const Presenter& presenter);
id<MTLTexture> presenter_test_native_target(const Presenter& presenter);
std::uint64_t presenter_test_native_target_allocations(const Presenter& presenter);
std::uint64_t presenter_test_native_residency_allocations(const Presenter& presenter);
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

std::chrono::steady_clock::time_point far_deadline() {
    return std::chrono::steady_clock::now() + std::chrono::seconds(10);
}

graphics::Rgba8Image gradient_image(std::uint32_t width, std::uint32_t height) {
    graphics::Rgba8Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(std::size_t{width} * height * 4);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            auto* pixel = image.pixels.data() + (std::size_t{y} * width + x) * 4;
            pixel[0] = static_cast<std::byte>((x * 7) & 0xFF);
            pixel[1] = static_cast<std::byte>((y * 11) & 0xFF);
            pixel[2] = static_cast<std::byte>(((x + y) * 3) & 0xFF);
            // Intermediate alpha values catch accidental premultiplication.
            pixel[3] = static_cast<std::byte>(((x * y) ^ 0x5A) & 0xFF);
        }
    }
    return image;
}

PresentationFrame make_frame(const graphics::Rgba8Image& image, std::uint64_t frame_number = 0) {
    PresentationFrame frame;
    frame.queued.frame_number = frame_number;
    frame.pixels = std::make_shared<const graphics::Rgba8Image>(image);
    return frame;
}

bool has_error(const PresentResult& result, PresentationError code) {
    const auto* failure = std::get_if<PresentationFailure>(&result);
    return failure && failure->code == code;
}

void test_factory_rejects_windowed() {
    PresenterOptions options;
    options.mode = PresentationMode::windowed;
    const PresenterCreation creation = create_presenter(options);
    const auto* failure = std::get_if<PresentationFailure>(&creation);
    CHECK(failure && failure->code == PresentationError::unsupported);
}

void test_invalid_frames_fail(Presenter& presenter) {
    const PresentationFrame missing;
    CHECK(has_error(presenter.present(missing, far_deadline(), {}), PresentationError::invalid_frame));

    graphics::Rgba8Image empty;
    empty.width = 0;
    empty.height = 8;
    empty.pixels.assign(32, std::byte{0});
    CHECK(has_error(presenter.present(make_frame(empty), far_deadline(), {}), PresentationError::invalid_frame));

    graphics::Rgba8Image truncated = gradient_image(4, 4);
    truncated.pixels.pop_back();
    CHECK(has_error(presenter.present(make_frame(truncated), far_deadline(), {}), PresentationError::invalid_frame));
}

void test_unsupported_metadata_fails(Presenter& presenter) {
    graphics::Rgba8Image image = gradient_image(4, 4);
    PresentationFrame frame = make_frame(image);
    frame.queued.transform = 0x10;
    CHECK(has_error(presenter.present(frame, far_deadline(), {}), PresentationError::unsupported));

    frame = make_frame(image);
    frame.queued.scaling_mode = 5;
    CHECK(has_error(presenter.present(frame, far_deadline(), {}), PresentationError::unsupported));

    // Beyond the 16384 color-target extent policy; validated before any
    // allocation, so the oversized byte count is never issued to Metal.
    graphics::Rgba8Image oversized;
    oversized.width = 16385;
    oversized.height = 1;
    oversized.pixels.resize(std::size_t{oversized.width} * 4);
    CHECK(has_error(presenter.present(make_frame(oversized), far_deadline(), {}), PresentationError::unsupported));
}

void test_headless_readback_matches_input(Presenter& presenter) {
    const graphics::Rgba8Image image = gradient_image(37, 23);
    const PresentResult result = presenter.present(make_frame(image, 41), far_deadline(), {});
    const auto* presented = std::get_if<PresentedFrame>(&result);
    CHECK(presented);
    if (!presented) return;
    CHECK(presented->frame_number == 41);
    CHECK(presented->completed_at.time_since_epoch().count() != 0);
    CHECK(presented->readback.has_value());
    if (!presented->readback) return;
    CHECK(presented->readback->width == image.width);
    CHECK(presented->readback->height == image.height);
    CHECK(presented->readback->pixels.size() == image.pixels.size());
    CHECK(std::memcmp(presented->readback->pixels.data(), image.pixels.data(), image.pixels.size()) == 0);
}

void test_timeout_retains_resources_until_drain(Presenter& presenter) {
    presenter_test_set_feedback_delay_ns(400'000'000);
    const graphics::Rgba8Image image = gradient_image(8, 8);
    const auto expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(60);
    const PresentResult result = presenter.present(make_frame(image, 7), expiry, {});
    CHECK(has_error(result, PresentationError::deadline_expired));

    // The frame stays inflight with its pixels, staging, target and readback
    // storage; the GPU work is still running and must not be released.
    CHECK(presenter_test_inflight_count(presenter) == 1);

    // A short drain still reports the deadline while work is inflight.
    const auto short_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(30);
    const PresenterStatus partial = presenter.drain(short_deadline);
    CHECK(std::holds_alternative<PresentationFailure>(partial));
    CHECK(presenter_test_inflight_count(presenter) == 1);

    // Only a successful drain retires the frame; this waits for real GPU
    // completion instead of assuming the timeout proved anything.
    const auto drained = presenter.drain(std::nullopt);
    CHECK(std::holds_alternative<std::monostate>(drained));
    CHECK(presenter_test_inflight_count(presenter) == 0);
    presenter_test_set_feedback_delay_ns(0);
}

void test_cancellation_retains_resources_until_drain(Presenter& presenter) {
    presenter_test_set_feedback_delay_ns(400'000'000);
    std::stop_source stop;
    std::atomic<bool> cancelled = false;
    std::jthread canceller([&stop, &cancelled] {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        stop.request_stop();
        cancelled = true;
    });
    const graphics::Rgba8Image image = gradient_image(8, 8);
    const PresentResult result = presenter.present(make_frame(image, 9), far_deadline(), stop.get_token());
    // The stop request can wake present before the helper stores its flag.
    canceller.join();
    CHECK(has_error(result, PresentationError::cancelled));
    CHECK(cancelled.load());
    CHECK(presenter_test_inflight_count(presenter) == 1);

    const auto drained = presenter.drain(std::nullopt);
    CHECK(std::holds_alternative<std::monostate>(drained));
    CHECK(presenter_test_inflight_count(presenter) == 0);
    presenter_test_set_feedback_delay_ns(0);
}

void test_presenter_stays_usable_after_drain(Presenter& presenter) {
    const graphics::Rgba8Image image = gradient_image(5, 5);
    const PresentResult result = presenter.present(make_frame(image, 12), far_deadline(), {});
    const auto* presented = std::get_if<PresentedFrame>(&result);
    CHECK(presented && presented->frame_number == 12);
    if (!presented) return;
    CHECK(presented->readback.has_value());
    CHECK(std::memcmp(presented->readback->pixels.data(), image.pixels.data(), image.pixels.size()) == 0);
    CHECK(presenter_test_inflight_count(presenter) == 0);
}

// A producer texture on the presenter's device holding the image's bytes.
id<MTLTexture> native_texture(MTLPixelFormat format, const graphics::Rgba8Image& image) {
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                          width:image.width
                                                                         height:image.height
                                                                      mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    id<MTLTexture> texture = [MTLCreateSystemDefaultDevice() newTextureWithDescriptor:descriptor];
    if (texture)
        [texture replaceRegion:MTLRegionMake2D(0, 0, image.width, image.height) mipmapLevel:0
                     withBytes:image.pixels.data() bytesPerRow:image.width * 4];
    return texture;
}

// Reads a private texture back through a blit after GPU completion.
std::vector<std::byte> texture_bytes(id<MTLTexture> texture) {
    const NSUInteger width = texture.width;
    const NSUInteger height = texture.height;
    id<MTLCommandQueue> queue = [texture.device newCommandQueue];
    id<MTLBuffer> buffer = [texture.device newBufferWithLength:width * height * 4
                                                       options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [commands blitCommandEncoder];
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(width, height, 1) toBuffer:buffer destinationOffset:0
      destinationBytesPerRow:width * 4 destinationBytesPerImage:width * height * 4];
    [blit endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    if (commands.status != MTLCommandBufferStatusCompleted) return {};
    const auto* bytes = static_cast<const std::byte*>(buffer.contents);
    return {bytes, bytes + width * height * 4};
}

// The crop's rows of a tightly packed image.
std::vector<std::byte> crop_bytes(const graphics::Rgba8Image& image, std::array<std::int32_t, 4> crop) {
    std::vector<std::byte> bytes;
    for (std::int32_t y = crop[1]; y < crop[3]; ++y) {
        const auto* row = image.pixels.data() + (std::size_t(y) * image.width + std::size_t(crop[0])) * 4;
        bytes.insert(bytes.end(), row, row + std::size_t(crop[2] - crop[0]) * 4);
    }
    return bytes;
}

PresentationFrame unread_frame(id<MTLTexture> texture, std::uint64_t frame_number,
                               std::array<std::int32_t, 4> crop = {}) {
    PresentationFrame frame;
    frame.queued.frame_number = frame_number;
    frame.queued.native_texture = std::shared_ptr<const void>(
        (__bridge_retained const void*)texture, [](const void* object) { CFRelease(object); });
    // Producers queue the texture's own extent.
    frame.queued.width = static_cast<std::uint32_t>(texture.width);
    frame.queued.height = static_cast<std::uint32_t>(texture.height);
    frame.queued.crop = crop;
    frame.readback = false;
    return frame;
}

bool presented_without_readback(const PresentResult& result, std::uint64_t frame_number) {
    const auto* presented = std::get_if<PresentedFrame>(&result);
    return presented && presented->frame_number == frame_number && !presented->readback &&
           presented->completed_at.time_since_epoch().count() != 0;
}

void test_unread_native_frames_reuse_presenter_storage(Presenter& presenter) {
    const graphics::Rgba8Image first_image = gradient_image(37, 23);
    graphics::Rgba8Image second_image = gradient_image(37, 23);
    for (auto& value : second_image.pixels) value = ~value;
    id<MTLTexture> first = native_texture(MTLPixelFormatRGBA8Unorm, first_image);
    id<MTLTexture> second = native_texture(MTLPixelFormatRGBA8Unorm, second_image);
    CHECK(first && second);
    if (!first || !second) return;
    const auto targets = presenter_test_native_target_allocations(presenter);
    const auto sets = presenter_test_native_residency_allocations(presenter);

    // The GPU copies the whole texture into the presenter's target; nothing
    // is read back.
    // A queued extent that is not the texture's is refused, as the read-back
    // path refuses pixels of another size.
    auto mismatched = unread_frame(first, 50);
    ++mismatched.queued.width;
    const auto refused = presenter.present(mismatched, far_deadline(), {});
    CHECK(std::holds_alternative<PresentationFailure>(refused) &&
          std::get<PresentationFailure>(refused).code == PresentationError::invalid_frame);
    CHECK(presented_without_readback(presenter.present(unread_frame(first, 51), far_deadline(), {}), 51));
    id<MTLTexture> target = presenter_test_native_target(presenter);
    CHECK(target && target.width == 37 && target.height == 23);
    CHECK(target && target.pixelFormat == MTLPixelFormatRGBA8Unorm);
    CHECK(target && texture_bytes(target) == first_image.pixels);
    CHECK(presenter_test_native_target_allocations(presenter) == targets + 1);
    CHECK(presenter_test_native_residency_allocations(presenter) == sets + 1);
    CHECK(presenter_test_inflight_count(presenter) == 0);

    // A repeated texture allocates nothing; a second producer texture only
    // extends the residency set, and cycling between the two reuses it.
    CHECK(presented_without_readback(presenter.present(unread_frame(first, 52), far_deadline(), {}), 52));
    CHECK(presented_without_readback(presenter.present(unread_frame(second, 53), far_deadline(), {}), 53));
    CHECK(target && texture_bytes(target) == second_image.pixels);
    CHECK(presented_without_readback(presenter.present(unread_frame(first, 54), far_deadline(), {}), 54));
    CHECK(presented_without_readback(presenter.present(unread_frame(second, 55), far_deadline(), {}), 55));
    CHECK(presenter_test_native_target(presenter) == target);
    CHECK(presenter_test_native_target_allocations(presenter) == targets + 1);
    CHECK(presenter_test_native_residency_allocations(presenter) == sets + 2);
    CHECK(target && texture_bytes(target) == second_image.pixels);

    // A new crop extent reallocates the target, which holds exactly the crop.
    const std::array<std::int32_t, 4> crop{3, 2, 13, 9};
    CHECK(presented_without_readback(presenter.present(unread_frame(first, 56, crop), far_deadline(), {}), 56));
    id<MTLTexture> cropped = presenter_test_native_target(presenter);
    CHECK(cropped && cropped != target && cropped.width == 10 && cropped.height == 7);
    CHECK(cropped && texture_bytes(cropped) == crop_bytes(first_image, crop));
    CHECK(presenter_test_native_target_allocations(presenter) == targets + 2);
    CHECK(presenter_test_native_residency_allocations(presenter) == sets + 3);

    // So does a new pixel format; the copy keeps the stored bytes.
    graphics::Rgba8Image bgra_image = gradient_image(10, 7);
    id<MTLTexture> bgra = native_texture(MTLPixelFormatBGRA8Unorm, bgra_image);
    CHECK(presented_without_readback(presenter.present(unread_frame(bgra, 57), far_deadline(), {}), 57));
    id<MTLTexture> reformatted = presenter_test_native_target(presenter);
    CHECK(reformatted && reformatted != cropped && reformatted.pixelFormat == MTLPixelFormatBGRA8Unorm);
    CHECK(reformatted && texture_bytes(reformatted) == bgra_image.pixels);
    CHECK(presenter_test_native_target_allocations(presenter) == targets + 3);
    CHECK(presenter_test_inflight_count(presenter) == 0);
}

void test_unread_native_frame_failures(Presenter& presenter) {
    const graphics::Rgba8Image image = gradient_image(8, 8);
    id<MTLTexture> texture = native_texture(MTLPixelFormatRGBA8Unorm, image);
    CHECK(texture);
    if (!texture) return;
    // A frame that asks for readback must carry decoded pixels.
    PresentationFrame wanted = unread_frame(texture, 61);
    wanted.readback = true;
    CHECK(has_error(presenter.present(wanted, far_deadline(), {}), PresentationError::invalid_frame));
    CHECK(has_error(presenter.present(unread_frame(texture, 62, {0, 0, 9, 8}), far_deadline(), {}),
                    PresentationError::invalid_frame));
    PresentationFrame transformed = unread_frame(texture, 63);
    transformed.queued.transform = 0x10;
    CHECK(has_error(presenter.present(transformed, far_deadline(), {}), PresentationError::unsupported));
    CHECK(presenter_test_inflight_count(presenter) == 0);
}

void test_reallocation_keeps_inflight_target(Presenter& presenter) {
    const graphics::Rgba8Image image = gradient_image(16, 16);
    id<MTLTexture> texture = native_texture(MTLPixelFormatRGBA8Unorm, image);
    CHECK(texture);
    if (!texture) return;
    presenter_test_set_feedback_delay_ns(400'000'000);
    const auto expiry = std::chrono::steady_clock::now() + std::chrono::milliseconds(60);
    CHECK(has_error(presenter.present(unread_frame(texture, 71), expiry, {}), PresentationError::deadline_expired));
    CHECK(presenter_test_inflight_count(presenter) == 1);
    __weak id<MTLTexture> inflight_target = nil;
    @autoreleasepool {
        inflight_target = presenter_test_native_target(presenter);
    }
    // A new extent replaces the presenter's target while the first copy may
    // still run; the inflight record keeps the old one alive.
    CHECK(presented_without_readback(
        presenter.present(unread_frame(texture, 72, {0, 0, 8, 8}), far_deadline(), {}), 72));
    @autoreleasepool {
        id<MTLTexture> retained = inflight_target;
        CHECK(retained != nil);
        CHECK(retained != presenter_test_native_target(presenter));
    }
    CHECK(presenter_test_inflight_count(presenter) == 1);
    presenter_test_set_feedback_delay_ns(0);
    CHECK(std::holds_alternative<std::monostate>(presenter.drain(std::nullopt)));
    CHECK(presenter_test_inflight_count(presenter) == 0);
}

} // namespace

int main() {
    @autoreleasepool {
        const PresenterCreation creation = create_presenter(PresenterOptions{});
        auto* failure = std::get_if<PresentationFailure>(&creation);
        if (failure) {
            std::fprintf(stderr, "SKIP: headless presenter unavailable: %s\n", failure->detail.c_str());
            return 1;
        }
        const std::shared_ptr<Presenter>& presenter = std::get<std::shared_ptr<Presenter>>(creation);
        CHECK(presenter->mode() == PresentationMode::headless);

        test_factory_rejects_windowed();
        test_invalid_frames_fail(*presenter);
        test_unsupported_metadata_fails(*presenter);
        test_headless_readback_matches_input(*presenter);
        test_timeout_retains_resources_until_drain(*presenter);
        test_cancellation_retains_resources_until_drain(*presenter);
        test_presenter_stays_usable_after_drain(*presenter);
        test_unread_native_frames_reuse_presenter_storage(*presenter);
        test_unread_native_frame_failures(*presenter);
        test_reallocation_keeps_inflight_target(*presenter);
        CHECK(presenter_test_presented_count(*presenter) == 0);
    }
    if (failures) {
        std::fprintf(stderr, "%u presenter checks failed\n", failures);
        return 1;
    }
    std::printf("presenter checks passed\n");
    return 0;
}
