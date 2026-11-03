// ============================================================================
// Armigo Platform - Apple Video Decoding
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/host1x/video_decoder.h>

#import <CoreFoundation/CoreFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <VideoToolbox/VideoToolbox.h>

#include <cstring>
#include <mutex>

namespace armigo::host1x {
namespace {

// VP9 decoding through VideoToolbox. One session decodes pictures of one size;
// a new size (only on a key frame) starts a new session. Decoding is
// synchronous: the shown picture of a call is delivered before it returns.
class VideoToolboxVp9 final : public Vp9Decoder {
public:
    ~VideoToolboxVp9() override { release(); }

    bool decode(std::span<const std::uint8_t> picture, std::uint32_t width, std::uint32_t height,
                std::vector<std::shared_ptr<const DecodedFrame>>& shown, std::string& failure) override {
        if (width == 0 || height == 0 || width > 8192 || height > 8192) {
            failure = "VP9 picture size is not supported";
            return false;
        }
        if (!session_ || width != width_ || height != height_) {
            release();
            if (!create(width, height, failure)) return false;
        }
        CMBlockBufferRef block = nullptr;
        OSStatus status = CMBlockBufferCreateWithMemoryBlock(kCFAllocatorDefault, nullptr, picture.size(),
                                                             kCFAllocatorDefault, nullptr, 0, picture.size(),
                                                             kCMBlockBufferAssureMemoryNowFlag, &block);
        if (status == noErr)
            status = CMBlockBufferReplaceDataBytes(picture.data(), block, 0, picture.size());
        CMSampleBufferRef sample = nullptr;
        const size_t sample_size = picture.size();
        if (status == noErr)
            status = CMSampleBufferCreateReady(kCFAllocatorDefault, block, format_, 1, 0, nullptr, 1, &sample_size,
                                               &sample);
        if (block) CFRelease(block);
        if (status != noErr) {
            failure = "VP9 sample buffer could not be created (" + std::to_string(status) + ")";
            if (sample) CFRelease(sample);
            return false;
        }
        output_.clear();
        callback_status_ = noErr;
        VTDecodeInfoFlags info = 0;
        status = VTDecompressionSessionDecodeFrame(session_, sample, 0, nullptr, &info);
        if (status == noErr) status = VTDecompressionSessionWaitForAsynchronousFrames(session_);
        CFRelease(sample);
        if (status == noErr) status = callback_status_;
        if (status != noErr) {
            failure = "VP9 picture could not be decoded (" + std::to_string(status) + ")";
            return false;
        }
        for (auto& frame : output_) shown.push_back(std::move(frame));
        output_.clear();
        return true;
    }

private:
    static void deliver(void* context, void*, OSStatus status, VTDecodeInfoFlags flags, CVImageBufferRef image,
                        CMTime, CMTime) {
        auto& self = *static_cast<VideoToolboxVp9*>(context);
        if (status != noErr) {
            self.callback_status_ = status;
            return;
        }
        // A picture that is not shown produces no image.
        if (!image || (flags & kVTDecodeInfo_FrameDropped)) return;
        auto frame = std::make_shared<DecodedFrame>();
        if (!copy(image, *frame)) {
            self.callback_status_ = kVTVideoDecoderBadDataErr;
            return;
        }
        self.output_.push_back(std::move(frame));
    }

    static bool copy(CVImageBufferRef image, DecodedFrame& frame) {
        if (CVPixelBufferGetPlaneCount(image) != 2) return false;
        if (CVPixelBufferLockBaseAddress(image, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) return false;
        frame.width = static_cast<std::uint32_t>(CVPixelBufferGetWidth(image));
        frame.height = static_cast<std::uint32_t>(CVPixelBufferGetHeight(image));
        const auto luma_rows = CVPixelBufferGetHeightOfPlane(image, 0);
        const auto chroma_rows = CVPixelBufferGetHeightOfPlane(image, 1);
        frame.luma_stride = static_cast<std::uint32_t>(CVPixelBufferGetBytesPerRowOfPlane(image, 0));
        frame.chroma_stride = static_cast<std::uint32_t>(CVPixelBufferGetBytesPerRowOfPlane(image, 1));
        const auto* luma = static_cast<const std::uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(image, 0));
        const auto* chroma = static_cast<const std::uint8_t*>(CVPixelBufferGetBaseAddressOfPlane(image, 1));
        frame.luma.assign(luma, luma + frame.luma_stride * luma_rows);
        frame.chroma.assign(chroma, chroma + frame.chroma_stride * chroma_rows);
        CVPixelBufferUnlockBaseAddress(image, kCVPixelBufferLock_ReadOnly);
        return true;
    }

    bool create(std::uint32_t width, std::uint32_t height, std::string& failure) {
        static std::once_flag registered;
        std::call_once(registered, [] {
            if (@available(iOS 26.2, macOS 11.0, *))
                VTRegisterSupplementalVideoDecoderIfAvailable(kCMVideoCodecType_VP9);
        });
        // VP codec configuration: version 1, profile 0, 8-bit 4:2:0, studio
        // range, unspecified colour description, no initialisation data.
        const std::uint8_t configuration[12] = {1, 0, 0, 0, 0, 0, (8 << 4) | (1 << 1), 2, 2, 2, 0, 0};
        CFDataRef atom = CFDataCreate(kCFAllocatorDefault, configuration, sizeof configuration);
        const void* atom_keys[] = {CFSTR("vpcC")};
        const void* atom_values[] = {atom};
        CFDictionaryRef atoms = CFDictionaryCreate(kCFAllocatorDefault, atom_keys, atom_values, 1,
                                                   &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        const void* keys[] = {kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms};
        const void* values[] = {atoms};
        CFDictionaryRef extensions = CFDictionaryCreate(kCFAllocatorDefault, keys, values, 1,
                                                        &kCFTypeDictionaryKeyCallBacks,
                                                        &kCFTypeDictionaryValueCallBacks);
        OSStatus status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault, kCMVideoCodecType_VP9,
                                                         static_cast<int32_t>(width), static_cast<int32_t>(height),
                                                         extensions, &format_);
        CFRelease(extensions);
        CFRelease(atoms);
        CFRelease(atom);
        if (status != noErr) {
            failure = "VP9 format description could not be created (" + std::to_string(status) + ")";
            return false;
        }
        const SInt32 pixel_format = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
        CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &pixel_format);
        const void* attribute_keys[] = {kCVPixelBufferPixelFormatTypeKey};
        const void* attribute_values[] = {number};
        CFDictionaryRef attributes = CFDictionaryCreate(kCFAllocatorDefault, attribute_keys, attribute_values, 1,
                                                        &kCFTypeDictionaryKeyCallBacks,
                                                        &kCFTypeDictionaryValueCallBacks);
        CFRelease(number);
        VTDecompressionOutputCallbackRecord callback{deliver, this};
        status = VTDecompressionSessionCreate(kCFAllocatorDefault, format_, nullptr, attributes, &callback, &session_);
        CFRelease(attributes);
        if (status != noErr) {
            failure = "VP9 decoding session could not be created (" + std::to_string(status) + ")";
            release();
            return false;
        }
        width_ = width;
        height_ = height;
        return true;
    }

    void release() {
        if (session_) {
            VTDecompressionSessionWaitForAsynchronousFrames(session_);
            VTDecompressionSessionInvalidate(session_);
            CFRelease(session_);
            session_ = nullptr;
        }
        if (format_) {
            CFRelease(format_);
            format_ = nullptr;
        }
    }

    VTDecompressionSessionRef session_ = nullptr;
    CMVideoFormatDescriptionRef format_ = nullptr;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    OSStatus callback_status_ = noErr;
    std::vector<std::shared_ptr<const DecodedFrame>> output_;
};

} // namespace

std::unique_ptr<Vp9Decoder> make_vp9_decoder(std::string&) {
    return std::make_unique<VideoToolboxVp9>();
}

} // namespace armigo::host1x
