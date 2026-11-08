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

#include <string>

namespace armigo::host1x {
namespace {

// Hardware AV1 decoding into NV12 frames. The session is made from the first
// temporal unit's sequence header and remade when a later one differs.
class VideoToolboxAv1 final : public Av1Decoder {
public:
    ~VideoToolboxAv1() override { release(); }

    bool decode(std::span<const std::uint8_t> temporal_unit, std::vector<std::shared_ptr<const DecodedFrame>>& shown,
                std::string& failure) override {
        if (const auto record = av1_configuration_record(temporal_unit); record && *record != record_) {
            release();
            if (!create(*record, failure)) return false;
        }
        if (!session_) {
            failure = "AV1 stream has no sequence header before its first picture";
            return false;
        }
        CMBlockBufferRef block = nullptr;
        OSStatus status = CMBlockBufferCreateWithMemoryBlock(kCFAllocatorDefault, nullptr, temporal_unit.size(),
                                                             kCFAllocatorDefault, nullptr, 0, temporal_unit.size(),
                                                             kCMBlockBufferAssureMemoryNowFlag, &block);
        if (status == noErr)
            status = CMBlockBufferReplaceDataBytes(temporal_unit.data(), block, 0, temporal_unit.size());
        CMSampleBufferRef sample = nullptr;
        const size_t sample_size = temporal_unit.size();
        if (status == noErr)
            status = CMSampleBufferCreateReady(kCFAllocatorDefault, block, format_, 1, 0, nullptr, 1, &sample_size,
                                               &sample);
        if (block) CFRelease(block);
        if (status != noErr) {
            failure = "AV1 sample buffer could not be created (" + std::to_string(status) + ")";
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
            failure = "AV1 temporal unit could not be decoded (" + std::to_string(status) + ")";
            return false;
        }
        for (auto& frame : output_) shown.push_back(std::move(frame));
        output_.clear();
        return true;
    }

private:
    static void deliver(void* context, void*, OSStatus status, VTDecodeInfoFlags flags, CVImageBufferRef image,
                        CMTime, CMTime) {
        auto& self = *static_cast<VideoToolboxAv1*>(context);
        if (status != noErr) {
            self.callback_status_ = status;
            return;
        }
        // A frame that is not shown produces no image.
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
    bool create(const std::vector<std::uint8_t>& record, std::string& failure) {
        CFDataRef atom = CFDataCreate(kCFAllocatorDefault, record.data(), static_cast<CFIndex>(record.size()));
        const void* atom_keys[] = {CFSTR("av1C")};
        const void* atom_values[] = {atom};
        CFDictionaryRef atoms = CFDictionaryCreate(kCFAllocatorDefault, atom_keys, atom_values, 1,
                                                   &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
        const void* keys[] = {kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms};
        const void* values[] = {atoms};
        CFDictionaryRef extensions = CFDictionaryCreate(kCFAllocatorDefault, keys, values, 1,
                                                        &kCFTypeDictionaryKeyCallBacks,
                                                        &kCFTypeDictionaryValueCallBacks);
        // The picture size comes from the sequence header the decoder reads;
        // the description's size is only a hint.
        OSStatus status = CMVideoFormatDescriptionCreate(kCFAllocatorDefault, kCMVideoCodecType_AV1, 16, 16,
                                                         extensions, &format_);
        CFRelease(extensions);
        CFRelease(atoms);
        CFRelease(atom);
        if (status != noErr) {
            failure = "AV1 format description could not be created (" + std::to_string(status) + ")";
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
            failure = "AV1 decoding session could not be created (" + std::to_string(status) + ")";
            release();
            return false;
        }
        record_ = record;
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
        record_.clear();
    }
    VTDecompressionSessionRef session_ = nullptr;
    CMVideoFormatDescriptionRef format_ = nullptr;
    std::vector<std::uint8_t> record_;
    OSStatus callback_status_ = noErr;
    std::vector<std::shared_ptr<const DecodedFrame>> output_;
};

} // namespace

std::unique_ptr<Av1Decoder> make_av1_decoder(std::string& failure) {
    if (!VTIsHardwareDecodeSupported(kCMVideoCodecType_AV1)) {
        failure = "This Mac has no hardware AV1 decoder";
        return nullptr;
    }
    return std::make_unique<VideoToolboxAv1>();
}

} // namespace armigo::host1x
