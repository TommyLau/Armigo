// ============================================================================
// Armigo Host1x - Multimedia Engine Command Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/common/diagnostics.h>
#include "engines.h"
#include "vp9.h"

#include <armigo/host1x/video_decoder.h>

#include <cstdio>
#include <deque>
#include <optional>

namespace armigo::host1x {
namespace {

// Codec numbers of the decoder engine's codec register.
constexpr std::uint64_t codec_vp9 = 0x9;

// The video decoder engine: methods set its registers; Execute decodes one
// picture. Shown pictures go to the frame queue with the luma surface offset
// of the execution that sent them, paired in order.
class DecoderEngine final : public CommandPusher {
public:
    DecoderEngine(Hub& hub, std::uint32_t fd, std::uint32_t syncpoint)
        : CommandPusher(hub, ClassId::video_decoder, syncpoint), fd_(fd) {
        start();
    }
    ~DecoderEngine() override { stop(); }

protected:
    bool process_method(std::uint32_t method, std::uint32_t argument, std::string& failure) override {
        if (method >= registers_.values.size()) {
            failure = "Video decoder method " + std::to_string(method) + " is out of range";
            return false;
        }
        registers_.values[method] = argument;
        if (method == DecoderRegisters::codec) return select_codec(argument, failure);
        if (method == DecoderRegisters::execute) {
            const double start = tracing() ? trace_seconds() : 0.0;
            const bool executed = execute(failure);
            if (tracing())
                armigo::diag::opt_in( "host1x %.3f decoder execution took %.2f ms\n", trace_seconds(),
                             (trace_seconds() - start) * 1000.0);
            return executed;
        }
        return true;
    }

private:
    bool select_codec(std::uint64_t codec, std::string& failure) {
        if (codec != codec_vp9) {
            failure = "Video decoder codec " + std::to_string(codec) + " is not implemented";
            return false;
        }
        if (composer_) return true;
        composer_.emplace(hub_.memory());
        decoder_ = make_vp9_decoder(failure);
        return decoder_ != nullptr;
    }

    bool execute(std::string& failure) {
        if (!composer_ || !decoder_) {
            failure = "Video decoder executed without a selected codec";
            return false;
        }
        bool hidden = false;
        if (!composer_->compose(registers_, picture_, hidden, failure)) return false;
        const auto luma = registers_.address(DecoderRegisters::luma_surfaces + 3);
        if (tracing())
            armigo::diag::opt_in( "host1x %.3f decoder %u picture %zu bytes hidden %d size %dx%d luma 0x%llx\n",
                         trace_seconds(), fd_, picture_.size(), hidden ? 1 : 0, composer_->frame_size().width,
                         composer_->frame_size().height, static_cast<unsigned long long>(luma));
        if (!hidden) pending_.push_back(luma);
        shown_.clear();
        const auto size = composer_->frame_size();
        if (!decoder_->decode(picture_, static_cast<std::uint32_t>(size.width), static_cast<std::uint32_t>(size.height),
                              shown_, failure))
            return false;
        for (auto& frame : shown_) {
            std::uint64_t offset = 0;
            if (!pending_.empty()) {
                offset = pending_.front();
                pending_.pop_front();
            }
            hub_.frames().push(fd_, offset, std::move(frame));
        }
        return true;
    }

    const std::uint32_t fd_;
    DecoderRegisters registers_{};
    std::optional<Vp9Composer> composer_;
    std::unique_ptr<Vp9Decoder> decoder_;
    std::vector<std::uint8_t> picture_;
    std::deque<std::uint64_t> pending_;
    std::vector<std::shared_ptr<const DecodedFrame>> shown_;
};

} // namespace

std::unique_ptr<CommandPusher> make_video_decoder(Hub& hub, std::uint32_t fd, std::uint32_t syncpoint) {
    return std::make_unique<DecoderEngine>(hub, fd, syncpoint);
}

} // namespace armigo::host1x
