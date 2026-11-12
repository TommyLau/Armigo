// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "nvn_pass_dump.h"

#include <armigo/common/diagnostics.h>

#import <CoreGraphics/CoreGraphics.h>
#import <ImageIO/ImageIO.h>

#include <TargetConditionals.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace armigo::platform::metal::nvn_driver {
namespace {

constexpr std::uint32_t kImageLimit = 400;
constexpr std::uint64_t kByteLimit = 300ull * 1024ull * 1024ull;

constexpr char kDumpSource[] = R"msl(
#include <metal_stdlib>
using namespace metal;

struct DumpArgs {
    uint src_w;
    uint src_h;
    uint dst_w;
    uint dst_h;
    uint mode;
    uint scale;
    uint depth;
};

inline uint2 source_coord(constant DumpArgs& args, uint2 gid) {
    uint x1 = (gid.x + 1) * args.src_w / args.dst_w;
    uint y1 = (gid.y + 1) * args.src_h / args.dst_h;
    uint x0 = gid.x * args.src_w / args.dst_w;
    uint y0 = gid.y * args.src_h / args.dst_h;
    if (x1 <= x0) x1 = x0 + 1;
    if (y1 <= y0) y1 = y0 + 1;
    return uint2(min(args.src_w - 1, (x0 + x1) / 2), min(args.src_h - 1, (y0 + y1) / 2));
}

inline uchar4 pack_rgb(float3 rgb) {
    float3 saturated = saturate(rgb);
    return uchar4(uchar(saturated.r * 255.0 + 0.5), uchar(saturated.g * 255.0 + 0.5),
                  uchar(saturated.b * 255.0 + 0.5), 255);
}

inline float3 shade_float(float3 rgb, uint mode, device float* raw, uint index) {
    if (mode == 1 || mode == 3) {
        float3 clamped = clamp(rgb, float3(0.0), float3(1e10));
        raw[index] = rgb.r;
        raw[index + 1] = rgb.g;
        raw[index + 2] = rgb.b;
        return clamped / (float3(1.0) + clamped);
    }
    if (mode == 2) return rgb * 0.5 + 0.5;
    return rgb;
}

kernel void armigo_pass_dump_float(texture2d<float, access::read> src [[texture(0)]],
                                   device uchar4* pixels [[buffer(0)]],
                                   device float* raw [[buffer(1)]],
                                   constant DumpArgs& args [[buffer(2)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.dst_w || gid.y >= args.dst_h) return;
    float4 color = src.read(source_coord(args, gid));
    uint index = (gid.y * args.dst_w + gid.x) * 3;
    pixels[gid.y * args.dst_w + gid.x] = pack_rgb(shade_float(color.rgb, args.mode, raw, index));
}

kernel void armigo_pass_dump_buffer(device const uchar* source [[buffer(0)]],
                                    device uchar* destination [[buffer(1)]],
                                    constant uint& bytes [[buffer(2)]],
                                    uint index [[thread_position_in_grid]]) {
    if (index < bytes) destination[index] = source[index];
}

kernel void armigo_pass_dump_volume(texture3d<float, access::read> src [[texture(0)]],
                                    device uchar4* pixels [[buffer(0)]],
                                    device float* raw [[buffer(1)]],
                                    constant DumpArgs& args [[buffer(2)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.dst_w || gid.y >= args.dst_h * args.depth) return;
    uint2 xy = source_coord(args, uint2(gid.x, gid.y % args.dst_h));
    float3 color = src.read(uint3(xy, gid.y / args.dst_h)).rgb;
    uint index = (gid.y * args.dst_w + gid.x) * 3;
    pixels[gid.y * args.dst_w + gid.x] = pack_rgb(shade_float(color, 3, raw, index));
}

kernel void armigo_pass_dump_uint(texture2d<uint, access::read> src [[texture(0)]],
                                  device uchar4* pixels [[buffer(0)]],
                                  device float* raw [[buffer(1)]],
                                  constant DumpArgs& args [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.dst_w || gid.y >= args.dst_h) return;
    uint4 value = src.read(source_coord(args, gid));
    float scale = max(float(args.scale), 1.0);
    float3 rgb = float3(float(value.r), float(value.g), float(value.b)) / scale;
    pixels[gid.y * args.dst_w + gid.x] = pack_rgb(rgb);
    if (args.mode == 3) {
        uint index = (gid.y * args.dst_w + gid.x) * 3;
        raw[index] = float(value.r); raw[index + 1] = float(value.g); raw[index + 2] = float(value.b);
    }
}

kernel void armigo_pass_dump_int(texture2d<int, access::read> src [[texture(0)]],
                                 device uchar4* pixels [[buffer(0)]],
                                 device float* raw [[buffer(1)]],
                                 constant DumpArgs& args [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.dst_w || gid.y >= args.dst_h) return;
    int4 value = src.read(source_coord(args, gid));
    float scale = max(float(args.scale), 1.0);
    float3 rgb = float3(float(value.r), float(value.g), float(value.b)) / scale;
    pixels[gid.y * args.dst_w + gid.x] = pack_rgb(rgb);
    if (args.mode == 3) {
        uint index = (gid.y * args.dst_w + gid.x) * 3;
        raw[index] = float(value.r); raw[index + 1] = float(value.g); raw[index + 2] = float(value.b);
    }
}

kernel void armigo_pass_dump_depth(depth2d<float, access::read> src [[texture(0)]],
                                   device uchar4* pixels [[buffer(0)]],
                                   device float* raw [[buffer(1)]],
                                   constant DumpArgs& args [[buffer(2)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.dst_w || gid.y >= args.dst_h) return;
    float depth = saturate(src.read(source_coord(args, gid)));
    pixels[gid.y * args.dst_w + gid.x] = pack_rgb(float3(depth));
    if (args.mode == 3) {
        uint index = (gid.y * args.dst_w + gid.x) * 3;
        raw[index] = depth; raw[index + 1] = depth; raw[index + 2] = depth;
    }
}
)msl";

struct DumpArgs {
    std::uint32_t src_w, src_h, dst_w, dst_h, mode, scale, depth;
};

struct DumpFormat {
    enum class Pipe : std::uint8_t { skip, floating, unsign, sign, depth };
    Pipe pipe = Pipe::skip;
    std::uint32_t mode = 0;
    std::uint32_t scale = 1;
    const char* name = nullptr;
};

DumpFormat unorm(const char* name) { return {DumpFormat::Pipe::floating, 0, 1, name}; }
DumpFormat hdr(const char* name) { return {DumpFormat::Pipe::floating, 1, 1, name}; }
DumpFormat snorm(const char* name) { return {DumpFormat::Pipe::floating, 2, 1, name}; }
DumpFormat unsign(const char* name, std::uint32_t scale) { return {DumpFormat::Pipe::unsign, 0, scale, name}; }
DumpFormat sign(const char* name, std::uint32_t scale) { return {DumpFormat::Pipe::sign, 0, scale, name}; }
DumpFormat depth_format(const char* name) { return {DumpFormat::Pipe::depth, 0, 1, name}; }

DumpFormat classify(MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatA8Unorm: return unorm("A8Unorm");
    case MTLPixelFormatR8Unorm: return unorm("R8Unorm");
    case MTLPixelFormatR8Unorm_sRGB: return unorm("R8Unorm_sRGB");
    case MTLPixelFormatR8Snorm: return snorm("R8Snorm");
    case MTLPixelFormatR8Uint: return unsign("R8Uint", 255);
    case MTLPixelFormatR8Sint: return sign("R8Sint", 127);
    case MTLPixelFormatRG8Unorm: return unorm("RG8Unorm");
    case MTLPixelFormatRG8Unorm_sRGB: return unorm("RG8Unorm_sRGB");
    case MTLPixelFormatRG8Snorm: return snorm("RG8Snorm");
    case MTLPixelFormatRG8Uint: return unsign("RG8Uint", 255);
    case MTLPixelFormatRG8Sint: return sign("RG8Sint", 127);
    case MTLPixelFormatRGBA8Unorm: return unorm("RGBA8Unorm");
    case MTLPixelFormatRGBA8Unorm_sRGB: return unorm("RGBA8Unorm_sRGB");
    case MTLPixelFormatRGBA8Snorm: return snorm("RGBA8Snorm");
    case MTLPixelFormatRGBA8Uint: return unsign("RGBA8Uint", 255);
    case MTLPixelFormatRGBA8Sint: return sign("RGBA8Sint", 127);
    case MTLPixelFormatBGRA8Unorm: return unorm("BGRA8Unorm");
    case MTLPixelFormatBGRA8Unorm_sRGB: return unorm("BGRA8Unorm_sRGB");
    case MTLPixelFormatRGB10A2Unorm: return unorm("RGB10A2Unorm");
    case MTLPixelFormatRGB10A2Uint: return unsign("RGB10A2Uint", 1023);
    case MTLPixelFormatRG11B10Float: return hdr("RG11B10Float");
    case MTLPixelFormatRGB9E5Float: return hdr("RGB9E5Float");
    case MTLPixelFormatB5G6R5Unorm: return unorm("B5G6R5Unorm");
    case MTLPixelFormatA1BGR5Unorm: return unorm("A1BGR5Unorm");
    case MTLPixelFormatABGR4Unorm: return unorm("ABGR4Unorm");
    case MTLPixelFormatBGR5A1Unorm: return unorm("BGR5A1Unorm");
    case MTLPixelFormatR16Unorm: return unorm("R16Unorm");
    case MTLPixelFormatR16Snorm: return snorm("R16Snorm");
    case MTLPixelFormatR16Uint: return unsign("R16Uint", 65535);
    case MTLPixelFormatR16Sint: return sign("R16Sint", 32767);
    case MTLPixelFormatR16Float: return hdr("R16Float");
    case MTLPixelFormatRG16Unorm: return unorm("RG16Unorm");
    case MTLPixelFormatRG16Snorm: return snorm("RG16Snorm");
    case MTLPixelFormatRG16Uint: return unsign("RG16Uint", 65535);
    case MTLPixelFormatRG16Sint: return sign("RG16Sint", 32767);
    case MTLPixelFormatRG16Float: return hdr("RG16Float");
    case MTLPixelFormatRGBA16Unorm: return unorm("RGBA16Unorm");
    case MTLPixelFormatRGBA16Snorm: return snorm("RGBA16Snorm");
    case MTLPixelFormatRGBA16Uint: return unsign("RGBA16Uint", 65535);
    case MTLPixelFormatRGBA16Sint: return sign("RGBA16Sint", 32767);
    case MTLPixelFormatRGBA16Float: return hdr("RGBA16Float");
    case MTLPixelFormatR32Uint: return unsign("R32Uint", 4294967295u);
    case MTLPixelFormatR32Sint: return sign("R32Sint", 2147483647u);
    case MTLPixelFormatR32Float: return hdr("R32Float");
    case MTLPixelFormatRG32Uint: return unsign("RG32Uint", 4294967295u);
    case MTLPixelFormatRG32Sint: return sign("RG32Sint", 2147483647u);
    case MTLPixelFormatRG32Float: return hdr("RG32Float");
    case MTLPixelFormatRGBA32Uint: return unsign("RGBA32Uint", 4294967295u);
    case MTLPixelFormatRGBA32Sint: return sign("RGBA32Sint", 2147483647u);
    case MTLPixelFormatRGBA32Float: return hdr("RGBA32Float");
    case MTLPixelFormatDepth16Unorm: return depth_format("Depth16Unorm");
    case MTLPixelFormatDepth32Float: return depth_format("Depth32Float");
    case MTLPixelFormatDepth32Float_Stencil8: return depth_format("Depth32Float_Stencil8");
#if !TARGET_OS_IPHONE
    case MTLPixelFormatDepth24Unorm_Stencil8: return depth_format("Depth24Unorm_Stencil8");
#endif
    default: return {};
    }
}

bool parse_u64(std::string_view text, std::uint64_t& out) {
    if (text.empty()) return false;
    const char* begin = text.data();
    const char* end = begin + text.size();
    auto [pointer, error] = std::from_chars(begin, end, out);
    return error == std::errc{} && pointer == end;
}

dispatch_queue_t pass_dump_queue() {
    static dispatch_queue_t queue = dispatch_queue_create("armigo.nvn.pass-dump", DISPATCH_QUEUE_SERIAL);
    return queue;
}

std::string index_text(std::uint32_t index) {
    char buffer[16];
    if (index < 10000) std::snprintf(buffer, sizeof buffer, "%04u", index);
    else std::snprintf(buffer, sizeof buffer, "%u", index);
    return buffer;
}

void retain_allocation(id<MTLResidencySet> residency, NSMutableArray<id>* retained, id<MTLAllocation> allocation) {
    if (!allocation) return;
    if (residency) [residency addAllocation:allocation];
    if (retained) [retained addObject:allocation];
}

void retain_texture(id<MTLResidencySet> residency, NSMutableArray<id>* retained, id<MTLTexture> texture) {
    for (id<MTLTexture> current = texture; current; current = current.parentTexture)
        retain_allocation(residency, retained, current);
}

bool write_png(const std::filesystem::path& path, const std::uint8_t* rgba, std::uint32_t width,
               std::uint32_t height) {
    if (!rgba || !width || !height) return false;
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithData(
        nullptr, rgba, static_cast<std::size_t>(width) * height * 4, nullptr);
    const auto bitmap = static_cast<CGBitmapInfo>(kCGBitmapByteOrder32Big) |
                        static_cast<CGBitmapInfo>(kCGImageAlphaLast);
    CGImageRef image = CGImageCreate(width, height, 8, 32, width * 4, space, bitmap, provider, nullptr, false,
                                     kCGRenderingIntentDefault);
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault, reinterpret_cast<const UInt8*>(path.c_str()),
        static_cast<CFIndex>(path.string().size()), false);
    CGImageDestinationRef destination = url ? CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, nullptr)
                                            : nullptr;
    bool wrote = false;
    if (destination && image) {
        CGImageDestinationAddImage(destination, image, nullptr);
        wrote = CGImageDestinationFinalize(destination);
    }
    if (destination) CFRelease(destination);
    if (url) CFRelease(url);
    if (image) CGImageRelease(image);
    if (provider) CGDataProviderRelease(provider);
    if (space) CGColorSpaceRelease(space);
    return wrote;
}

std::string image_stats(const PassDumpImage& image) {
    if (!image.pixels || !image.dst_w || !image.dst_h || !image.pixels.contents)
        return image.kind + ":" + (image.note.empty() ? "dump:skipped" : image.note);
    const auto count = static_cast<std::uint64_t>(image.dst_w) * image.dst_h;
    const auto* bytes = static_cast<const std::uint8_t*>(image.pixels.contents);
    std::uint64_t red = 0, green = 0, blue = 0;
    unsigned min_luma = 255, max_luma = 0;
    for (std::uint64_t pixel = 0; pixel < count; ++pixel) {
        const unsigned r = bytes[pixel * 4];
        const unsigned g = bytes[pixel * 4 + 1];
        const unsigned b = bytes[pixel * 4 + 2];
        red += r;
        green += g;
        blue += b;
        const unsigned luma = (54u * r + 183u * g + 19u * b) >> 8;
        min_luma = std::min(min_luma, luma);
        max_luma = std::max(max_luma, luma);
    }
    const auto mean = [&](std::uint64_t sum) { return static_cast<unsigned>((sum + count / 2) / count); };
    std::string raw = "-";
    if (image.hdr && image.raw && image.raw.contents) {
        const auto* samples = static_cast<const float*>(image.raw.contents);
        double sums[3]{};
        float minima[3]{INFINITY, INFINITY, INFINITY}, maxima[3]{-INFINITY, -INFINITY, -INFINITY};
        std::uint64_t finite[3]{}, nan = 0, inf = 0;
        for (std::uint64_t pixel = 0; pixel < count; ++pixel)
            for (unsigned channel = 0; channel != 3; ++channel) {
                const auto value = samples[pixel * 3 + channel];
                if (std::isnan(value)) { ++nan; continue; }
                if (std::isinf(value)) { ++inf; continue; }
                sums[channel] += value;
                ++finite[channel];
                minima[channel] = std::min(minima[channel], value);
                maxima[channel] = std::max(maxima[channel], value);
            }
        char formatted[320];
        std::snprintf(formatted, sizeof formatted,
            "%.6g,%.6g,%.6g:min:%.6g,%.6g,%.6g:max:%.6g,%.6g,%.6g:nan:%llu:inf:%llu",
            sums[0] / std::max<std::uint64_t>(1, finite[0]),
            sums[1] / std::max<std::uint64_t>(1, finite[1]),
            sums[2] / std::max<std::uint64_t>(1, finite[2]),
            minima[0], minima[1], minima[2], maxima[0], maxima[1], maxima[2],
            static_cast<unsigned long long>(nan), static_cast<unsigned long long>(inf));
        raw = formatted;
    }
    char text[512];
    std::snprintf(text, sizeof text, "%s:mean:%u,%u,%u:lum:%u..%u:raw:%s", image.kind.c_str(), mean(red),
                  mean(green), mean(blue), min_luma, max_luma, raw.c_str());
    return text;
}

std::string line_text(const PassDumpLine& line) {
    std::string images;
    for (const auto& image : line.images) {
        if (!images.empty()) images += ' ';
        if (!image.pixels) images += image.kind + ":" + (image.note.empty() ? "dump:skipped" : image.note);
        else images += image_stats(image);
    }
    if (images.empty()) images = "-";
    std::string text;
    text += "p=" + index_text(line.index);
    text += line.compute ? " kind=compute" : " kind=render";
    text += " why=" + line.why;
    text += " draws=" + std::to_string(line.draws);
    text += " vs=" + (line.vs.empty() ? "-" : line.vs);
    text += " fs=" + (line.fs.empty() ? "-" : line.fs);
    text += " cs=" + (line.cs.empty() ? "-" : line.cs);
    text += " attachments=" + (line.attachments.empty() ? "-" : line.attachments);
    text += " images=" + images;
    return text;
}

void warn_once(bool& warned, const char* message) {
    if (warned) return;
    warned = true;
    armigo::diag::write(armigo::diag::Level::warning, "%s", message);
}

bool accept_metadata_text(const std::string& directory, std::size_t bytes) {
    struct Budget { std::uint64_t bytes = 0; bool warned = false; };
    static std::mutex mutex;
    static std::map<std::string, Budget> budgets;
    const std::lock_guard lock(mutex);
    auto& budget = budgets[directory];
    if (bytes > kByteLimit || budget.bytes > kByteLimit - bytes) {
        warn_once(budget.warned, "Armigo NVN: metadata-only pass dump reached its 300 MiB text limit\n");
        return false;
    }
    budget.bytes += bytes;
    return true;
}

void write_pass_dumps(const std::vector<PassDumpLine>& lines) {
    static bool write_warned = false;
    for (const auto& line : lines) {
        std::error_code error;
        std::filesystem::create_directories(line.directory, error);
        if (error) {
            warn_once(write_warned, "Armigo NVN: pass dump could not create its output directory\n");
            continue;
        }
        const auto directory = std::filesystem::path(line.directory);
        for (const auto& image : line.images) {
            if (image.note.starts_with("buffer:gpu_address:") && image.raw && image.raw.contents) {
                const auto name = "f" + std::to_string(line.frame) + "-p" + index_text(line.index) + "-" + image.kind + ".bin";
                std::ofstream output(directory / name, std::ios::binary);
                output.write(static_cast<const char*>(image.raw.contents), image.raw.length);
                if (!output) warn_once(write_warned, "Armigo NVN: pass dump could not write a buffer snapshot\n");
            }
            if (!image.pixels || !image.dst_w || !image.dst_h || !image.pixels.contents) continue;
            const auto name = "f" + std::to_string(line.frame) + "-p" + index_text(line.index) + "-" + image.kind +
                              "-" + std::to_string(image.serial) + ".png";
            if (!write_png(directory / name, static_cast<const std::uint8_t*>(image.pixels.contents), image.dst_w,
                           image.dst_h))
                warn_once(write_warned, "Armigo NVN: pass dump could not write a PNG\n");
        }
        std::string text = line_text(line) + '\n';
        for (const auto& read : line.reads)
            text += "read p=" + index_text(line.index) + ' ' + read + '\n';
        for (const auto& draw : line.draw_metadata)
            text += "draw p=" + index_text(line.index) + ' ' + draw + '\n';
        if (line.metadata_only && !accept_metadata_text(line.directory, text.size())) continue;
        const auto path = directory / ("f" + std::to_string(line.frame) + "-passes.txt");
        std::ofstream output(path, std::ios::app);
        output << text;
        output.flush();
        if (!output) warn_once(write_warned, "Armigo NVN: pass dump could not write its pass list\n");
    }
}

} // namespace

PassDumpConfig read_pass_dump_config() {
    PassDumpConfig config;
    const char* range = std::getenv("ARMIGO_NVN_PASS_DUMP");
    const char* directory = std::getenv("ARMIGO_NVN_PASS_DUMP_DIR");
    if (!range || !*range) return config;
    if (!directory || !*directory) {
        armigo::diag::write(armigo::diag::Level::warning,
                            "Armigo NVN: pass dump needs ARMIGO_NVN_PASS_DUMP_DIR\n");
        return config;
    }
    const std::string_view text(range);
    const auto dash = text.find('-');
    std::uint64_t first = 0, last = 0;
    const bool parsed = dash == std::string_view::npos
                            ? parse_u64(text, first) && (last = first, true)
                            : parse_u64(text.substr(0, dash), first) && parse_u64(text.substr(dash + 1), last) &&
                                  last >= first;
    if (!parsed) {
        armigo::diag::write(armigo::diag::Level::warning,
                            "Armigo NVN: pass dump ignored an unreadable frame selection\n");
        return config;
    }
    config.enabled = true;
    config.first = first;
    config.last = last;
    config.directory = directory;
    if (const char* inputs = std::getenv("ARMIGO_NVN_PASS_DUMP_INPUTS")) config.inputs = *inputs == '1';
    if (const char* metadata = std::getenv("ARMIGO_NVN_PASS_DUMP_METADATA_ONLY")) config.metadata_only = *metadata == '1';
    if (config.metadata_only && config.last - config.first >= 8192) {
        armigo::diag::write(armigo::diag::Level::warning,
                            "Armigo NVN: metadata-only pass dump exceeds 8192 selected frames\n");
        config.enabled = false;
        return config;
    }
    if (const char* size = std::getenv("ARMIGO_NVN_PASS_DUMP_DRAW_SIZE"); size && *size) {
        const std::string_view dimensions(size);
        const auto separator = dimensions.find('x');
        std::uint64_t width = 0, height = 0;
        if (separator != std::string_view::npos && parse_u64(dimensions.substr(0, separator), width) &&
            parse_u64(dimensions.substr(separator + 1), height) && width && height &&
            width <= std::numeric_limits<std::uint32_t>::max() &&
            height <= std::numeric_limits<std::uint32_t>::max()) {
            config.draw_width = static_cast<std::uint32_t>(width);
            config.draw_height = static_cast<std::uint32_t>(height);
        } else {
            armigo::diag::write(armigo::diag::Level::warning,
                                "Armigo NVN: pass dump ignored an unreadable draw size\n");
        }
    }
    if (const char* width = std::getenv("ARMIGO_NVN_PASS_DUMP_WIDTH"); width && *width) {
        std::uint64_t parsed_width = 0;
        if (parse_u64(width, parsed_width) && parsed_width >= 1 &&
            parsed_width <= std::numeric_limits<std::uint32_t>::max())
            config.width = static_cast<std::uint32_t>(parsed_width);
    }
    return config;
}

PassDump::PassDump(id<MTLDevice> device, PassDumpConfig config) : device_(device), config_(std::move(config)) {}

PassDump::~PassDump() = default;

bool PassDump::ensure_directory() {
    if (stopped_) return false;
    if (directory_ready_) return true;
    std::error_code error;
    std::filesystem::create_directories(config_.directory, error);
    if (error || config_.directory.empty()) {
        stopped_ = true;
        armigo::diag::write(armigo::diag::Level::warning,
                            "Armigo NVN: pass dump cannot create its output directory\n");
        return false;
    }
    directory_ready_ = true;
    return true;
}

std::uint32_t PassDump::next_index(std::uint64_t frame) {
    if (!index_valid_ || frame != index_frame_) {
        index_valid_ = true;
        index_frame_ = frame;
        inputs_seen_.clear();
        index_ = 0;
    }
    return index_++;
}

bool PassDump::pipelines() {
    if (pipelines_ready_) return pipelines_ok_;
    pipelines_ready_ = true;
    NSError* error = nil;
    id<MTLLibrary> library = [device_ newLibraryWithSource:@(kDumpSource) options:nil error:&error];
    const auto make = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> function = [library newFunctionWithName:name];
        if (!function) return nil;
        return [device_ newComputePipelineStateWithFunction:function error:&error];
    };
    float_pipeline_ = make(@"armigo_pass_dump_float");
    volume_pipeline_ = make(@"armigo_pass_dump_volume");
    copy_pipeline_ = make(@"armigo_pass_dump_buffer");
    uint_pipeline_ = make(@"armigo_pass_dump_uint");
    int_pipeline_ = make(@"armigo_pass_dump_int");
    depth_pipeline_ = make(@"armigo_pass_dump_depth");
    pipelines_ok_ = float_pipeline_ && volume_pipeline_ && copy_pipeline_ && uint_pipeline_ && int_pipeline_ && depth_pipeline_;
    if (!pipelines_ok_)
        armigo::diag::write(armigo::diag::Level::warning,
                            "Armigo NVN: pass dump kernels are unavailable; images are not written\n");
    return pipelines_ok_;
}

id<MTLBuffer> PassDump::scratch() {
    if (!scratch_) scratch_ = [device_ newBufferWithLength:16 options:MTLResourceStorageModeShared];
    return scratch_;
}

bool PassDump::accept(std::uint64_t bytes) {
    if (stopped_) return false;
    const bool over = images_ >= (config_.inputs ? 800u : kImageLimit) || bytes > kByteLimit || bytes_ > kByteLimit - bytes;
    if (over) {
        stopped_ = true;
        if (!budget_warned_) {
            budget_warned_ = true;
            armigo::diag::write(armigo::diag::Level::warning,
                                "Armigo NVN: pass dump stopped after %u images and %llu bytes\n", images_,
                                static_cast<unsigned long long>(bytes_));
        }
        return false;
    }
    ++images_;
    bytes_ += bytes;
    return true;
}

PassDumpJob PassDump::prepare(id<MTLTexture> texture, std::string kind, std::uint64_t serial, NSUInteger level,
                              NSUInteger slice, NSUInteger depth_plane, bool numeric) {
    PassDumpJob job;
    job.image.kind = std::move(kind);
    job.image.serial = serial;
    if (!texture) {
        job.image.note = "dump:skipped";
        job.image.format = "?";
        return job;
    }
    auto format = classify(texture.pixelFormat);
    const auto pixel_format = static_cast<unsigned>(texture.pixelFormat);
    if (numeric && format.pipe == DumpFormat::Pipe::skip &&
        ((pixel_format >= MTLPixelFormatBC1_RGBA && pixel_format <= MTLPixelFormatBC7_RGBAUnorm_sRGB) ||
         (pixel_format >= MTLPixelFormatASTC_4x4_sRGB && pixel_format <= MTLPixelFormatASTC_12x12_HDR)))
        format = hdr("CompressedFloat");
    job.image.format = format.name ? format.name : "fmt" + std::to_string(static_cast<unsigned>(texture.pixelFormat));
    job.image.width = static_cast<std::uint32_t>(std::max<NSUInteger>(1, texture.width >> level));
    job.image.height = static_cast<std::uint32_t>(std::max<NSUInteger>(1, texture.height >> level));
    job.image.native_id = texture.gpuResourceID._impl;
    id<MTLTexture> root = texture;
    while (root.parentTexture) root = root.parentTexture;
    job.image.native_root_id = root.gpuResourceID._impl;
    if (config_.metadata_only) {
        job.image.note = "metadata";
        return job;
    }
    if (texture.sampleCount > 1 || format.pipe == DumpFormat::Pipe::skip) {
        job.image.note = "dump:skipped";
        return job;
    }
    // Metal makes a 2D view only of a 2D, 2D array or cube texture; a 3D
    // target (a pass drawing into one depth plane) is not dumped.
    const auto type = texture.textureType;
    if (type != MTLTextureType2D && type != MTLTextureType2DArray && type != MTLTextureTypeCube &&
        type != MTLTextureTypeCubeArray && !(numeric && type == MTLTextureType3D && level == 0)) {
        job.image.note = "dump:skipped";
        return job;
    }
    const bool volume = type == MTLTextureType3D;
    const bool direct = volume || (type == MTLTextureType2D && level == 0 && slice == 0 && depth_plane == 0);
    id<MTLTexture> view = texture;
    if (!direct)
        view = [texture newTextureViewWithPixelFormat:texture.pixelFormat textureType:MTLTextureType2D
                                               levels:NSMakeRange(level, 1) slices:NSMakeRange(slice, 1)];
    if (!view || !view.width || !view.height) {
        job.image.note = "dump:skipped";
        return job;
    }
    job.image.width = static_cast<std::uint32_t>(view.width);
    job.image.height = static_cast<std::uint32_t>(view.height);
    if (!pipelines()) {
        job.image.note = "dump:skipped";
        return job;
    }
    id<MTLTexture> sample = view;
    id<MTLTexture> blit_from = nil;
    if (!(view.usage & MTLTextureUsageShaderRead)) {
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:view.pixelFormat width:view.width
                                                                            height:view.height mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageShaderRead;
        if (format.pipe == DumpFormat::Pipe::depth)
            descriptor.usage |= MTLTextureUsageRenderTarget;
        else
            descriptor.allowGPUOptimizedContents = NO;
        sample = [device_ newTextureWithDescriptor:descriptor];
        if (!sample) {
            job.image.note = "dump:skipped";
            return job;
        }
        blit_from = view;
    }
    const std::uint32_t src_w = job.image.width;
    const std::uint32_t src_h = job.image.height;
    std::uint32_t dst_w = src_w;
    std::uint32_t dst_h = src_h;
    if (src_w > config_.width) {
        dst_w = config_.width;
        dst_h = static_cast<std::uint32_t>(std::max<std::uint64_t>(
            1, (static_cast<std::uint64_t>(src_h) * config_.width + src_w / 2) / src_w));
    }
    const std::uint32_t depth = volume ? static_cast<std::uint32_t>(texture.depth) : 1u;
    const auto plane_height = dst_h;
    dst_h *= depth;
    const bool hdr = numeric || (format.pipe == DumpFormat::Pipe::floating && format.mode == 1);
    const std::uint64_t bytes = static_cast<std::uint64_t>(dst_w) * dst_h * 4 +
                                (hdr ? static_cast<std::uint64_t>(dst_w) * dst_h * 12 : 0) + 256;
    if (!accept(bytes)) {
        job.image.note = "dump:stopped";
        return job;
    }
    id<MTLBuffer> pixels = [device_ newBufferWithLength:static_cast<NSUInteger>(dst_w) * dst_h * 4
                                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> raw = nil;
    if (hdr)
        raw = [device_ newBufferWithLength:static_cast<NSUInteger>(dst_w) * dst_h * 12
                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> args = [device_ newBufferWithLength:256 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bound_raw = raw ? raw : scratch();
    auto* table_descriptor = [MTL4ArgumentTableDescriptor new];
    table_descriptor.maxTextureBindCount = 1;
    table_descriptor.maxBufferBindCount = 3;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [device_ newArgumentTableWithDescriptor:table_descriptor error:&error];
    if (!pixels || !args || !bound_raw || !table || !pixels.contents || !args.contents) {
        if (images_) --images_;
        if (bytes_ >= bytes) bytes_ -= bytes;
        job.image.note = "dump:skipped";
        return job;
    }
    std::memset(pixels.contents, 0, pixels.length);
    if (raw && raw.contents) std::memset(raw.contents, 0, raw.length);
    auto* arguments = static_cast<DumpArgs*>(args.contents);
    *arguments = DumpArgs{job.image.width, job.image.height, dst_w, plane_height, numeric ? 3u : format.mode, format.scale, depth};
    [table setTexture:sample.gpuResourceID atIndex:0];
    [table setAddress:pixels.gpuAddress atIndex:0];
    [table setAddress:bound_raw.gpuAddress atIndex:1];
    [table setAddress:args.gpuAddress atIndex:2];
    id<MTLComputePipelineState> pipeline = volume ? volume_pipeline_ : float_pipeline_;
    if (format.pipe == DumpFormat::Pipe::unsign) pipeline = uint_pipeline_;
    else if (format.pipe == DumpFormat::Pipe::sign) pipeline = int_pipeline_;
    else if (format.pipe == DumpFormat::Pipe::depth) pipeline = depth_pipeline_;
    job.image.dst_w = dst_w;
    job.image.dst_h = dst_h;
    job.image.hdr = hdr;
    job.image.pixels = pixels;
    job.image.raw = raw;
    job.pipeline = pipeline;
    job.table = table;
    job.sample = sample;
    job.blit_from = blit_from;
    job.args = args;
    job.bound_raw = bound_raw;
    return job;
}

PassDumpJob PassDump::prepare_buffer(std::string kind, std::uint64_t address, std::uint64_t bytes) {
    PassDumpJob job;
    job.image.kind = std::move(kind);
    job.image.format = "Bytes";
    if (config_.metadata_only) {
        job.image.note = "buffer:gpu_address:" + std::to_string(address) + ":bytes:" + std::to_string(bytes);
        return job;
    }
    if (!address || !bytes || bytes > 1024 * 1024 || buffer_count_ >= 4096 || !pipelines() || !accept(bytes)) {
        job.image.note = "buffer:skipped";
        return job;
    }
    --images_;
    ++buffer_count_;
    job.image.raw = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    job.args = [device_ newBufferWithLength:16 options:MTLResourceStorageModeShared];
    auto* descriptor = [MTL4ArgumentTableDescriptor new];
    descriptor.maxBufferBindCount = 3;
    NSError* error = nil;
    job.table = [device_ newArgumentTableWithDescriptor:descriptor error:&error];
    if (!job.image.raw || !job.args || !job.table || !job.args.contents) {
        job.image.note = "buffer:unavailable";
        return job;
    }
    *static_cast<std::uint32_t*>(job.args.contents) = static_cast<std::uint32_t>(bytes);
    [job.table setAddress:address atIndex:0];
    [job.table setAddress:job.image.raw.gpuAddress atIndex:1];
    [job.table setAddress:job.args.gpuAddress atIndex:2];
    job.pipeline = copy_pipeline_;
    job.buffer_copy = true;
    job.image.note = "buffer:gpu_address:" + std::to_string(address) + ":bytes:" + std::to_string(bytes);
    return job;
}

void encode_pass_dump_jobs(id<MTL4CommandBuffer> commands, id<MTLResidencySet> residency,
                           NSMutableArray<id>* retained, const std::vector<PassDumpJob>& jobs) {
    if (!commands || jobs.empty()) return;
    static bool warned = false;
    id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    if (!encoder) {
        warn_once(warned, "Armigo NVN: pass dump encoder was not created\n");
        return;
    }
    [encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit | MTLStageDispatch
                   visibilityOptions:MTL4VisibilityOptionDevice];
    for (const auto& job : jobs) {
        if (!job.pipeline || !job.table || (!job.sample && !job.buffer_copy)) continue;
        retain_texture(residency, retained, job.sample);
        retain_texture(residency, retained, job.blit_from);
        retain_allocation(residency, retained, job.image.pixels);
        retain_allocation(residency, retained, job.image.raw);
        retain_allocation(residency, retained, job.args);
        retain_allocation(residency, retained, job.bound_raw);
        if (retained) {
            [retained addObject:job.pipeline];
            [retained addObject:job.table];
        }
        if (job.blit_from) {
            [encoder copyFromTexture:job.blit_from toTexture:job.sample];
            [encoder barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageDispatch
                            visibilityOptions:MTL4VisibilityOptionDevice];
        }
        [encoder setComputePipelineState:job.pipeline];
        [encoder setArgumentTable:job.table];
        // A threadgroup larger than the grid is rejected. 16x16 is the cap
        // when the downscaled image is at least that big.
        const NSUInteger grid_w = job.buffer_copy ? job.image.raw.length : std::max<NSUInteger>(job.image.dst_w, 1);
        const NSUInteger grid_h = std::max<NSUInteger>(job.image.dst_h, 1);
        [encoder dispatchThreads:MTLSizeMake(grid_w, grid_h, 1)
           threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(16, grid_w),
                                             std::min<NSUInteger>(16, grid_h), 1)];
    }
    [encoder endEncoding];
}

void publish_pass_dumps(std::vector<PassDumpLine> lines) {
    if (lines.empty()) return;
    auto held = std::make_shared<std::vector<PassDumpLine>>(std::move(lines));
    dispatch_async(pass_dump_queue(), ^{
        @autoreleasepool { write_pass_dumps(*held); }
    });
}

} // namespace armigo::platform::metal::nvn_driver
