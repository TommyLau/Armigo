// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "commands.h"
#include "entry_table.h"
#include "nvn_replay.h"
#include "nvn_draw_state.h"
#include <armigo/memory/write_tracker.h>
#include <armigo/platform/presentation_thread.h>
#include <armigo/shader/vertex_fetch.h>
#include "programs.h"
#include "resource_layout.h"

#include <map>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>
#include <pthread.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <sys/mman.h>

namespace {

using namespace armigo;
using namespace armigo::nvn;
using namespace armigo::nvn::api;
namespace replay = armigo::platform::metal::nvn_driver;

unsigned failures = 0;
#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition);       \
            ++failures;                                                         \
        }                                                                       \
    } while (false)

template <Entry E>
typename EntrySignature<E>::type load() {
    return reinterpret_cast<typename EntrySignature<E>::type>(
        armigo_nvn_bootstrap_loader(entry_name(E).data()));
}

std::vector<std::byte> build_library(std::string_view source) {
    auto pattern = (std::filesystem::temp_directory_path() /
                    "armigo-nvn-replay-test-XXXXXX").string();
    std::vector<char> path(pattern.begin(), pattern.end());
    path.push_back('\0');
    if (!mkdtemp(path.data())) return {};
    const std::filesystem::path root(path.data());
    const auto source_path = root / "synthetic.metal";
    const auto air_path = root / "synthetic.air";
    const auto library_path = root / "synthetic.metallib";
    {
        std::ofstream output(source_path);
        output.write(source.data(), static_cast<std::streamsize>(source.size()));
    }
    const std::string compile = "xcrun --sdk macosx metal -c '" + source_path.string() +
        "' -o '" + air_path.string() + "'";
    const std::string link = "xcrun --sdk macosx metallib '" + air_path.string() +
        "' -o '" + library_path.string() + "'";
    if (std::system(compile.c_str()) || std::system(link.c_str())) return {};
    std::ifstream input(library_path, std::ios::binary);
    const std::string bytes{std::istreambuf_iterator<char>(input),
                            std::istreambuf_iterator<char>()};
    const auto span = std::as_bytes(std::span(bytes.data(), bytes.size()));
    std::vector<std::byte> result{span.begin(), span.end()};
    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);
    return result;
}

struct Stages {
    std::shared_ptr<const platform::CompiledGraphicsStage> vertex;
    std::shared_ptr<const platform::CompiledGraphicsStage> fragment;
    std::shared_ptr<const platform::CompiledGraphicsStage> compute;
    std::shared_ptr<const platform::CompiledGraphicsStage> chain_compute;
    std::shared_ptr<const platform::CompiledGraphicsStage> attribute_vertex;
    std::shared_ptr<const platform::CompiledGraphicsStage> fetch_vertex;
    std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 2> bit_vertices;
    std::shared_ptr<const platform::CompiledGraphicsStage> uniform_fragment;
    std::shared_ptr<const platform::CompiledGraphicsStage> far_uniform_fragment;
    std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 3> mrt_fragments;
    std::shared_ptr<const platform::CompiledGraphicsStage> sampled_fragment;
    std::shared_ptr<const platform::CompiledGraphicsStage> mip_fragment;
    std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 2> depth_reader_fragments;
    std::shared_ptr<const platform::CompiledGraphicsStage> writer_fragment;
    std::shared_ptr<const platform::CompiledGraphicsStage> reader_fragment;
    std::shared_ptr<const platform::CompiledGraphicsStage> writer_vertex;
    std::shared_ptr<const platform::CompiledGraphicsStage> handle_fragment;
    std::shared_ptr<const platform::CompiledGraphicsStage> descriptor_fragment;
    std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 3> report_texture_fragments;
    std::shared_ptr<const platform::CompiledGraphicsStage> report_vertex;

    static bool resolve(void* raw, const DeviceState*, std::span<const NVNshaderData> shaders,
                        std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 6>& out,
                        std::string& error) {
        auto& self = *static_cast<Stages*>(raw);
        if (shaders.size() == 2 && shaders[0].data == 54) {
            out[NVN_SHADER_STAGE_VERTEX] = self.report_vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.fragment;
            return true;
        }
        if (shaders.size() == 2 && shaders[0].data >= 51 && shaders[0].data <= 53) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.report_texture_fragments[shaders[0].data - 51];
            return true;
        }
        if (shaders.size() == 2 && shaders[0].data >= 9 && shaders[0].data <= 11) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.mrt_fragments[shaders[0].data - 9];
            return true;
        }
        if (shaders.size() == 2 && (shaders[0].data == 21 || shaders[0].data == 22)) {
            out[NVN_SHADER_STAGE_VERTEX] = self.bit_vertices[shaders[0].data - 21];
            out[NVN_SHADER_STAGE_FRAGMENT] = self.fragment;
            return true;
        }
        if (shaders.size() == 2 && shaders[0].data == 1 && shaders[1].data == 1) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.fragment;
            return true;
        }
        if (shaders.size() == 1 && shaders[0].data == 31) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            return true;
        }
        if (shaders.size() == 1 && shaders[0].data == 2) {
            out[NVN_SHADER_STAGE_COMPUTE] = self.compute;
            return true;
        }
        if (shaders.size() == 1 && shaders[0].data == 20) {
            out[NVN_SHADER_STAGE_COMPUTE] = self.chain_compute;
            return true;
        }
        if (shaders.size() == 2 && shaders[0].data == 23) {
            out[NVN_SHADER_STAGE_VERTEX] = self.fetch_vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.uniform_fragment;
            return true;
        }
        if (shaders.size() == 2 && shaders[0].data == 3) {
            out[NVN_SHADER_STAGE_VERTEX] = self.attribute_vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.uniform_fragment;
            return true;
        }
        if (shaders.size() == 2 && shaders[0].data == 4) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.sampled_fragment;
            return true;
        }
        if (shaders.size() == 2 && shaders[0].data == 5) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.uniform_fragment;
            return true;
        }
        if (shaders.size() == 2 && (shaders[0].data == 6 || shaders[0].data == 7)) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = shaders[0].data == 6
                ? self.writer_fragment : self.reader_fragment;
            return true;
        }
        if (shaders.size() == 2 && shaders[0].data == 17) {
            out[NVN_SHADER_STAGE_VERTEX] = self.writer_vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.fragment;
            return true;
        }
        if (shaders.size() == 2 && (shaders[0].data == 13 || shaders[0].data == 14)) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.depth_reader_fragments[shaders[0].data - 13];
            return true;
        }
        if (shaders.size() == 2 && shaders[0].data == 12) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.mip_fragment;
            return true;
        }
        if (shaders.size() == 2 && (shaders[0].data == 15 || shaders[0].data == 16)) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = shaders[0].data == 15
                ? self.handle_fragment : self.descriptor_fragment;
            return true;
        }
        if (shaders.size() == 2 && shaders[0].data == 8) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            out[NVN_SHADER_STAGE_FRAGMENT] = self.far_uniform_fragment;
            return true;
        }
        error = "synthetic resolver received an unknown program marker";
        return false;
    }
};

class TestResources final : public replay::ReplayResources {
public:
    static constexpr std::uint64_t base = 0x10000;

    explicit TestResources(id<MTLDevice> device) {
        memory = [device newBufferWithLength:0x10000 options:MTLResourceStorageModeShared];
    }

    void set_completion_hook(std::function<bool()> callback) override { completion_hook = std::move(callback); }
    void begin_walk() override { if (on_begin_walk) on_begin_walk(); }
    std::function<void()> on_begin_walk;
    // Upload batches by value, one per committed segment: the batch with
    // value failing_upload "failed".
    void commit_uploads(replay::UploadFence& fence) override {
        fence = {};
        fence.floor = fenced_uploads;
        fence.value = ++fenced_uploads;
    }
    bool uploads_failed(const replay::UploadFence& fence) override {
        return failing_upload > fence.floor && failing_upload <= fence.value;
    }
    std::uint64_t fenced_uploads = 0, failing_upload = 0;

    replay::BufferSpan buffer_at(std::uint64_t address,
                                 std::uint64_t size) override {
        if (require_completion && pending_write) {
            CHECK(completion_hook && completion_hook());
            pending_write = false;
            ++coherence_waits;
        }
        if (!memory || address < base || address - base > memory.length ||
            size > memory.length - (address - base)) return {};
        return replay::BufferSpan{memory, address - base};
    }

    id<MTLTexture> texture(const TextureState& state,
                           const TextureViewState*) override {
        if (on_texture_resolve) on_texture_resolve(state);
        const auto found = textures.find(state.serial);
        return found == textures.end() ? nil : found->second;
    }

    // Attachments exist from the start here, so the look-ahead can key on them.
    AttachmentFormat peek_attachment(const TextureState& state, const TextureViewState*) override {
        const auto found = textures.find(state.serial);
        if (found == textures.end() || !found->second) return {};
        return {found->second.pixelFormat, found->second.sampleCount};
    }

    id<MTLTexture> texture_for_id(std::int32_t texture_id) override {
        ++texture_lookups;
        const auto found = texture_ids.find(texture_id);
        id<MTLTexture> texture = found == texture_ids.end() ? nil : found->second;
        if (on_texture_lookup) on_texture_lookup();
        return texture;
    }

    // With `stamped`, a descriptor lookup carries the ID's own version as its
    // stamp (as Resources does), so only a change to that ID resolves it again.
    id<MTLTexture> texture_for_id_stamped(std::int32_t texture_id, DescriptorStamp& stamp) override {
        id<MTLTexture> texture = texture_for_id(texture_id);
        stamp = {};
        if (stamped && texture) {
            stamp.valid = true;
            stamp.slot = texture_versions[texture_id];
        }
        return texture;
    }
    bool texture_stamp_current(std::int32_t texture_id, const DescriptorStamp& stamp) const override {
        if (!stamped || !stamp.valid) return false;
        const auto found = texture_versions.find(texture_id);
        return stamp.slot == (found == texture_versions.end() ? 0 : found->second);
    }

    id<MTLSamplerState> sampler_for_id(std::int32_t id) override {
        const auto found = samplers.find(id);
        return found == samplers.end() ? nil : found->second;
    }
    // Versions by the sampler registration that made them, as Resources keeps
    // them: a binding resolves the newest version made before it.
    id<MTLSamplerState> sampler_for_id(std::int32_t sampler_id, std::uint64_t registrations) override {
        const auto versions = sampler_versions.find(sampler_id);
        if (versions == sampler_versions.end()) return sampler_for_id(sampler_id);
        id<MTLSamplerState> chosen = nil;
        for (const auto& [registration, sampler] : versions->second)
            if (registration <= registrations) chosen = sampler;
        return chosen;
    }

    void note_gpu_write(const TextureState& state) override {
        ++texture_writes;
        if (on_texture_write) on_texture_write(state);
    }
    std::uint64_t registration_generation() const override { return registration; }
    // These fixture textures are separate allocations, not views of memory.
    bool overlaps_texture_backing(std::uint64_t, std::uint64_t) const override { return false; }
    void note_gpu_write(std::uint64_t, std::uint64_t) override {
        ++buffer_writes;
        pending_write = true;
    }

    unsigned texture_lookups = 0;
    bool stamped = false;
    std::unordered_map<std::int32_t, std::uint32_t> texture_versions;
    std::function<void()> on_texture_lookup;
    id<MTLBuffer> memory = nil;
    std::unordered_map<std::uint64_t, id<MTLTexture>> textures;
    std::unordered_map<std::int32_t, id<MTLTexture>> texture_ids;
    std::unordered_map<std::int32_t, id<MTLSamplerState>> samplers;
    std::unordered_map<std::int32_t, std::map<std::uint64_t, id<MTLSamplerState>>> sampler_versions;
    unsigned texture_writes = 0, buffer_writes = 0, coherence_waits = 0;
    bool require_completion = false, pending_write = false;
    std::function<bool()> completion_hook;
    std::function<void(const TextureState&)> on_texture_write;
    std::function<void(const TextureState&)> on_texture_resolve;
    std::uint64_t registration = 1;
};

// These contracts cover validity and ordering independently of Metal encoding.
void test_draw_tracking_contracts() {
    replay::DrawInvalidation state;
    CHECK(state.pipeline && state.fixed && state.vertices);
    for (bool dirty : state.arguments) CHECK(dirty);
    state.arguments.fill(false);
    CHECK(state.sync_resources(1, 2));
    for (bool dirty : state.arguments) CHECK(dirty);
    state.arguments.fill(false);
    CHECK(!state.sync_resources(1, 2));
    CHECK(state.sync_resources(2, 2));
    state.arguments.fill(false);
    CHECK(state.sync_resources(2, 3));
    CHECK(!state.attachments_written);
    state.attachments_written = true;
    state.pipeline = false;
    state.fixed = 0;
    state.vertices = false;
    state.begin_pass();
    CHECK(!state.attachments_written && state.pipeline && state.fixed && state.vertices);
    for (bool dirty : state.arguments) CHECK(dirty);
    // A later submission cannot inherit resource validity or write publication.
    replay::DrawInvalidation next_submission;
    CHECK(!next_submission.attachments_written);
    for (bool dirty : next_submission.arguments) CHECK(dirty);

    using Writer = replay::DrawWriteOrder::Writer;
    replay::DrawWriteOrder writes;
    CHECK(writes.consume() == Writer::none);
    writes.draw(true, false);
    CHECK(writes.consume() == Writer::vertex);
    CHECK(writes.consume() == Writer::none);
    writes.draw(false, false);
    CHECK(writes.consume() == Writer::none);
    // A fragment-stage write, alone or with a vertex one, needs the whole pass ordered.
    writes.draw(false, true);
    CHECK(writes.consume() == Writer::any);
    writes.draw(true, true);
    CHECK(writes.consume() == Writer::any);
    // A later read-only draw clears the obligation; another writer replaces it.
    writes.draw(true, true);
    writes.draw(false, false);
    CHECK(writes.consume() == Writer::none);
    writes.draw(true, false);
    writes.write();
    CHECK(writes.consume() == Writer::any);
    writes.write();
    writes.clear();
    CHECK(writes.consume() == Writer::none);
}

void test_draw_table_contracts() {
    struct Key { std::uint64_t first, second; };
    replay::DrawTable<Key, std::uint64_t> table;
    CHECK(table.find({7, 9}) == nullptr);
    std::array<std::uint64_t*, 513> values{};
    for (std::uint64_t i = 0; i < values.size(); ++i)
        values[i] = table.insert({i % 3, i}, std::make_unique<std::uint64_t>(i + 11));
    // Growth must not invalidate an encoder's current pipeline pointer.
    for (std::uint64_t i = 0; i < values.size(); ++i) {
        CHECK(table.find({i % 3, i}) == values[i]);
        CHECK(*values[i] == i + 11);
        CHECK(table.find({i % 3 + 4, i}) == nullptr);
    }
    CHECK(table.size() == values.size());
}

struct Recorder {
    NVNdevice device{};
    NVNcommandBuffer commands{};
    NVNmemoryPool pool{};
    std::array<std::byte, 64 * 1024> donated{};

    Recorder() {
        auto& pool_state = *reinterpret_cast<MemoryPoolState*>(&pool);
        pool_state.memory = reinterpret_cast<std::uint8_t*>(donated.data());
        pool_state.size = donated.size();
        CHECK(load<Entry::nvnCommandBufferInitialize>()(&commands, &device) == 1);
        load<Entry::nvnCommandBufferAddCommandMemory>()(
            &commands, &pool, 0, donated.size());
        load<Entry::nvnCommandBufferAddControlMemory>()(
            &commands, donated.data(), donated.size());
    }

    NVNcommandHandle finish() {
        return load<Entry::nvnCommandBufferEndRecording>()(&commands);
    }
};

TextureState texture_state(std::uint64_t serial, std::uint32_t width,
                           std::uint32_t height) {
    TextureState state{};
    state.serial = serial;
    state.description.target = NVN_TEXTURE_TARGET_2D;
    state.description.format = NVN_FORMAT_RGBA8;
    state.description.width = width;
    state.description.height = height;
    state.description.depth = 1;
    state.description.levels = 1;
    state.snapshot = &state;
    return state;
}

NVNtexture texture_object(TextureState& snapshot) {
    NVNtexture object{};
    auto state = snapshot;
    state.snapshot = &snapshot;
    std::memcpy(&object, &state, sizeof state);
    return object;
}

// A descriptor generation can change between two draws without another bind
// command: the new texture must resolve again while the first draw retains
// its old one. A guest store to bound uniform memory is read where the GPU
// reads it, at execution, as on the hardware: both draws see the new bytes.
void test_binding_generations(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    for (unsigned mode = 0; mode < 8; ++mode) {
        const bool textured = mode == 0 || mode >= 3;
        const bool tracked = mode == 2 || mode == 3;
        TestResources resources(device);
        resources.stamped = mode >= 6;
        auto backend = replay::create_replay_backend(device, queue, resources);
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
            width:4 height:4 mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        resources.textures[95] = [device newTextureWithDescriptor:descriptor];
        auto state = texture_state(95, 4, 4);
        auto target = texture_object(state);
        const NVNtexture* targets[]{&target};
        id<MTLTexture> red_texture = [device newTextureWithDescriptor:descriptor];
        id<MTLTexture> green_texture = [device newTextureWithDescriptor:descriptor];
        std::array<std::uint32_t, 16> red_pixels{}, green_pixels{};
        red_pixels.fill(0xff0000ffu);
        green_pixels.fill(0xff00ff00u);
        [red_texture replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0
            withBytes:red_pixels.data() bytesPerRow:16];
        [green_texture replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0
            withBytes:green_pixels.data() bytesPerRow:16];
        resources.texture_ids[300] = red_texture;
        const std::array<float, 4> red{1, 0, 0, 1}, green{0, 1, 0, 1};
        std::memcpy(resources.memory.contents, red.data(), sizeof red);
        if (tracked) memory::WriteTracker::add_range(resources.memory.contents, resources.memory.length);
        resources.on_texture_write = [&](const TextureState&) {
            if (resources.texture_writes != 1 || (mode >= 4 && mode < 6)) return;
            // Stamped lookups: mode 6 registers another ID between the draws,
            // which must not resolve this one again; mode 7 registers this ID
            // again with another texture, which must.
            if (mode == 6) { ++resources.registration; return; }
            if (mode == 7) {
                resources.texture_ids[300] = green_texture;
                ++resources.texture_versions[300];
                ++resources.registration;
                return;
            }
            if (textured) resources.texture_ids[300] = green_texture;
            else std::memcpy(resources.memory.contents, green.data(), sizeof green);
            if (tracked) memory::WriteTracker::note_store(resources.memory.contents, sizeof green);
            else ++resources.registration;
        };
        // A registration during resolution must invalidate the value returned
        // by that resolution on the following draw.
        if (mode == 5) resources.on_texture_lookup = [&] {
            if (resources.texture_lookups != 1) return;
            resources.texture_ids[300] = green_texture;
            ++resources.registration;
        };
        Recorder recorder;
        NVNprogram program{};
        CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
        const NVNshaderData shaders[]{{textured ? 4u : 5u, nullptr}, {textured ? 4u : 5u, nullptr}};
        CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
        load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, 300);
        load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands,
            NVN_SHADER_STAGE_FRAGMENT, 0, TestResources::base, 16);
        load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 2, 4);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        // Force another stage bind without changing the descriptor. Reusing
        // its resolved texture must not require another resource-table lookup.
        if (mode == 4) {
            load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, 301);
            load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, 300);
        }
        load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 2, 0, 2, 4);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        const auto handle = recorder.finish();
        const auto fence = backend->submit({&handle, 1});
        CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
        CHECK(!backend->last_error() && backend->statistics().skipped == 0);
        std::array<std::uint8_t, 64> pixels{};
        [resources.textures[95] getBytes:pixels.data() bytesPerRow:16
            fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
        for (unsigned y = 0; y < 4; ++y)
            for (unsigned x = 0; x < 4; ++x) {
                const auto i = (y * 4 + x) * 4;
                const bool first = textured && (mode == 4 || mode == 6 || x < 2);
                CHECK(pixels[i] == (first ? 255 : 0));
                CHECK(pixels[i + 1] == (first ? 0 : 255));
            }
        if (mode == 4 || mode == 6) CHECK(resources.texture_lookups == 1);
        if (mode == 7) CHECK(resources.texture_lookups == 2);
        if (tracked) memory::WriteTracker::remove_range(resources.memory.contents);
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
        load<Entry::nvnProgramFinalize>()(&program);
        load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
    }
}

// Queued submissions retain their uniform snapshots and query storage until
// retirement, including after the allocator slots wrap several times.
void test_submission_slot_reuse(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    constexpr unsigned submissions = 24;
    TestResources resources(device);
    std::memset(resources.memory.contents, 0, resources.memory.length);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:submissions height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget;
    resources.textures[91] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(91, submissions, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{5, nullptr}, {5, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    constexpr auto uniform = TestResources::base + 0x200;
    id<MTLSharedEvent> gate = [device newSharedEvent];
    std::uint64_t fence = 0;
    for (unsigned index = 0; index != submissions; ++index) {
        // The preceding batch has retired before this queue wait is inserted.
        // Every wrap therefore fills all six slots while the GPU is blocked.
        if (index % 6 == 0) [queue waitForEvent:gate value:index / 6 + 1];
        const std::array<float, 4> color{float(index & 1), float((index >> 1) & 1),
                                       float((index >> 2) & 1), 1};
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
        if (!index) {
            const float black[]{0, 0, 0, 1};
            load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, black, 15);
        }
        load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 0, 16, color.data());
        load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, uniform, 128);
        load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, int(index), 0, 1, int(index % 4 + 1));
        load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
            TestResources::base + 0x1000 + index * sizeof(NVNcounterData));
        const auto handle = recorder.finish();
        fence = backend->submit({&handle, 1});
        CHECK(fence != 0);
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
        if (index % 6 == 5) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (backend->statistics().submissions < index + 1 && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            CHECK(backend->statistics().submissions == index + 1);
            CHECK(backend->completed_fence() == index - 5);
            gate.signaledValue = index / 6 + 1;
            CHECK(backend->wait_fence(fence, 10'000'000'000ull));
        }
    }
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    std::array<std::uint8_t, submissions * 4 * 4> pixels{};
    [resources.textures[91] getBytes:pixels.data() bytesPerRow:submissions * 4
        fromRegion:MTLRegionMake2D(0, 0, submissions, 4) mipmapLevel:0];
    for (unsigned index = 0; index != submissions; ++index) {
        NVNcounterData samples{};
        std::memcpy(&samples, static_cast<const std::byte*>(resources.memory.contents) +
            0x1000 + index * sizeof samples, sizeof samples);
        CHECK(samples.counter == index % 4 + 1);
        for (unsigned y = 0; y != 4; ++y)
            for (unsigned channel = 0; channel != 4; ++channel) {
                const auto expected = channel == 3 || (y <= index % 4 && ((index >> channel) & 1));
                CHECK(pixels[(y * submissions + index) * 4 + channel] == (expected ? 255 : 0));
            }
    }
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// Reports overwrite their destination only at retirement. Reusing it must not
// block encoding, but a GPU copy of the reported bytes remains a real reader.
void test_reused_report_destination(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                                    bool repeated_in_submission) {
    TestResources resources(device);
    std::memset(resources.memory.contents, 0, resources.memory.length);
    std::atomic<unsigned> walks{0};
    resources.on_begin_walk = [&] { walks.fetch_add(1, std::memory_order_release); };
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget;
    resources.textures[91] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(91, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    constexpr auto counter = TestResources::base + 0x4000;
    constexpr auto copied = TestResources::base + 0x5000;
    const NVNcounterData sentinel{99, 77};
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memcpy(bytes + 0x4000, &sentinel, sizeof sentinel);
    id<MTLSharedEvent> gate = [device newSharedEvent];
    [queue waitForEvent:gate value:1];
    std::array<NVNcommandHandle, 3> handles{};
    const auto report = [&](unsigned width) {
        load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, int(width), 4);
        load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED, counter);
    };
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    if (repeated_in_submission) report(1);
    report(2);
    handles[0] = recorder.finish();
    CHECK(backend->submit({&handles[0], 1}) != 0);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    report(3);
    handles[1] = recorder.finish();
    const auto producer = backend->submit({&handles[1], 1});
    CHECK(producer != 0);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferCopyBufferToBuffer>()(&recorder.commands, counter, copied, sizeof(NVNcounterData), 0);
    handles[2] = recorder.finish();
    const auto consumer = backend->submit({&handles[2], 1});
    CHECK(consumer != 0);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (walks.load(std::memory_order_acquire) < 3 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(walks.load(std::memory_order_acquire) == 3);
    // The consumer has started walking, but its source lookup must still wait.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(backend->statistics().submissions == 2);
    CHECK(backend->completed_fence() == 0);
    CHECK(std::memcmp(bytes + 0x4000, &sentinel, sizeof sentinel) == 0);
    gate.signaledValue = 1;
    CHECK(backend->wait_fence(producer, 10'000'000'000ull));
    NVNcounterData published{};
    std::memcpy(&published, bytes + 0x4000, sizeof published);
    CHECK(published.counter == 12 && published.timestamp != 0);
    CHECK(backend->wait_fence(consumer, 10'000'000'000ull));
    CHECK(std::memcmp(bytes + 0x5000, &published, sizeof published) == 0);
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    CHECK(backend->statistics().synchronize_gpu_waits == 0);
    for (const auto handle : handles)
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// Query destinations can alias texels even when the native texture is cached
// or uses separate tiled storage. A partial overlap is still a dependency.
void test_report_texture_alias(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                               unsigned kind, bool cached, bool partial, bool overlaps,
                               bool copy_consumer = false, bool samples = false) {
    const auto before = failures;
    constexpr std::size_t size = 64 * 1024;
    auto* bytes = static_cast<std::uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(bytes != MAP_FAILED);
    if (bytes == MAP_FAILED) return;
    {
        replay::Resources resources(device);
        MemoryPoolState pool{};
        pool.memory = bytes;
        pool.size = size;
        pool.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
        CHECK(resources.register_pool(pool));
        const auto make = [&](std::uint64_t serial, std::size_t offset, bool sampled) {
            auto state = texture_state(serial, sampled ? (partial ? 5 : 1) : 4, sampled ? 1 : 4);
            state.description.samples = 1;
            if (sampled) {
                state.description.format = NVN_FORMAT_R32UI;
                if (kind == 0) state.description.target = NVN_TEXTURE_TARGET_BUFFER;
                if (kind == 1) {
                    state.description.flags = NVN_TEXTURE_FLAGS_LINEAR;
                    state.description.stride = 256;
                }
            }
            const auto layout = resource_layout(state.description);
            CHECK(layout.has_value());
            state.size = layout ? layout->total_size : 0;
            state.layer_size = layout ? layout->layer_stride : 0;
            state.pool = &pool;
            state.offset = offset;
            state.memory = bytes + offset;
            resources.register_texture(state);
            return state;
        };
        auto input = make(211, 0x4000, true);
        auto output = make(212, 0x8000, false);
        for (std::size_t offset = 0; offset < input.size; offset += 4) {
            const std::uint32_t sentinel = 99;
            std::memcpy(input.memory + offset, &sentinel, sizeof sentinel);
        }
        resources.register_texture_descriptor(71, input, nullptr);
        auto backend = replay::create_replay_backend(device, queue, resources);
        Recorder recorder;
        NVNprogram program{};
        CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
        const auto shader = samples ? 53u : kind == 0 ? 51u : 52u;
        const NVNshaderData shaders[]{{shader, nullptr}, {shader, nullptr}};
        CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
        auto target = texture_object(output);
        const NVNtexture* targets[]{&target};
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
        load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, 71);
        if (samples)
            load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
        if (cached)
            load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        // In tiled storage the fifth R32 texel starts the GOB's second
        // 16-byte sector at byte 32; linear storage places it at byte 16.
        const auto texel_offset = partial ? (kind == 2 ? 32u : 16u) : 0u;
        const auto counter = reinterpret_cast<std::uint64_t>(bytes) +
            (overlaps ? 0x4000 + texel_offset : 0x6000);
        for (unsigned index = 0; index != (samples ? 1u : 2u); ++index)
            load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands,
                samples ? NVN_COUNTER_TYPE_SAMPLES_PASSED : NVN_COUNTER_TYPE_TIMESTAMP, counter);
        // The samples case deliberately leaves the visibility-enabled pass
        // and all bindings unchanged, exercising report-driven revalidation.
        if (!samples)
            load<Entry::nvnCommandBufferBarrier>()(&recorder.commands,
                NVN_BARRIER_INVALIDATE_TEXTURE_BIT | NVN_BARRIER_INVALIDATE_SHADER_BIT);
        if (copy_consumer) {
            auto source = texture_object(input);
            const NVNcopyRegion texel{partial ? 4 : 0, 0, 0, 1, 1, 1};
            load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &source, nullptr, &texel,
                reinterpret_cast<std::uint64_t>(bytes + 0xd000), 0);
        } else {
            load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
            const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
            load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target, nullptr, &whole,
                reinterpret_cast<std::uint64_t>(bytes + 0xc000), 0);
        }
        const auto handle = recorder.finish();
        const auto fence = backend->submit({&handle, 1});
        CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
        CHECK(!backend->last_error() && backend->statistics().skipped == 0);
        if (samples) {
            NVNcounterData value{};
            std::memcpy(&value, reinterpret_cast<const void*>(counter), sizeof value);
            CHECK(value.counter == 16);
        }
        if (copy_consumer) {
            std::uint32_t copied = 0;
            std::memcpy(&copied, bytes + 0xd000, sizeof copied);
            CHECK(copied == (overlaps ? 0u : 99u));
        } else {
            for (unsigned pixel = 0; pixel != 16; ++pixel) {
                CHECK(bytes[0xc000 + pixel * 4] == (overlaps ? 0 : 255));
                CHECK(bytes[0xc000 + pixel * 4 + 1] == (overlaps ? 255 : 0));
            }
        }
        if (!overlaps) CHECK(backend->statistics().synchronize_gpu_waits == 0);
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
        load<Entry::nvnProgramFinalize>()(&program);
        load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
    }
    munmap(bytes, size);
    if (failures != before)
        std::fprintf(stderr, "report texture alias kind=%u cached=%d partial=%d overlaps=%d copy=%d samples=%d failures=%u\n",
                     kind, cached, partial, overlaps, copy_consumer, samples, failures - before);
}

// A real texture registry behind the replay seam, with one ordered mutation
// point between reports. The unknown-provider case keeps the base fallback.
class ReportProbeResources final : public replay::ReplayResources {
public:
    explicit ReportProbeResources(replay::Resources& resources) : resources_(resources) {}
    replay::BufferSpan buffer_at(std::uint64_t address, std::uint64_t size) override {
        if (address == marker && mutation) std::exchange(mutation, {})();
        return resources_.buffer_at(address, size);
    }
    id<MTLTexture> texture(const TextureState& state, const TextureViewState* view) override {
        return resources_.texture(state, view);
    }
    id<MTLTexture> texture_for_id(std::int32_t id) override { return resources_.texture_for_id(id); }
    id<MTLSamplerState> sampler_for_id(std::int32_t id) override { return resources_.sampler_for_id(id); }
    void set_completion_hook(std::function<bool()> hook) override { resources_.set_completion_hook(std::move(hook)); }
    void note_gpu_write(const TextureState& state) override { resources_.note_gpu_write(state); }
    void note_gpu_write(std::uint64_t address, std::uint64_t size) override { resources_.note_gpu_write(address, size); }
    std::uint64_t registration_generation() const override { return resources_.registration_generation(); }
    bool overlaps_texture_backing(std::uint64_t address, std::uint64_t size) const override {
        return unknown ? ReplayResources::overlaps_texture_backing(address, size)
                       : resources_.overlaps_texture_backing(address, size);
    }
    std::uint64_t gpu_write_generation() const override { return resources_.gpu_write_generation(); }
    void complete_gpu_writes(std::uint64_t generation) override { resources_.complete_gpu_writes(generation); }
    std::uint64_t marker = 0;
    std::function<void()> mutation;
    bool unknown = false;
private:
    replay::Resources& resources_;
};

void test_report_isolation_probe(id<MTLDevice> device, id<MTL4CommandQueue> queue, unsigned mode) {
    constexpr std::size_t size = 64 * 1024;
    auto* bytes = static_cast<std::uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(bytes != MAP_FAILED);
    if (bytes == MAP_FAILED) return;
    {
        replay::Resources resources(device);
        MemoryPoolState pool{};
        pool.memory = bytes; pool.size = size;
        CHECK(resources.register_pool(pool));
        auto texture = texture_state(214, 4, 4);
        const auto layout = resource_layout(texture.description);
        CHECK(layout.has_value());
        texture.size = layout ? layout->total_size : 0;
        texture.layer_size = layout ? layout->layer_stride : 0;
        texture.memory = bytes + 0x4000; texture.pool = &pool; texture.offset = 0x4000;
        if (mode == 2) resources.register_texture(texture);
        ReportProbeResources probe(resources);
        probe.marker = reinterpret_cast<std::uint64_t>(bytes + 0x100);
        probe.unknown = mode == 4;
        probe.mutation = [&] {
            if (mode == 0) {
                resources.register_texture(texture);
                auto nested = texture;
                nested.serial = 216;
                nested.description.target = NVN_TEXTURE_TARGET_BUFFER;
                nested.description.format = NVN_FORMAT_R32UI;
                nested.description.width = 64; nested.description.height = 1;
                nested.memory += 256; nested.offset += 256; nested.size = 256;
                resources.register_texture(nested);
                auto adjacent = nested;
                adjacent.serial = 217; adjacent.memory += 256; adjacent.offset += 256;
                resources.register_texture(adjacent);
                auto disjoint = adjacent;
                disjoint.serial = 218; disjoint.memory += 512; disjoint.offset += 512;
                resources.register_texture(disjoint);
                const auto base = reinterpret_cast<std::uint64_t>(texture.memory);
                CHECK(resources.overlaps_texture_backing(base + 767, 1));
                CHECK(!resources.overlaps_texture_backing(base + 768, 1));
                CHECK(!resources.overlaps_texture_backing(base + 800, 32));
                CHECK(resources.overlaps_texture_backing(base + 1024, 1));
            }
            if (mode == 1) resources.register_texture_descriptor(72, texture, nullptr);
            if (mode == 2) {
                resources.release_texture(texture);
                auto replacement = texture;
                replacement.serial = 215;
                resources.register_texture(replacement);
            }
        };
        auto backend = replay::create_replay_backend(device, queue, probe);
        Recorder recorder;
        const auto destination = reinterpret_cast<std::uint64_t>(bytes + (mode == 3 ? 0x4ffc : 0x4000));
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP, destination);
        load<Entry::nvnCommandBufferCopyBufferToBuffer>()(&recorder.commands, probe.marker,
            reinterpret_cast<std::uint64_t>(bytes + 0x200), 4, 0);
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP, destination);
        const auto handle = recorder.finish();
        const auto fence = backend->submit({&handle, 1});
        CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
        CHECK(!backend->last_error() && backend->statistics().skipped == 0);
        const auto statistics = backend->statistics();
        CHECK(statistics.report_fast == (mode < 2 ? 1u : 0u));
        CHECK(statistics.report_ordered == (mode < 2 ? 1u : 2u));
        NVNcounterData value{};
        std::memcpy(&value, reinterpret_cast<const void*>(destination), sizeof value);
        CHECK(value.counter == 0 && value.timestamp != 0);
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
        load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
    }
    munmap(bytes, size);
}

void test_report_vertex_alias(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    std::memset(resources.memory.contents, 0, resources.memory.length);
    const std::uint32_t sentinel = 99;
    std::memcpy(static_cast<std::byte*>(resources.memory.contents) + 0x4000, &sentinel, sizeof sentinel);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget;
    resources.textures[91] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(91, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{54, nullptr}, {54, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    NVNvertexAttribState attribute{};
    NVNvertexStreamState stream{};
    load<Entry::nvnVertexAttribStateSetDefaults>()(&attribute);
    load<Entry::nvnVertexAttribStateSetFormat>()(&attribute, NVN_FORMAT_R32UI, 0);
    load<Entry::nvnVertexAttribStateSetStreamIndex>()(&attribute, 0);
    load<Entry::nvnVertexStreamStateSetDefaults>()(&stream);
    load<Entry::nvnVertexStreamStateSetStride>()(&stream, 4);
    load<Entry::nvnVertexStreamStateSetDivisor>()(&stream, 1);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    const float black[]{0, 0, 0, 1};
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, black, 15);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindVertexAttribState>()(&recorder.commands, 1, &attribute);
    load<Entry::nvnCommandBufferBindVertexStreamState>()(&recorder.commands, 1, &stream);
    load<Entry::nvnCommandBufferBindVertexBuffer>()(&recorder.commands, 0, TestResources::base + 0x4000, 4);
    load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    for (unsigned index = 0; index != 2; ++index)
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
            TestResources::base + 0x4000);
    // The unchanged stream must consume the zero-sample report, not 99.
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[91] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned pixel = 0; pixel != 16; ++pixel) CHECK(pixels[pixel * 4 + 1] == 255);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_report_attachment_alias(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    constexpr std::size_t size = 64 * 1024;
    auto* bytes = static_cast<std::uint8_t*>(mmap(nullptr, size, PROT_READ | PROT_WRITE,
                                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(bytes != MAP_FAILED);
    if (bytes == MAP_FAILED) return;
    {
        replay::Resources resources(device);
        MemoryPoolState pool{};
        pool.memory = bytes; pool.size = size;
        pool.flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
        CHECK(resources.register_pool(pool));
        auto state = texture_state(213, 4, 4);
        state.description.flags = NVN_TEXTURE_FLAGS_LINEAR;
        state.description.stride = 256;
        const auto layout = resource_layout(state.description);
        CHECK(layout.has_value());
        state.size = layout ? layout->total_size : 0;
        state.layer_size = layout ? layout->layer_stride : 0;
        state.memory = bytes + 0x4000; state.pool = &pool; state.offset = 0x4000;
        resources.register_texture(state);
        auto backend = replay::create_replay_backend(device, queue, resources);
        Recorder recorder;
        NVNprogram program{};
        CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
        const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
        CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
        auto target = texture_object(state);
        const NVNtexture* targets[]{&target};
        NVNcolorState color{};
        NVNblendState blend{};
        load<Entry::nvnColorStateSetDefaults>()(&color);
        load<Entry::nvnColorStateSetBlendEnable>()(&color, 0, 1);
        load<Entry::nvnBlendStateSetDefaults>()(&blend);
        load<Entry::nvnBlendStateSetBlendFunc>()(&blend, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ONE,
                                               NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ONE);
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
        load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        for (unsigned index = 0; index != 2; ++index)
            load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
                reinterpret_cast<std::uint64_t>(state.memory));
        load<Entry::nvnCommandBufferBindColorState>()(&recorder.commands, &color);
        load<Entry::nvnCommandBufferBindBlendState>()(&recorder.commands, &blend);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
        load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target, nullptr, &whole,
            reinterpret_cast<std::uint64_t>(bytes + 0xc000), 0);
        const auto handle = recorder.finish();
        const auto fence = backend->submit({&handle, 1});
        CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
        CHECK(!backend->last_error() && backend->statistics().skipped == 0);
        CHECK(bytes[0xc000] == 16 && bytes[0xc001] == 255 && bytes[0xc002] == 0 && bytes[0xc003] == 255);
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
        load<Entry::nvnProgramFinalize>()(&program);
        load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
    }
    munmap(bytes, size);
}

void test_shutdown_waits_for_feedback_logging_tail(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    struct Observation {
        std::promise<void> joined, tail_entered, release_tail;
        std::shared_future<void> released = release_tail.get_future().share();
        std::atomic<bool> reported_join{false};
    } observation;
    replay::ReplaySchedulingPolicy policy;
    policy.context = &observation;
    policy.observe_shutdown_workers_joined = [](void* context) {
        auto& state = *static_cast<Observation*>(context);
        if (!state.reported_join.exchange(true)) state.joined.set_value();
    };
    policy.observe_retired_log = [](void* context, std::uint64_t) {
        auto& state = *static_cast<Observation*>(context);
        state.tail_entered.set_value();
        state.released.wait();
    };
    auto backend = replay::create_replay_backend(device, queue, resources, policy);
    id<MTLSharedEvent> gate = [device newSharedEvent];
    [queue waitForEvent:gate value:1];
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferCopyBufferToBuffer>()(&recorder.commands,
        TestResources::base, TestResources::base + 0x100, 16, 0);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence != 0);
    auto stopped = std::async(std::launch::async, [&] {
        return backend->stop_submission_workers();
    });
    // Keep the GPU blocked until the replay worker has joined. Retirement and
    // its logging tail must therefore execute on GPU feedback, not that worker.
    CHECK(observation.joined.get_future().wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    gate.signaledValue = 1;
    CHECK(observation.tail_entered.get_future().wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    CHECK(backend->completed_fence() >= fence);
    // Both handshakes completed: the worker joined and the GPU published its
    // fence. With no shutdown deadline, a fence timeout cannot pass this test.
    // Only the held logging tail keeps shutdown pending, until explicitly released.
    CHECK(stopped.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout);
    observation.release_tail.set_value();
    CHECK(stopped.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    CHECK(stopped.get());
    CHECK(backend->stop_submission_workers(std::chrono::steady_clock::now() + std::chrono::seconds(5)));
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_submission_shutdown_joins_and_retries(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    resources.on_begin_walk = [&] {
        entered.set_value();
        released.wait();
    };
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence != 0);
    CHECK(entered.get_future().wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    // A blocked command walk cannot be reported quiescent. Timeout leaves the
    // submission retained, closes admission, and supports a later drain retry.
    CHECK(!backend->stop_submission_workers(std::chrono::steady_clock::now() + std::chrono::milliseconds(10)));
    CHECK(backend->submit({&handle, 1}) == 0);
    release.set_value();
    CHECK(backend->stop_submission_workers(std::chrono::steady_clock::now() + std::chrono::seconds(5)));
    CHECK(backend->completed_fence() >= fence);
    CHECK(backend->stop_submission_workers());
    // Command registration can retire while the process-wide backend still
    // exists: no worker can walk this list or its registry after the join.
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
    CHECK(backend->submit({&handle, 1}) == 0);
}

void test_submission_scheduling_period(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    struct Observation { bool periodic; std::uint32_t period; qos_class_t qos; };
    std::vector<Observation> observations;
    resources.on_begin_walk = [&] {
        thread_time_constraint_policy_data_t policy{};
        mach_msg_type_number_t count = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
        boolean_t is_default = false;
        CHECK(thread_policy_get(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
            reinterpret_cast<thread_policy_t>(&policy), &count, &is_default) == KERN_SUCCESS);
        qos_class_t qos = QOS_CLASS_UNSPECIFIED;
        int priority = 0;
        CHECK(pthread_get_qos_class_np(pthread_self(), &qos, &priority) == 0);
        observations.push_back({!is_default, policy.period, qos});
    };
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    const auto handle = recorder.finish();
    std::uint64_t fence = 0;
    // Two submissions in one frame must not sample the worker budget twice.
    // Queued frame markers retain their own period after the next hint arrives.
    const auto submit = [&](std::uint64_t period) {
        backend->set_scheduling_period(period);
        fence = backend->submit({&handle, 1});
        CHECK(fence);
    };
    submit(4'000'000);
    submit(8'000'000);
    backend->finish_frame(4'000'000);
    submit(8'000'000);
    backend->finish_frame(8'000'000);
    submit(0);
    backend->finish_frame(0);
    submit(0);
    CHECK(backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    CHECK(observations.size() == 5);
    if (observations.size() == 5) {
        mach_timebase_info_data_t timebase{};
        CHECK(mach_timebase_info(&timebase) == KERN_SUCCESS);
        CHECK(!observations[0].periodic && observations[0].qos == QOS_CLASS_USER_INTERACTIVE);
        CHECK(!observations[1].periodic && observations[1].qos == QOS_CLASS_USER_INTERACTIVE);
        CHECK(observations[2].periodic &&
            observations[2].period == 4'000'000ull * timebase.denom / timebase.numer);
        CHECK(observations[3].periodic &&
            observations[3].period == 8'000'000ull * timebase.denom / timebase.numer);
        CHECK(!observations[4].periodic && observations[4].qos == QOS_CLASS_UNSPECIFIED);
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_submission_encode_wait_budget(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    const char* encode = std::getenv("ARMIGO_NVN_ENCODE_THREAD");
    if (encode && std::strcmp(encode, "0") == 0) return;
    struct Observation {
        std::uint64_t fence, period, spin;
        bool operator==(const Observation&) const = default;
    };
    struct Probe {
        std::mutex mutex;
        std::vector<Observation> observations;
        static void observe(void* context, std::uint64_t fence, std::uint64_t period,
                            std::uint64_t spin) {
            auto& self = *static_cast<Probe*>(context);
            const std::lock_guard lock(self.mutex);
            self.observations.push_back({fence, period, spin});
        }
    } probe;
    TestResources resources(device);
    std::promise<void> release;
    const auto ready = release.get_future().share();
    // Hold the walk until every hint has changed. Later submissions must use
    // the state they captured, not the newest period or another frame's spin.
    resources.on_begin_walk = [ready] { ready.wait(); };
    auto backend = replay::create_replay_backend(device, queue, resources,
        {&probe, [](void*, replay::ReplayThread, std::uint64_t, std::uint64_t) { return true; },
         Probe::observe});
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    const auto handle = recorder.finish();
    std::vector<Observation> expected;
    const auto submit = [&](std::uint64_t period, bool retain_spin, std::uint64_t spin) {
        backend->set_scheduling_period(period, retain_spin);
        const auto fence = backend->submit({&handle, 1});
        CHECK(fence);
        expected.push_back({fence, period, spin});
    };
    submit(33'333'332, true, 500'000);
    submit(16'666'666, false, 0);
    submit(0, false, 500'000);
    submit(3'333'333, false, 500'000);
    submit(33'333'332, true, 500'000);
    backend->set_scheduling_period(64'000'000, false);
    release.set_value();
    CHECK(backend->wait_fence(expected.back().fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    {
        const std::lock_guard lock(probe.mutex);
        CHECK(probe.observations == expected);
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_guest_present_scheduling(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    // The guest thread that acquires window textures declares its own budget
    // under its own role; an acquire without a known period keeps it.
    struct Probe {
        std::mutex mutex;
        unsigned declarations = 0, restorations = 0;
        std::thread::id thread;
        static bool apply(void* context, replay::ReplayThread role, std::uint64_t period, std::uint64_t) {
            auto& self = *static_cast<Probe*>(context);
            const std::lock_guard lock(self.mutex);
            if (role != replay::ReplayThread::guest_present) return true;
            self.thread = std::this_thread::get_id();
            ++(period ? self.declarations : self.restorations);
            return true;
        }
    } probe;
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources, {&probe, Probe::apply});
    std::thread::id acquirer;
    std::thread([&] {
        acquirer = std::this_thread::get_id();
        for (unsigned frame = 0; frame < 4; ++frame) backend->acquire_tick(4'000'000);
        for (unsigned frame = 0; frame < 4; ++frame) backend->acquire_tick(0);
    }).join();
    const std::lock_guard lock(probe.mutex);
    CHECK(probe.declarations == 1);
    CHECK(probe.restorations == 0);
    CHECK(probe.thread == acquirer);
}

void test_scheduling_failures_and_encode_policy(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    struct Probe {
        std::mutex mutex;
        std::array<unsigned, 2> declarations{}, restorations{};
        bool valid = true;
        std::array<std::thread::id, 2> threads{};
        static bool apply(void* context, replay::ReplayThread role, std::uint64_t period,
                          std::uint64_t computation) {
            auto& self = *static_cast<Probe*>(context);
            const std::lock_guard lock(self.mutex);
            // Only the replay worker and encode thread run in this test.
            self.valid &= role != replay::ReplayThread::guest_present;
            const auto index = role == replay::ReplayThread::worker ? 0u : 1u;
            self.threads[index] = std::this_thread::get_id();
            auto& attempts = period ? self.declarations[index] : self.restorations[index];
            ++attempts;
            const bool reject = attempts == 1;
            if (!reject) {
                if (period) {
                    self.valid &= platform::declare_periodic_thread(
                        std::chrono::nanoseconds(period), std::chrono::nanoseconds(computation));
                } else {
                    thread_standard_policy_data_t policy{};
                    self.valid &= thread_policy_set(pthread_mach_thread_np(pthread_self()),
                        THREAD_STANDARD_POLICY, reinterpret_cast<thread_policy_t>(&policy),
                        THREAD_STANDARD_POLICY_COUNT) == KERN_SUCCESS;
                }
            }
            thread_time_constraint_policy_data_t observed{};
            mach_msg_type_number_t count = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
            boolean_t is_default = false;
            self.valid &= thread_policy_get(pthread_mach_thread_np(pthread_self()),
                THREAD_TIME_CONSTRAINT_POLICY, reinterpret_cast<thread_policy_t>(&observed),
                &count, &is_default) == KERN_SUCCESS;
            // A rejected declaration leaves ordinary scheduling intact; a
            // rejected restore keeps the accepted real-time declaration.
            self.valid &= bool(is_default) == (period ? reject : !reject);
            if (!is_default) {
                mach_timebase_info_data_t timebase{};
                self.valid &= mach_timebase_info(&timebase) == KERN_SUCCESS;
                if (timebase.numer)
                    self.valid &= observed.period == 4'000'000ull * timebase.denom / timebase.numer;
            }
            return !reject;
        }
    } probe;
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources, {&probe, Probe::apply});
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    const auto handle = recorder.finish();
    std::uint64_t fence = 0;
    for (const auto period : {4'000'000ull, 0ull}) {
        for (unsigned frame = 0; frame < 122; ++frame) {
            backend->set_scheduling_period(period);
            fence = backend->submit({&handle, 1});
            CHECK(fence);
            backend->finish_frame(period);
        }
    }
    CHECK(backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    // Joining drains the final frame marker and encode-thread policy tick.
    backend.reset();
    CHECK(probe.valid);
    CHECK(probe.declarations[0] == 2 && probe.restorations[0] == 2);
    const char* encode = std::getenv("ARMIGO_NVN_ENCODE_THREAD");
    if (!encode || std::strcmp(encode, "0") != 0) {
        CHECK(probe.declarations[1] == 2 && probe.restorations[1] == 2);
        CHECK(probe.threads[0] != probe.threads[1]);
    } else {
        CHECK(probe.declarations[1] == 0 && probe.restorations[1] == 0);
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A writable draw must order a following read-only draw. Attachment write
// publication is once per uninterrupted pass and is rearmed by a guest barrier.
void test_draw_write_order_and_submission_bindings(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    // The callback executes on the replay worker, whose QoS must not depend
    // on the caller's thread policy.
    bool worker_qos_seen = false;
    resources.on_texture_write = [&](const TextureState&) {
        qos_class_t qos = QOS_CLASS_UNSPECIFIED;
        int relative_priority = 0;
        CHECK(pthread_get_qos_class_np(pthread_self(), &qos, &relative_priority) == 0);
        CHECK(qos == QOS_CLASS_USER_INTERACTIVE);
        worker_qos_seen = true;
    };
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[91] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(91, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram writer{}, reader{};
    CHECK(load<Entry::nvnProgramInitialize>()(&writer, &recorder.device));
    CHECK(load<Entry::nvnProgramInitialize>()(&reader, &recorder.device));
    const NVNshaderData write_shaders[]{{6, nullptr}, {6, nullptr}};
    const NVNshaderData read_shaders[]{{7, nullptr}, {7, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&writer, 2, write_shaders));
    CHECK(load<Entry::nvnProgramSetShaders>()(&reader, 2, read_shaders));
    std::memset(resources.memory.contents, 0, resources.memory.length);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands,
        NVN_SHADER_STAGE_FRAGMENT, 0, TestResources::base, 4);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &writer, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &reader, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    for (unsigned i = 0; i < 5; ++i)
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferBarrier>()(&recorder.commands, 0);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    auto handle = recorder.finish();
    auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    // The writer's fragment-stage storage write ends the pass before the
    // readers (a render encoder can order only vertex-stage work), and the
    // barrier ends the next one: three passes, each noting its target.
    CHECK(resources.texture_writes == 3 && resources.buffer_writes == 1);
    CHECK(worker_qos_seen);
    CHECK(backend->statistics().render_passes == 3);
    CHECK(*static_cast<const std::uint32_t*>(resources.memory.contents) == 16);
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[91] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned i = 0; i < 16; ++i) CHECK(pixels[i * 4 + 1] == 255);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);

    // Retained queue state is reused, but the new submission must bind and
    // retain its resources again; the fresh target verifies the actual draw.
    resources.textures[92] = [device newTextureWithDescriptor:descriptor];
    auto next_state = texture_state(92, 4, 4);
    auto next_target = texture_object(next_state);
    const NVNtexture* next_targets[]{&next_target};
    const auto before = backend->statistics();
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, next_targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    handle = recorder.finish();
    fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    CHECK(backend->statistics().stage_binds == before.stage_binds + 2);
    CHECK(resources.texture_writes == 4 && resources.buffer_writes == 1);
    [resources.textures[92] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned i = 0; i < 16; ++i) CHECK(pixels[i * 4 + 1] == 255);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);

    // A vertex-stage storage write is ordered inside the pass: the reader
    // waits on an encoder barrier after vertex work and the pass stays open.
    NVNprogram vertex_writer{};
    CHECK(load<Entry::nvnProgramInitialize>()(&vertex_writer, &recorder.device));
    const NVNshaderData vertex_write_shaders[]{{17, nullptr}, {17, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&vertex_writer, 2, vertex_write_shaders));
    resources.textures[93] = [device newTextureWithDescriptor:descriptor];
    auto vertex_state = texture_state(93, 4, 4);
    auto vertex_target = texture_object(vertex_state);
    const NVNtexture* vertex_targets[]{&vertex_target};
    std::memset(resources.memory.contents, 0, 4);
    const auto passes = backend->statistics().render_passes;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, vertex_targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands,
        NVN_SHADER_STAGE_VERTEX, 0, TestResources::base, 4);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands,
        NVN_SHADER_STAGE_FRAGMENT, 0, TestResources::base, 4);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &vertex_writer,
        NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &reader, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    handle = recorder.finish();
    fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    CHECK(backend->statistics().render_passes == passes + 1);
    CHECK(resources.texture_writes == 5 && resources.buffer_writes == 2);
    CHECK(*static_cast<const std::uint32_t*>(resources.memory.contents) == 16);
    [resources.textures[93] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned i = 0; i < 16; ++i) CHECK(pixels[i * 4 + 1] == 255);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&vertex_writer);
    load<Entry::nvnProgramFinalize>()(&writer);
    load<Entry::nvnProgramFinalize>()(&reader);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// The first segment must execute while later work is gated, without retiring
// the guest fence or applying final effects. Uniform prefixes stay immutable.
void test_segmented_uniforms_and_effects(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    const char* old_cut = std::getenv("ARMIGO_NVN_CUT");
    const std::string saved_cut = old_cut ? old_cut : "";
    const char* old_thread = std::getenv("ARMIGO_NVN_ENCODE_THREAD");
    const std::string saved_thread = old_thread ? old_thread : "";
    setenv("ARMIGO_NVN_CUT", "16", 1);
    // The texture hook below queues GPU events from the worker, so the
    // worker's own commits must order them: no encode thread here.
    setenv("ARMIGO_NVN_ENCODE_THREAD", "0", 1);
    auto backend = replay::create_replay_backend(device, queue, resources);
    if (old_cut) setenv("ARMIGO_NVN_CUT", saved_cut.c_str(), 1);
    else unsetenv("ARMIGO_NVN_CUT");
    if (old_thread) setenv("ARMIGO_NVN_ENCODE_THREAD", saved_thread.c_str(), 1);
    else unsetenv("ARMIGO_NVN_ENCODE_THREAD");
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[101] = [device newTextureWithDescriptor:descriptor];
    resources.textures[102] = [device newTextureWithDescriptor:descriptor];
    auto first_state = texture_state(101, 4, 4), second_state = texture_state(102, 4, 4);
    auto first_target = texture_object(first_state), second_target = texture_object(second_state);
    const NVNtexture* first_targets[]{&first_target};
    const NVNtexture* second_targets[]{&second_target};
    id<MTLSharedEvent> prefix_done = [device newSharedEvent];
    id<MTLSharedEvent> tail_gate = [device newSharedEvent];
    bool gated = false;
    resources.on_texture_resolve = [&](const TextureState& state) {
        if (state.serial != 102 || gated) return;
        gated = true;
        // This lookup follows the render-target cut. Queue order lets the test
        // observe prefix completion while holding every following segment.
        [queue signalEvent:prefix_done value:1];
        [queue waitForEvent:tail_gate value:1];
    };
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{5, nullptr}, {5, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    std::memset(resources.memory.contents, 0, resources.memory.length);
    const std::array<float, 4> red{1, 0, 0, 1};
    const std::array<float, 4> straddling{1, 1, 7, 8};
    constexpr auto uniform = TestResources::base + 0x100;
    auto* event_value = reinterpret_cast<volatile std::uint32_t*>(
        static_cast<std::byte*>(resources.memory.contents) + 0x4000);
    NVNevent event{};
    reinterpret_cast<EventState*>(&event)->value = event_value;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, first_targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, uniform, 128);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 0, 16, red.data());
    load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
    for (unsigned i = 0; i < 16; ++i)
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    // Keep versions live across the first cut; a straddling update must not
    // change the old 16-byte prefix while the first draws read it.
    load<Entry::nvnCommandBufferSignalEvent>()(&recorder.commands, &event,
        NVN_EVENT_SIGNAL_MODE_WRITE, NVN_EVENT_SIGNAL_LOCATION_BOTTOM, 0, 99);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, second_targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 8, 16, straddling.data());
    for (unsigned i = 0; i < 16; ++i)
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
        TestResources::base + 0x3000);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, first_targets, nullptr, nullptr, nullptr);
    // Reading a deferred report still drains its committed prefix before copy.
    load<Entry::nvnCommandBufferCopyBufferToBuffer>()(&recorder.commands,
        TestResources::base + 0x3000, TestResources::base + 0x3100, sizeof(NVNcounterData), 0);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (prefix_done.signaledValue != 1 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(prefix_done.signaledValue == 1);
    CHECK(backend->completed_fence() == 0);
    CHECK(__atomic_load_n(event_value, __ATOMIC_SEQ_CST) == 0);
    std::array<std::uint8_t, 64> pixels{};
    if (prefix_done.signaledValue == 1) {
        [resources.textures[101] getBytes:pixels.data() bytesPerRow:16
            fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
        for (unsigned i = 0; i < 16; ++i)
            CHECK(pixels[i * 4] == 255 && pixels[i * 4 + 2] == 0);
    }
    tail_gate.signaledValue = 1;
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    CHECK(backend->statistics().synchronize_gpu_waits == 1);
    CHECK(__atomic_load_n(event_value, __ATOMIC_SEQ_CST) == 99);
    [resources.textures[102] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned i = 0; i < 16; ++i)
        CHECK(pixels[i * 4] == 255 && pixels[i * 4 + 2] == 255);
    NVNcounterData report{}, copied{};
    std::memcpy(&report, static_cast<std::byte*>(resources.memory.contents) + 0x3000, sizeof report);
    std::memcpy(&copied, static_cast<std::byte*>(resources.memory.contents) + 0x3100, sizeof copied);
    CHECK(report.counter == 512 && report.timestamp != 0);
    CHECK(std::memcmp(&report, &copied, sizeof report) == 0);
    CHECK(std::memcmp(static_cast<std::byte*>(resources.memory.contents) + 0x108,
                      straddling.data(), sizeof straddling) == 0);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A depth-only draw and later depth tests remain in one pass across a barrier.
// Sampling that depth through a view must instead observe a completed store.
void test_depth_barrier_continuation(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                                     bool sample_depth, bool reflected = false, bool clear_color = false,
                                     bool sample_color_first = false) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[111] = [device newTextureWithDescriptor:descriptor];
    descriptor.pixelFormat = MTLPixelFormatDepth32Float;
    resources.textures[112] = [device newTextureWithDescriptor:descriptor];
    resources.texture_ids[1] = [resources.textures[112] newTextureViewWithPixelFormat:MTLPixelFormatDepth32Float];
    resources.texture_ids[2] = [resources.textures[111] newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
    auto color_state = texture_state(111, 4, 4), depth_state = texture_state(112, 4, 4);
    depth_state.description.format = NVN_FORMAT_DEPTH32F;
    auto color = texture_object(color_state), depth = texture_object(depth_state);
    const NVNtexture* targets[]{&color};
    Recorder recorder;
    NVNprogram program{}, sampled{}, color_reader{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    CHECK(load<Entry::nvnProgramInitialize>()(&sampled, &recorder.device));
    CHECK(load<Entry::nvnProgramInitialize>()(&color_reader, &recorder.device));
    const NVNshaderData color_reader_shaders[]{{4, nullptr}, {4, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&color_reader, 2, color_reader_shaders));
    const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
    const NVNshaderData depth_shaders[]{{reflected ? 14u : 13u, nullptr}, {reflected ? 14u : 13u, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    CHECK(load<Entry::nvnProgramSetShaders>()(&sampled, 2, depth_shaders));
    NVNdepthStencilState depth_test{};
    load<Entry::nvnDepthStencilStateSetDefaults>()(&depth_test);
    load<Entry::nvnDepthStencilStateSetDepthTestEnable>()(&depth_test, 1);
    load<Entry::nvnDepthStencilStateSetDepthWriteEnable>()(&depth_test, 1);
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&depth_test, NVN_DEPTH_FUNC_ALWAYS);
    NVNchannelMaskState channels{};
    load<Entry::nvnChannelMaskStateSetDefaults>()(&channels);
    load<Entry::nvnChannelMaskStateSetChannelMask>()(&channels, 0, 0, 0, 0, 0);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, &depth, nullptr);
    load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands, 1.0f, 1, 0, 0);
    if (clear_color) {
        const float blue[]{0, 0, 1, 1};
        load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, blue, 15);
    }
    // The depth-only draw before the barrier may sample the color attachment
    // that a draw after the barrier writes: the barrier must then end the pass.
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, sample_color_first ? &color_reader : &program,
        NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    if (sample_color_first)
        load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, 2);
    load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &channels);
    load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &depth_test);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferBarrier>()(&recorder.commands, NVN_BARRIER_ORDER_FRAGMENTS_BIT |
        NVN_BARRIER_INVALIDATE_TEXTURE_BIT);
    if (sample_color_first)
        load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnChannelMaskStateSetDefaults>()(&channels);
    load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &channels);
    load<Entry::nvnDepthStencilStateSetDepthWriteEnable>()(&depth_test, 0);
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&depth_test, NVN_DEPTH_FUNC_EQUAL);
    load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &depth_test);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    if (sample_depth) {
        load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &sampled, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, 1);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    }
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    CHECK(backend->statistics().render_passes == (sample_depth || clear_color || sample_color_first ? 2 : 1));
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[111] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned i = 0; i < 16; ++i)
        CHECK(pixels[i * 4] == 0 && pixels[i * 4 + 1] == 255 && pixels[i * 4 + 2] == 0);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnProgramFinalize>()(&sampled);
    load<Entry::nvnProgramFinalize>()(&color_reader);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// Metal has no depth-bounds test: an enabled one is approximated as disabled,
// so the draws after it still render instead of the submission being abandoned.
void test_depth_bounds_draw(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[131] = [device newTextureWithDescriptor:descriptor];
    auto color_state = texture_state(131, 4, 4);
    auto color = texture_object(color_state);
    const NVNtexture* targets[]{&color};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferSetDepthBounds>()(&recorder.commands, 1, 0.25f, 0.75f);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0 && backend->statistics().render_passes == 1);
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[131] getBytes:pixels.data() bytesPerRow:16 fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned i = 0; i < 16; ++i)
        CHECK(pixels[i * 4] == 0 && pixels[i * 4 + 1] == 255 && pixels[i * 4 + 2] == 0);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A program without a fragment stage still rasterizes: the draw writes depth
// and leaves the bound color targets unchanged.
// Pipelines come from a pipeline archive instead of compiling: a session that
// captures writes one; a later session takes the same pipeline from it (exact
// hit), a pipeline for another color format from its open form (specialized),
// and compiles one with other non-color state (a miss). Every draw renders as
// it does with archives off.
struct ArchiveDraws {
    std::array<std::array<std::uint8_t, 64>, 2> rgba8{};
    std::array<std::uint16_t, 64> rgba16f{};
    replay::PipelineWarmupReport report;
};

ArchiveDraws archive_draws(id<MTLDevice> device, id<MTL4CommandQueue> queue, const std::filesystem::path& root,
                           bool archive, bool capture) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    replay::PipelineWarmupConfig config;
    config.warm = false;
    if (archive) {
        config.archive_root = root.string();
        config.archive_identity = "archive test";
        config.archive_capture = capture;
    }
    backend->configure_pipeline_warmup(config);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[161] = [device newTextureWithDescriptor:descriptor];
    resources.textures[162] = [device newTextureWithDescriptor:descriptor];
    descriptor.pixelFormat = MTLPixelFormatRGBA16Float;
    resources.textures[163] = [device newTextureWithDescriptor:descriptor];
    auto first_state = texture_state(161, 4, 4), second_state = texture_state(162, 4, 4),
         wide_state = texture_state(163, 4, 4);
    wide_state.description.format = NVN_FORMAT_RGBA16F;
    auto first = texture_object(first_state), second = texture_object(second_state), wide = texture_object(wide_state);
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    NVNmultisampleState coverage{};
    load<Entry::nvnMultisampleStateSetDefaults>()(&coverage);
    load<Entry::nvnMultisampleStateSetAlphaToCoverageEnable>()(&coverage, 1);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    const NVNtexture* first_targets[]{&first};
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, first_targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    if (!capture) {
        // The open form leaves blending and the write mask open too: additive
        // blending into red and alpha only must survive the specialization.
        NVNcolorState blending{}, opaque{};
        load<Entry::nvnColorStateSetDefaults>()(&blending);
        load<Entry::nvnColorStateSetDefaults>()(&opaque);
        load<Entry::nvnColorStateSetBlendEnable>()(&blending, 0, 1);
        NVNblendState additive{};
        load<Entry::nvnBlendStateSetDefaults>()(&additive);
        load<Entry::nvnBlendStateSetBlendFunc>()(&additive, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ONE,
                                                 NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ONE);
        NVNchannelMaskState red_alpha{}, all_channels{};
        load<Entry::nvnChannelMaskStateSetDefaults>()(&red_alpha);
        load<Entry::nvnChannelMaskStateSetDefaults>()(&all_channels);
        load<Entry::nvnChannelMaskStateSetChannelMask>()(&red_alpha, 0, 1, 0, 0, 1);
        const float base[]{0.25f, 0.5f, 0.75f, 0.5f};
        const NVNtexture* wide_targets[]{&wide};
        load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, wide_targets, nullptr, nullptr,
                                                        nullptr);
        load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, base, 15);
        load<Entry::nvnCommandBufferBindColorState>()(&recorder.commands, &blending);
        load<Entry::nvnCommandBufferBindBlendState>()(&recorder.commands, &additive);
        load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &red_alpha);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        load<Entry::nvnCommandBufferBindColorState>()(&recorder.commands, &opaque);
        load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &all_channels);
        const NVNtexture* second_targets[]{&second};
        load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, second_targets, nullptr, nullptr,
                                                        nullptr);
        load<Entry::nvnCommandBufferBindMultisampleState>()(&recorder.commands, &coverage);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    }
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    ArchiveDraws result;
    [resources.textures[161] getBytes:result.rgba8[0].data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    [resources.textures[162] getBytes:result.rgba8[1].data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    [resources.textures[163] getBytes:result.rgba16f.data() bytesPerRow:32
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    result.report = backend->finish_pipeline_warmup();
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
    return result;
}

void test_pipeline_archive(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    const auto root = std::filesystem::temp_directory_path() /
        ("armigo-pso-archive-test-" + std::to_string(getpid()));
    std::filesystem::remove_all(root);
    // Capture: only the RGBA8 pipeline and its open form go into an archive.
    const auto captured = archive_draws(device, queue, root, true, true);
    CHECK(!captured.report.archive.empty() && std::filesystem::exists(captured.report.archive));
    // Lookup: the same pipeline is an exact hit, the RGBA16F one (blending,
    // red and alpha only) specializes the open form, alpha-to-coverage is a
    // miss that compiles.
    const auto looked_up = archive_draws(device, queue, root, true, false);
    CHECK(looked_up.report.archive_hits == 2);
    CHECK(looked_up.report.archive_open_hits == 1);
    CHECK(looked_up.report.archive_misses == 1);
    const auto plain = archive_draws(device, queue, root, false, false);
    CHECK(plain.report.archive_hits == 0 && plain.report.archive_misses == 0);
    CHECK(looked_up.rgba8 == plain.rgba8);
    CHECK(looked_up.rgba16f == plain.rgba16f);
    for (unsigned i = 0; i < 16; ++i) {
        CHECK(plain.rgba8[0][i * 4] == 0 && plain.rgba8[0][i * 4 + 1] == 255 && plain.rgba8[1][i * 4 + 1] == 255);
        // Red 0.25 + 0 (blended), green 0.5 and blue 0.75 kept (masked).
        CHECK(plain.rgba16f[i * 4] == 0x3400 && plain.rgba16f[i * 4 + 1] == 0x3800 &&
              plain.rgba16f[i * 4 + 2] == 0x3a00);
    }
    std::filesystem::remove_all(root);
}

// A clear pipeline that compiles at first use is recorded as a K row, so the
// next launch's warm-up builds it. The configuration (RGBA16F, red and alpha
// only) is one no other test clears with.
void test_clear_pipeline_record(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    const auto record = std::filesystem::temp_directory_path() /
        ("armigo-clear-record-test-" + std::to_string(getpid()) + ".tsv");
    std::filesystem::remove(record);
    {
        TestResources resources(device);
        auto backend = replay::create_replay_backend(device, queue, resources);
        replay::PipelineWarmupConfig config;
        config.warm = false;
        config.record_manifest = record.string();
        backend->configure_pipeline_warmup(config);
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
            width:4 height:4 mipmapped:NO];
        descriptor.storageMode = MTLStorageModePrivate;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        resources.textures[171] = [device newTextureWithDescriptor:descriptor];
        auto state = texture_state(171, 4, 4);
        state.description.format = NVN_FORMAT_RGBA16F;
        auto target = texture_object(state);
        Recorder recorder;
        const NVNcopyRegion middle{1, 1, 0, 2, 2, 1};
        const float value[]{1, 1, 1, 1};
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        load<Entry::nvnCommandBufferClearTexture>()(&recorder.commands, &target, nullptr, &middle, value, 9);
        const auto handle = recorder.finish();
        const auto fence = backend->submit({&handle, 1});
        CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
        CHECK(!backend->last_error() && backend->statistics().skipped == 0);
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
        load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
    }
    std::ifstream in(record);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const auto expected = "K\tclear_vertex\tclear_color_frag\t" + std::to_string(MTLPixelFormatRGBA16Float) + "\t" +
        std::to_string(MTLColorWriteMaskRed | MTLColorWriteMaskAlpha) + "\t0\t0\t1\n";
    CHECK(text.find(expected) != std::string::npos);
    if (text.find(expected) == std::string::npos) std::fprintf(stderr, "clear record: %s\n", text.c_str());
    std::filesystem::remove(record);
}

void test_vertex_only_depth_draw(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[141] = [device newTextureWithDescriptor:descriptor];
    descriptor.pixelFormat = MTLPixelFormatDepth32Float;
    resources.textures[142] = [device newTextureWithDescriptor:descriptor];
    auto color_state = texture_state(141, 4, 4), depth_state = texture_state(142, 4, 4);
    depth_state.description.format = NVN_FORMAT_DEPTH32F;
    auto color = texture_object(color_state), depth = texture_object(depth_state);
    const NVNtexture* targets[]{&color};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{31, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 1, shaders));
    NVNdepthStencilState depth_test{};
    load<Entry::nvnDepthStencilStateSetDefaults>()(&depth_test);
    load<Entry::nvnDepthStencilStateSetDepthTestEnable>()(&depth_test, 1);
    load<Entry::nvnDepthStencilStateSetDepthWriteEnable>()(&depth_test, 1);
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&depth_test, NVN_DEPTH_FUNC_LESS);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, &depth, nullptr);
    load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands, 1.0f, 1, 0, 0);
    const float blue[]{0, 0, 1, 1};
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, blue, 15);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &depth_test);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    std::array<float, 16> depths{};
    [resources.textures[142] getBytes:depths.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (const auto value : depths) CHECK(value < 1.0f);
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[141] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned i = 0; i < 16; ++i)
        CHECK(pixels[i * 4] == 0 && pixels[i * 4 + 1] == 0 && pixels[i * 4 + 2] == 255);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void check_stencil_plane(id<MTLDevice> device, id<MTLTexture> texture, std::uint32_t expected) {
    NSError* error = nil;
    auto* library = [device newLibraryWithSource:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "kernel void read_stencil(texture2d<uint, access::read> source [[texture(0)]], "
         "device uint* result [[buffer(0)]], uint2 p [[thread_position_in_grid]]) "
         "{ result[p.y * 4 + p.x] = source.read(p).r; }"
        options:nil error:&error];
    id<MTLComputePipelineState> pipeline = library ? [device newComputePipelineStateWithFunction:
        [library newFunctionWithName:@"read_stencil"] error:&error] : nil;
    id<MTLTexture> view = [texture newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
    id<MTLBuffer> sampled = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
    id<MTLBuffer> copied = [device newBufferWithLength:1024 options:MTLResourceStorageModeShared];
    CHECK(pipeline && view && sampled && copied);
    if (!pipeline || !view || !sampled || !copied) return;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> compute = [command computeCommandEncoder];
    [compute setComputePipelineState:pipeline];
    [compute setTexture:view atIndex:0];
    [compute setBuffer:sampled offset:0 atIndex:0];
    [compute dispatchThreads:MTLSizeMake(4, 4, 1) threadsPerThreadgroup:MTLSizeMake(4, 4, 1)];
    [compute endEncoding];
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
        sourceSize:MTLSizeMake(4, 4, 1) toBuffer:copied destinationOffset:0
        destinationBytesPerRow:256 destinationBytesPerImage:1024 options:MTLBlitOptionStencilFromDepthStencil];
    [blit endEncoding];
    [command commit];
    [command waitUntilCompleted];
    CHECK(command.status == MTLCommandBufferStatusCompleted && !command.error);
    const auto* samples = static_cast<const std::uint32_t*>(sampled.contents);
    const auto* bytes = static_cast<const std::uint8_t*>(copied.contents);
    for (unsigned y = 0; y != 4; ++y)
        for (unsigned x = 0; x != 4; ++x) {
            CHECK(samples[y * 4 + x] == expected);
            CHECK(bytes[y * 256 + x] == expected);
        }
}

// Packed stencil survives draws with stencil testing disabled. Later tests
// and masked clears observe its contents, while sample reports remain cumulative.
void test_inactive_packed_stencil(id<MTLDevice> device, id<MTL4CommandQueue> queue,
                                 bool write_depth, bool stencil_matches, bool masked_clear,
                                 bool later_submission = false, bool drain = false, bool private_storage = false,
                                 bool unrelated_clear = false) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[181] = [device newTextureWithDescriptor:descriptor];
    descriptor.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
    descriptor.usage |= MTLTextureUsagePixelFormatView;
    if (private_storage) descriptor.storageMode = MTLStorageModePrivate;
    resources.textures[182] = [device newTextureWithDescriptor:descriptor];
    if (unrelated_clear) resources.textures[183] = [device newTextureWithDescriptor:descriptor];
    auto color_state = texture_state(181, 4, 4), depth_state = texture_state(182, 4, 4);
    depth_state.description.format = NVN_FORMAT_DEPTH32F_STENCIL8;
    auto color = texture_object(color_state), depth = texture_object(depth_state);
    auto other_state = depth_state;
    other_state.serial = 183;
    auto other_depth = texture_object(other_state);
    const NVNtexture* targets[]{&color};
    Recorder recorder;
    const float blue[]{0, 0, 1, 1};
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, &depth, nullptr);
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, blue, 15);
    load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands, 1.0f, 1, 0xa7, 0xff);
    const auto initial = recorder.finish();
    const auto initialized = backend->submit({&initial, 1});
    CHECK(initialized && backend->wait_fence(initialized, 10'000'000'000ull));
    CHECK(!backend->last_error());

    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    NVNprogram drain_program{};
    if (drain) {
        CHECK(load<Entry::nvnProgramInitialize>()(&drain_program, &recorder.device));
        const NVNshaderData uniform_shaders[]{{5, nullptr}, {5, nullptr}};
        CHECK(load<Entry::nvnProgramSetShaders>()(&drain_program, 2, uniform_shaders));
        const float green[]{0, 1, 0, 1};
        std::memcpy(static_cast<std::byte*>(resources.memory.contents) + 0x200, green, sizeof green);
        resources.require_completion = true;
        resources.pending_write = false;
    }
    NVNdepthStencilState inactive{}, observe{};
    load<Entry::nvnDepthStencilStateSetDefaults>()(&inactive);
    load<Entry::nvnDepthStencilStateSetDepthTestEnable>()(&inactive, 1);
    load<Entry::nvnDepthStencilStateSetDepthWriteEnable>()(&inactive, write_depth);
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&inactive, NVN_DEPTH_FUNC_ALWAYS);
    load<Entry::nvnDepthStencilStateSetDefaults>()(&observe);
    load<Entry::nvnDepthStencilStateSetDepthTestEnable>()(&observe, 1);
    load<Entry::nvnDepthStencilStateSetDepthWriteEnable>()(&observe, 0);
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&observe,
        write_depth ? NVN_DEPTH_FUNC_EQUAL : NVN_DEPTH_FUNC_LESS);
    load<Entry::nvnDepthStencilStateSetStencilTestEnable>()(&observe, 1);
    load<Entry::nvnDepthStencilStateSetStencilFunc>()(&observe, NVN_FACE_FRONT_AND_BACK, NVN_STENCIL_FUNC_EQUAL);
    NVNchannelMaskState hidden{}, shown{};
    load<Entry::nvnChannelMaskStateSetDefaults>()(&hidden);
    load<Entry::nvnChannelMaskStateSetChannelMask>()(&hidden, 0, 0, 0, 0, 0);
    load<Entry::nvnChannelMaskStateSetDefaults>()(&shown);
    const unsigned iterations = later_submission || drain ? 1 : 3;
    for (unsigned iteration = 0; iteration != iterations; ++iteration) {
        if (iteration) {
            const auto reset = backend->submit({&initial, 1});
            CHECK(reset && backend->wait_fence(reset, 10'000'000'000ull));
        }
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        if (unrelated_clear) {
            load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 0, nullptr, nullptr, &other_depth, nullptr);
            load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands, 1.0f, 0, 0x42, 0xff);
        }
        load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, &depth, nullptr);
        load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &hidden);
        load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &inactive);
        load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
            TestResources::base + 0x700);
        NVNcommandHandle prefix = 0;
        if (later_submission) {
            prefix = recorder.finish();
            const auto submitted = backend->submit({&prefix, 1});
            CHECK(submitted && backend->wait_fence(submitted, 10'000'000'000ull));
            load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        }
        if (drain) {
            // Resolving this uniform after the report invokes the resource
            // completion hook while the old pass descriptor is still in use.
            load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &drain_program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
            load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT,
                0, TestResources::base + 0x200, 16);
            load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        }
        if (masked_clear)
            load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands, 0.0f, 0, 9, 0x0f);
        const auto expected_stencil = masked_clear ? 0xa9 : 0xa7;
        load<Entry::nvnCommandBufferSetStencilRef>()(&recorder.commands, NVN_FACE_FRONT_AND_BACK,
            stencil_matches ? expected_stencil : expected_stencil ^ 1);
        load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &observe);
        load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &shown);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
            TestResources::base + 0x720);
        const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
        load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &color, nullptr, &whole,
            TestResources::base + 0x2000, 0);
        const auto handle = recorder.finish();
        const auto fence = backend->submit({&handle, 1});
        CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
        CHECK(!backend->last_error() && backend->statistics().skipped == 0);
        const auto* bytes = static_cast<const std::byte*>(resources.memory.contents);
        NVNcounterData first{}, second{};
        std::memcpy(&first, bytes + 0x700, sizeof first);
        std::memcpy(&second, bytes + 0x720, sizeof second);
        CHECK(first.counter == 16);
        CHECK(second.counter == 16u + (drain ? 16u : 0u) + (stencil_matches ? 16u : 0u));
        const auto* pixels = reinterpret_cast<const std::uint8_t*>(bytes + 0x2000);
        for (unsigned pixel = 0; pixel != 16; ++pixel) {
            CHECK(pixels[pixel * 4] == 0);
            CHECK(pixels[pixel * 4 + 1] == (stencil_matches ? 255 : 0));
            CHECK(pixels[pixel * 4 + 2] == (stencil_matches ? 0 : 255));
            CHECK(pixels[pixel * 4 + 3] == 255);
        }
        check_stencil_plane(device, resources.textures[182], expected_stencil);
        if (unrelated_clear) check_stencil_plane(device, resources.textures[183], 0x42);
        if (drain) CHECK(resources.coherence_waits >= 2);
        if (prefix) load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, prefix);
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, initial);
    if (drain) load<Entry::nvnProgramFinalize>()(&drain_program);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A conditional block whose counters this submission reports is predicated on
// the GPU: its draws render only when the occlusion query saw samples, and the
// replay never waits for the GPU. A block longer than its argument records
// still renders correctly (it is resolved on the CPU).
enum class ConditionalContract {
    ordinary, occluded, report_in_block, report_after_block, storage_write,
    nonzero_reference, two_reports, pending_reference_copy, stencil_depth_fail, depth_write,
};

void test_gpu_conditional_render(id<MTLDevice> device, id<MTL4CommandQueue> queue, bool visible,
                                 unsigned draws, ConditionalContract which = ConditionalContract::ordinary) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[151] = [device newTextureWithDescriptor:descriptor];
    descriptor.pixelFormat = which == ConditionalContract::stencil_depth_fail ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatDepth32Float;
    resources.textures[152] = [device newTextureWithDescriptor:descriptor];
    auto color_state = texture_state(151, 4, 4), depth_state = texture_state(152, 4, 4);
    depth_state.description.format = which == ConditionalContract::stencil_depth_fail ? NVN_FORMAT_DEPTH32F_STENCIL8 : NVN_FORMAT_DEPTH32F;
    auto color = texture_object(color_state), depth = texture_object(depth_state);
    const NVNtexture* targets[]{&color};
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    if (which == ConditionalContract::nonzero_reference) {
        const std::uint64_t reference = 16;
        std::memcpy(bytes + 0x710, &reference, sizeof reference);
    }
    const std::uint64_t nonzero = 16;
    std::memcpy(bytes + 0x780, &nonzero, sizeof nonzero);
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    NVNprogram writer{};
    if (which == ConditionalContract::storage_write) {
        CHECK(load<Entry::nvnProgramInitialize>()(&writer, &recorder.device));
        const NVNshaderData writer_shaders[]{{6, nullptr}, {6, nullptr}};
        CHECK(load<Entry::nvnProgramSetShaders>()(&writer, 2, writer_shaders));
    }
    // The query draw tests depth without writing anything; the guarded draws
    // pass the depth test and write green.
    NVNdepthStencilState query_depth{}, draw_depth{};
    load<Entry::nvnDepthStencilStateSetDefaults>()(&query_depth);
    load<Entry::nvnDepthStencilStateSetDepthTestEnable>()(&query_depth, 1);
    load<Entry::nvnDepthStencilStateSetDepthWriteEnable>()(&query_depth, 0);
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&query_depth,
        visible ? NVN_DEPTH_FUNC_ALWAYS : NVN_DEPTH_FUNC_NEVER);
    load<Entry::nvnDepthStencilStateSetDefaults>()(&draw_depth);
    if (which != ConditionalContract::ordinary) {
        load<Entry::nvnDepthStencilStateSetDepthTestEnable>()(&draw_depth, 1);
        load<Entry::nvnDepthStencilStateSetDepthWriteEnable>()(&draw_depth, which == ConditionalContract::depth_write);
        load<Entry::nvnDepthStencilStateSetDepthFunc>()(&draw_depth, NVN_DEPTH_FUNC_ALWAYS);
    }
    if (which == ConditionalContract::occluded || which == ConditionalContract::stencil_depth_fail) {
        // A real depth occluder: the full-screen triangle at depth zero fails
        // LESS against zero, and passes against one on the visible control.
        load<Entry::nvnDepthStencilStateSetDepthFunc>()(&query_depth, NVN_DEPTH_FUNC_LESS);
        load<Entry::nvnDepthStencilStateSetDepthFunc>()(&draw_depth, NVN_DEPTH_FUNC_LESS);
    }
    if (which == ConditionalContract::stencil_depth_fail) {
        load<Entry::nvnDepthStencilStateSetStencilTestEnable>()(&draw_depth, 1);
        load<Entry::nvnDepthStencilStateSetStencilFunc>()(&draw_depth, NVN_FACE_FRONT_AND_BACK, NVN_STENCIL_FUNC_ALWAYS);
        load<Entry::nvnDepthStencilStateSetStencilOp>()(&draw_depth, NVN_FACE_FRONT_AND_BACK,
            NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_REPLACE, NVN_STENCIL_OP_KEEP);
    }
    NVNchannelMaskState hidden{}, shown{};
    load<Entry::nvnChannelMaskStateSetDefaults>()(&hidden);
    load<Entry::nvnChannelMaskStateSetChannelMask>()(&hidden, 0, 0, 0, 0, 0);
    load<Entry::nvnChannelMaskStateSetDefaults>()(&shown);
    const float black[]{0, 0, 0, 1};
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, &depth, nullptr);
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, black, 15);
    load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands,
        (which == ConditionalContract::occluded || which == ConditionalContract::stencil_depth_fail) ? (visible ? 1.0f : 0.0f) : 0.5f,
        1, 0, which == ConditionalContract::stencil_depth_fail ? 0xff : 0);
    if (which == ConditionalContract::pending_reference_copy)
        load<Entry::nvnCommandBufferCopyBufferToBuffer>()(&recorder.commands,
            TestResources::base + 0x780, TestResources::base + 0x710, 8, 0);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &hidden);
    load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &query_depth);
    load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    // The second counter is a zero/nonzero reference or another report.
    // None of these predicates permits inferring the guarded draw's coverage.
    load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
        TestResources::base + 0x700);
    if (which == ConditionalContract::two_reports)
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
            TestResources::base + 0x710);
    load<Entry::nvnCommandBufferSetRenderEnableConditional>()(&recorder.commands,
        NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_NOT_EQUAL, TestResources::base + 0x700);
    load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &shown);
    load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &draw_depth);
    if (which == ConditionalContract::stencil_depth_fail)
        load<Entry::nvnCommandBufferSetStencilRef>()(&recorder.commands, NVN_FACE_FRONT_AND_BACK, 1);
    for (unsigned index = 0; index != draws; ++index) {
        // A later writable shader disqualifies the entire block, including
        // earlier read-only draws. The atomic output must stay zero if hidden.
        if (which == ConditionalContract::storage_write && index == 1) {
            load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &writer, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
            load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT,
                0, TestResources::base + 0x800, 4);
        }
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    }
    if (which == ConditionalContract::report_in_block)
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
            TestResources::base + 0x740);
    load<Entry::nvnCommandBufferSetRenderEnable>()(&recorder.commands, 1);
    if (which == ConditionalContract::report_after_block)
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
            TestResources::base + 0x740);
    if (which != ConditionalContract::ordinary)
        load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
    if (which == ConditionalContract::stencil_depth_fail) {
        // Observe stencil after the sample reset: a rejected hidden draw must
        // not have executed its depth-fail REPLACE operation.
        NVNdepthStencilState observe{};
        load<Entry::nvnDepthStencilStateSetDefaults>()(&observe);
        load<Entry::nvnDepthStencilStateSetStencilTestEnable>()(&observe, 1);
        load<Entry::nvnDepthStencilStateSetStencilFunc>()(&observe, NVN_FACE_FRONT_AND_BACK, NVN_STENCIL_FUNC_EQUAL);
        load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &observe);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
            TestResources::base + 0x760);
    }
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &color, nullptr, &whole,
        TestResources::base + 0x2000, 0);
    const auto handle = recorder.finish();
    const auto waits = backend->statistics().synchronize_gpu_waits;
    const auto fence = backend->submit({&handle, 1});
    if (!fence) std::fprintf(stderr, "gpu conditional: %s\n", backend->last_error().detail.c_str());
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    // Only a block longer than its argument records waits for the GPU.
    CHECK((backend->statistics().synchronize_gpu_waits == waits) == (draws <= 128));
    const auto* output = reinterpret_cast<const std::uint8_t*>(bytes + 0x2000);
    const bool rendered = which == ConditionalContract::two_reports ? false : (which == ConditionalContract::nonzero_reference || which == ConditionalContract::pending_reference_copy) ? !visible : visible;
    for (unsigned pixel = 0; pixel != 16; ++pixel) {
        CHECK(output[pixel * 4] == ((which == ConditionalContract::storage_write && rendered) ? 255 : 0));
        CHECK(output[pixel * 4 + 2] == 0);
        CHECK(output[pixel * 4 + 1] == (rendered && which != ConditionalContract::storage_write ? 255 : 0));
    }
    NVNcounterData samples{};
    std::memcpy(&samples, bytes + 0x700, sizeof samples);
    CHECK(samples.counter == (visible ? 16u : 0u));
    if (which == ConditionalContract::report_in_block || which == ConditionalContract::report_after_block) {
        NVNcounterData after{};
        std::memcpy(&after, bytes + 0x740, sizeof after);
        CHECK(after.counter == (visible ? 16u * (draws + 1) : 0u));
    }
    if (which == ConditionalContract::stencil_depth_fail) {
        NVNcounterData downstream{};
        std::memcpy(&downstream, bytes + 0x760, sizeof downstream);
        CHECK(downstream.counter == 0);
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    if (which == ConditionalContract::storage_write) {
        std::uint32_t writes = 0;
        std::memcpy(&writes, bytes + 0x800, sizeof writes);
        CHECK(writes == (visible ? 16u * (draws - 1) : 0u));
        load<Entry::nvnProgramFinalize>()(&writer);
    }
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// Counter operations inside a pass that counts samples keep the pass: a reset
// between two draws drops the first draw's samples, and the report counts the
// second draw only.
void test_counters_inside_pass(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[161] = [device newTextureWithDescriptor:descriptor];
    auto color_state = texture_state(161, 4, 4);
    auto color = texture_object(color_state);
    const NVNtexture* targets[]{&color};
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
        TestResources::base + 0x700);
    load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
        TestResources::base + 0x710);
    const auto handle = recorder.finish();
    const auto passes = backend->statistics().render_passes;
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    CHECK(backend->statistics().render_passes == passes + 1);
    std::array<NVNcounterData, 2> samples{};
    std::memcpy(samples.data(), bytes + 0x700, sizeof samples);
    CHECK(samples[0].counter == 32 && samples[1].counter == 16);
    CHECK(samples[0].timestamp != 0 && samples[1].timestamp >= samples[0].timestamp);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A stage left to background translation that fails leaves its program
// without that stage. A draw or dispatch that would use it is refused (and
// counted as skipped) instead of running without it: without its fragment
// stage the draw would write depth only.
void test_failed_pending_stage(id<MTLDevice> device, id<MTL4CommandQueue> queue, Stages& stages) {
    struct Failed final : PendingStage {
        std::shared_ptr<const platform::CompiledGraphicsStage> wait() override { return nullptr; }
    };
    set_deferred_program_resolver([](void* raw, const DeviceState*, std::span<const NVNshaderData> shaders,
                                     std::array<std::shared_ptr<const platform::CompiledGraphicsStage>, 6>& out,
                                     std::array<std::shared_ptr<PendingStage>, 6>& pending, std::string&) {
        auto& self = *static_cast<Stages*>(raw);
        if (shaders.size() == 2) {
            out[NVN_SHADER_STAGE_VERTEX] = self.vertex;
            pending[NVN_SHADER_STAGE_FRAGMENT] = std::make_shared<Failed>();
        } else {
            pending[NVN_SHADER_STAGE_COMPUTE] = std::make_shared<Failed>();
        }
        return true;
    }, &stages);
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[143] = [device newTextureWithDescriptor:descriptor];
    descriptor.pixelFormat = MTLPixelFormatDepth32Float;
    resources.textures[144] = [device newTextureWithDescriptor:descriptor];
    auto color_state = texture_state(143, 4, 4), depth_state = texture_state(144, 4, 4);
    depth_state.description.format = NVN_FORMAT_DEPTH32F;
    auto color = texture_object(color_state), depth = texture_object(depth_state);
    const NVNtexture* targets[]{&color};
    Recorder recorder;
    NVNprogram graphics{}, compute{};
    CHECK(load<Entry::nvnProgramInitialize>()(&graphics, &recorder.device));
    CHECK(load<Entry::nvnProgramInitialize>()(&compute, &recorder.device));
    const NVNshaderData graphics_shaders[]{{41, nullptr}, {42, nullptr}};
    const NVNshaderData compute_shaders[]{{43, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&graphics, 2, graphics_shaders));
    CHECK(load<Entry::nvnProgramSetShaders>()(&compute, 1, compute_shaders));
    NVNdepthStencilState depth_test{};
    load<Entry::nvnDepthStencilStateSetDefaults>()(&depth_test);
    load<Entry::nvnDepthStencilStateSetDepthTestEnable>()(&depth_test, 1);
    load<Entry::nvnDepthStencilStateSetDepthWriteEnable>()(&depth_test, 1);
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&depth_test, NVN_DEPTH_FUNC_LESS);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, &depth, nullptr);
    load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands, 1.0f, 1, 0, 0);
    const float blue[]{0, 0, 1, 1};
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, blue, 15);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &graphics, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &depth_test);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &compute, NVN_SHADER_STAGE_COMPUTE_BIT);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 2);
    std::array<float, 16> depths{};
    [resources.textures[144] getBytes:depths.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (const auto value : depths) CHECK(value == 1.0f);
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[143] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned i = 0; i < 16; ++i)
        CHECK(pixels[i * 4] == 0 && pixels[i * 4 + 1] == 0 && pixels[i * 4 + 2] == 255);
    const auto retained = retain_program(*reinterpret_cast<const ProgramState*>(&graphics));
    CHECK(retained && retained->stages[NVN_SHADER_STAGE_VERTEX] && !retained->stages[NVN_SHADER_STAGE_FRAGMENT] &&
          retained->failed_stages == 1u << NVN_SHADER_STAGE_FRAGMENT);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&graphics);
    load<Entry::nvnProgramFinalize>()(&compute);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
    set_program_resolver(Stages::resolve, &stages);
}

// Conditional blocks around the GPU predicate's edges. The query draw sees
// samples when `visible`; counters[1], zero in memory, is the reference.
enum class ConditionalCase {
    if_equal,              // RENDER_IF_EQUAL: renders only when nothing passed
    clear_in_block,        // a clear inside the block resolves it on the CPU
    reused_address,        // a timestamp report to the same address falls back
    across_submissions,    // the block stays open into the next submission
    stale_after_drain,     // indices of reports before a drain are not reused
    copy_after_condition,  // a copy into the counters after the conditional does not change the block
};

void test_conditional_case(id<MTLDevice> device, id<MTL4CommandQueue> queue, ConditionalCase which,
                           bool visible) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[171] = [device newTextureWithDescriptor:descriptor];
    descriptor.pixelFormat = MTLPixelFormatDepth32Float;
    resources.textures[172] = [device newTextureWithDescriptor:descriptor];
    auto color_state = texture_state(171, 4, 4), depth_state = texture_state(172, 4, 4);
    depth_state.description.format = NVN_FORMAT_DEPTH32F;
    auto color = texture_object(color_state), depth = texture_object(depth_state);
    const NVNtexture* targets[]{&color};
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    NVNdepthStencilState query_depth{}, draw_depth{};
    load<Entry::nvnDepthStencilStateSetDefaults>()(&query_depth);
    load<Entry::nvnDepthStencilStateSetDepthTestEnable>()(&query_depth, 1);
    load<Entry::nvnDepthStencilStateSetDepthWriteEnable>()(&query_depth, 0);
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&query_depth,
        visible ? NVN_DEPTH_FUNC_ALWAYS : NVN_DEPTH_FUNC_NEVER);
    load<Entry::nvnDepthStencilStateSetDefaults>()(&draw_depth);
    NVNchannelMaskState hidden{}, shown{};
    load<Entry::nvnChannelMaskStateSetDefaults>()(&hidden);
    load<Entry::nvnChannelMaskStateSetChannelMask>()(&hidden, 0, 0, 0, 0, 0);
    load<Entry::nvnChannelMaskStateSetDefaults>()(&shown);
    const float black[]{0, 0, 0, 1}, blue[]{0, 0, 1, 1};
    // An indirect draw of no vertices waits for the GPU (a drain).
    const MTLDrawPrimitivesIndirectArguments no_vertices{0, 1, 0, 0};
    std::memcpy(bytes + 0x800, &no_vertices, sizeof no_vertices);
    // copy_after_condition compares two counters already in memory, which a
    // copy then makes compare the other way: the block keeps the result of
    // the conditional's own point of the stream.
    const auto memory_counter = TestResources::base + 0x780;
    if (which == ConditionalCase::copy_after_condition) {
        NVNcounterData values[2]{{visible ? 5u : 0u, 0}, {0, 0}}, flipped[2]{{visible ? 0u : 7u, 0}, {0, 0}};
        std::memcpy(bytes + 0x780, values, sizeof values);
        std::memcpy(bytes + 0x7c0, flipped, sizeof flipped);
    }
    const auto counter = TestResources::base + 0x700;
    const auto mode = which == ConditionalCase::if_equal ? NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL
                                                          : NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_NOT_EQUAL;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, &depth, nullptr);
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, black, 15);
    load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands, 0.5f, 1, 0, 0);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &hidden);
    load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &query_depth);
    load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED, counter);
    if (which == ConditionalCase::reused_address)
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP, counter);
    if (which == ConditionalCase::stale_after_drain) {
        load<Entry::nvnCommandBufferDrawArraysIndirect>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES,
            TestResources::base + 0x800);
        // The drain applied both effects; these take their indices again.
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP,
            TestResources::base + 0x740);
        load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP, counter);
    }
    load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &shown);
    load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &draw_depth);
    load<Entry::nvnCommandBufferSetRenderEnableConditional>()(&recorder.commands, mode,
        which == ConditionalCase::copy_after_condition ? memory_counter : counter);
    if (which == ConditionalCase::copy_after_condition)
        load<Entry::nvnCommandBufferCopyBufferToBuffer>()(&recorder.commands, TestResources::base + 0x7c0,
                                                          memory_counter, 2 * sizeof(NVNcounterData), 0);
    std::vector<NVNcommandHandle> handles;
    if (which == ConditionalCase::across_submissions) {
        handles.push_back(recorder.finish());
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    }
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    if (which == ConditionalCase::clear_in_block) {
        load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, blue, 15);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    }
    load<Entry::nvnCommandBufferSetRenderEnable>()(&recorder.commands, 1);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &color, nullptr, &whole,
        TestResources::base + 0x2000, 0);
    handles.push_back(recorder.finish());
    const auto waits = backend->statistics().synchronize_gpu_waits;
    std::uint64_t fence = 0;
    for (const auto handle : handles) {
        fence = backend->submit({&handle, 1});
        if (!fence) std::fprintf(stderr, "conditional case: %s\n", backend->last_error().detail.c_str());
        CHECK(fence != 0);
    }
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    const auto drained = backend->statistics().synchronize_gpu_waits - waits;
    bool rendered = visible;
    switch (which) {
        case ConditionalCase::if_equal: rendered = !visible; CHECK(drained == 0); break;
        case ConditionalCase::clear_in_block: CHECK(drained == 1); break;
        // The later timestamp report writes a zero counter. Reusing its
        // destination does not drain; only the conditional's CPU read does.
        case ConditionalCase::reused_address: rendered = false; CHECK(drained == 1); break;
        case ConditionalCase::across_submissions: CHECK(drained == 0); break;
        case ConditionalCase::stale_after_drain: rendered = false; CHECK(drained == 2); break;
        case ConditionalCase::copy_after_condition: CHECK(drained == 0); break;
    }
    const auto* output = reinterpret_cast<const std::uint8_t*>(bytes + 0x2000);
    for (unsigned pixel = 0; pixel != 16; ++pixel) {
        CHECK(output[pixel * 4] == 0 && output[pixel * 4 + 2] == 0);
        CHECK(output[pixel * 4 + 1] == (rendered ? 255 : 0));
    }
    for (const auto handle : handles)
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// Ordering stress for predicates with interleaved or batched queries. Every
// block owns one pixel (scissor) and a pseudo-random
// visibility; a block whose draw read its arguments before the predicate wrote
// them would light an occluded pixel. The first runs take the kernel while the
// in-pass pipeline compiles in the background (both paths in one submission);
// the counted runs then take the path ARMIGO_NVN_PREDICATE_IN_PASS selects.
void test_predicate_ordering_stress(id<MTLDevice> device, id<MTL4CommandQueue> queue, unsigned iterations,
                                    bool batched_queries = false, bool reference_writes = false) {
    constexpr unsigned width = 64, height = 8, blocks = width * height;
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:width height:height mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[181] = [device newTextureWithDescriptor:descriptor];
    descriptor.pixelFormat = MTLPixelFormatDepth32Float;
    resources.textures[182] = [device newTextureWithDescriptor:descriptor];
    auto color_state = texture_state(181, width, height), depth_state = texture_state(182, width, height);
    depth_state.description.format = NVN_FORMAT_DEPTH32F;
    auto color = texture_object(color_state), depth = texture_object(depth_state);
    const NVNtexture* targets[]{&color};
    const NVNcopyRegion whole{0, 0, 0, int(width), int(height), 1};
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    NVNprogram writer{};
    if (reference_writes) {
        CHECK(load<Entry::nvnProgramInitialize>()(&writer, &recorder.device));
        const NVNshaderData write_shaders[]{{6, nullptr}, {6, nullptr}};
        CHECK(load<Entry::nvnProgramSetShaders>()(&writer, 2, write_shaders));
    }
    NVNdepthStencilState pass_query{}, fail_query{}, draw_depth{};
    for (auto* state : {&pass_query, &fail_query}) {
        load<Entry::nvnDepthStencilStateSetDefaults>()(state);
        load<Entry::nvnDepthStencilStateSetDepthTestEnable>()(state, 1);
        load<Entry::nvnDepthStencilStateSetDepthWriteEnable>()(state, 0);
    }
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&pass_query, NVN_DEPTH_FUNC_ALWAYS);
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&fail_query, NVN_DEPTH_FUNC_NEVER);
    load<Entry::nvnDepthStencilStateSetDefaults>()(&draw_depth);
    NVNchannelMaskState hidden{}, shown{};
    load<Entry::nvnChannelMaskStateSetDefaults>()(&hidden);
    load<Entry::nvnChannelMaskStateSetChannelMask>()(&hidden, 0, 0, 0, 0, 0);
    load<Entry::nvnChannelMaskStateSetDefaults>()(&shown);
    const float black[]{0, 0, 0, 1};
    std::uint32_t seed = 0x2545f491u;
    unsigned wrong = 0;
    const auto run = [&] {
        std::memset(bytes, 0, resources.memory.length);
        std::array<bool, blocks> visible{}, rendered{};
        for (unsigned block = 0; block != blocks; ++block) {
            seed = seed * 1664525u + 1013904223u;
            visible[block] = (seed >> 16) & 1;
            rendered[block] = visible[block] != (reference_writes && block % 16 == 1);
        }
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, &depth, nullptr);
        load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, int(width), int(height));
        load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, black, 15);
        load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands, 0.5f, 1, 0, 0);
        load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        const auto query = [&](unsigned block) {
            const auto counter = TestResources::base + 0x4000 + 32u * block;
            load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &hidden);
            load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands,
                visible[block] ? &pass_query : &fail_query);
            load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, int(width), int(height));
            load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
            load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
            load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
                                                         counter);
        };
        const auto conditional = [&](unsigned block) {
            const auto counter = TestResources::base + 0x4000 + 32u * block;
            if (reference_writes && block % 16 == 1) {
                // Change the memory operand in the current render pass, then
                // rebind it away. Its dependency survives that state change.
                load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &hidden);
                load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &draw_depth);
                load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, int(width), int(height));
                load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &writer, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
                load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT,
                    0, counter + 16, 4);
                load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
                load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT,
                    0, TestResources::base + 0x3000, 4);
                load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
            }
            load<Entry::nvnCommandBufferSetRenderEnableConditional>()(&recorder.commands,
                NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_NOT_EQUAL, counter);
            load<Entry::nvnCommandBufferBindChannelMaskState>()(&recorder.commands, &shown);
            load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &draw_depth);
            load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, int(block % width), int(block / width), 1, 1);
            load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
            load<Entry::nvnCommandBufferSetRenderEnable>()(&recorder.commands, 1);
        };
        if (batched_queries) {
            for (unsigned block = 0; block != blocks; ++block) query(block);
            for (unsigned block = 0; block != blocks; ++block) conditional(block);
            load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
                TestResources::base + 0x8000);
        } else {
            for (unsigned block = 0; block != blocks; ++block) { query(block); conditional(block); }
        }
        load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &color, nullptr, &whole,
            TestResources::base + 0xA000, 0);
        const auto handle = recorder.finish();
        const auto fence = backend->submit({&handle, 1});
        CHECK(fence && backend->wait_fence(fence, 30'000'000'000ull));
        CHECK(!backend->last_error());
        const auto* output = reinterpret_cast<const std::uint8_t*>(bytes + 0xA000);
        for (unsigned block = 0; block != blocks; ++block) {
            if (output[block * 4] != 0 || output[block * 4 + 1] != (rendered[block] ? 255 : 0) ||
                output[block * 4 + 2] != 0 || output[block * 4 + 3] != 255) ++wrong;
            NVNcounterData samples{};
            std::memcpy(&samples, bytes + 0x4000 + 32u * block, sizeof samples);
            CHECK(samples.counter == (visible[block] ? blocks : 0));
        }
        if (batched_queries) {
            NVNcounterData cumulative{};
            std::memcpy(&cumulative, bytes + 0x8000, sizeof cumulative);
            const auto expected = (visible.back() ? blocks : 0) +
                std::count(rendered.begin(), rendered.end(), true) +
                (reference_writes ? (blocks / 16) * blocks : 0);
            CHECK(cumulative.counter == static_cast<std::uint64_t>(expected));
        }
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    };
    const char* setting = std::getenv("ARMIGO_NVN_PREDICATE_IN_PASS");
    const bool in_pass = !(setting && *setting == '0');
    unsigned runs = 0;
    for (; in_pass && backend->statistics().predicate_draws == 0 && runs != 200; ++runs) run();
    const auto before = backend->statistics();
    for (unsigned iteration = 0; iteration != iterations; ++iteration) run();
    const auto after = backend->statistics();
    runs += iterations;
    if (wrong) std::fprintf(stderr, "predicate ordering stress: %u of %u blocks wrong\n", wrong, blocks * runs);
    CHECK(wrong == 0);
    CHECK(after.predicate_draws - before.predicate_draws == (in_pass ? blocks * iterations : 0));
    CHECK(after.predicate_kernels - before.predicate_kernels == (in_pass ? 0 : blocks * iterations));
    CHECK(after.synchronize_gpu_waits == before.synchronize_gpu_waits);
    const char* batching_setting = std::getenv("ARMIGO_NVN_BATCH_FINALIZED_QUERIES");
    const bool batching = !(batching_setting && *batching_setting == '0') && in_pass && batched_queries;
    const auto passes_per_run = batching ? 2u + (reference_writes ? blocks / 16 : 0u) : blocks + 1;
    CHECK(after.render_passes - before.render_passes == passes_per_run * iterations);
    if (reference_writes) load<Entry::nvnProgramFinalize>()(&writer);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A depth-stencil clear of a depth-only target clears the depth and ignores the
// stencil part, as the GPU does, instead of failing the clear.
void test_stencil_clear_without_stencil(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[121] = [device newTextureWithDescriptor:descriptor];
    auto depth_state = texture_state(121, 4, 4);
    depth_state.description.format = NVN_FORMAT_DEPTH32F;
    auto depth = texture_object(depth_state);
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 0, nullptr, nullptr, &depth, nullptr);
    load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands, 0.5f, 1, 7, 0xff);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    std::array<float, 16> values{};
    [resources.textures[121] getBytes:values.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (const auto value : values) CHECK(value == 0.5f);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_segmented_failed_prefix(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    const char* old_cut = std::getenv("ARMIGO_NVN_CUT");
    const std::string saved_cut = old_cut ? old_cut : "";
    setenv("ARMIGO_NVN_CUT", "16", 1);
    auto backend = replay::create_replay_backend(device, queue, resources);
    if (old_cut) setenv("ARMIGO_NVN_CUT", saved_cut.c_str(), 1);
    else unsetenv("ARMIGO_NVN_CUT");
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget;
    resources.textures[103] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(103, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{1, nullptr}, {1, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    id<MTLSharedEvent> gate = [device newSharedEvent];
    [queue waitForEvent:gate value:1];
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    for (unsigned i = 0; i < 16; ++i)
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferBarrier>()(&recorder.commands, 0);
    // An unchanged target with an already closed encoder is a legal cut point.
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, -1, 4);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence != 0);
    CHECK(!backend->wait_fence(fence, 10'000'000));
    CHECK(backend->completed_fence() == 0);
    gate.signaledValue = 1;
    CHECK(backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(backend->last_error().code == replay::ReplayErrorCode::invalid_command);
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[103] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned i = 0; i < 16; ++i) CHECK(pixels[i * 4 + 1] == 255);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    // An encoding error after a committed prefix leaves no hole in retirement.
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferClearBuffer>()(&recorder.commands, TestResources::base, 4, 17);
    const auto next = recorder.finish();
    const auto next_fence = backend->submit({&next, 1});
    CHECK(next_fence == fence + 1 && backend->wait_fence(next_fence, 10'000'000'000ull));
    CHECK(*static_cast<const std::uint32_t*>(resources.memory.contents) == 17);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, next);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// Updates issued before binding still publish to a later constant window in
// command order. Rewriting an outer range supersedes every older inner write.
void test_overlapping_uniform_update_order(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    std::memset(resources.memory.contents, 0, resources.memory.length);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget;
    resources.textures[91] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(91, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{5, nullptr}, {5, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    constexpr auto uniform = TestResources::base + 0x200;
    const std::array<float, 4> red{1, 0, 0, 1}, blue{0, 0, 1, 1};
    const float green = 1;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 0, 16, red.data());
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 4, 4, &green);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 0, 16, blue.data());
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, uniform, 128);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[91] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    CHECK(std::memcmp(static_cast<const std::byte*>(resources.memory.contents) + 0x200,
                      blue.data(), sizeof blue) == 0);
    for (unsigned i = 0; i < 16; ++i)
        CHECK(pixels[i * 4] == 0 && pixels[i * 4 + 1] == 0 &&
              pixels[i * 4 + 2] == 255 && pixels[i * 4 + 3] == 255);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_single_writer_mrt_alias(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    // Every draw exports slot zero and exactly one material slot. No case
    // depends on the order of two writers to the same subresource.
    for (bool srgb : {false, true}) for (unsigned writer : {1u, 2u})
        for (unsigned layout = 0; layout < 7; ++layout) {
            TestResources resources(device);
            auto backend = replay::create_replay_backend(device, queue, resources);
            auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                width:4 height:4 mipmapped:NO];
            descriptor.storageMode = MTLStorageModeShared;
            descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
            id<MTLTexture> separate = [device newTextureWithDescriptor:descriptor];
            const auto format = srgb ? MTLPixelFormatRGBA8Unorm_sRGB : MTLPixelFormatRGBA8Unorm;
            descriptor.pixelFormat = format;
            descriptor.textureType = MTLTextureType2DArray;
            descriptor.arrayLength = 2;
            id<MTLTexture> root = [device newTextureWithDescriptor:descriptor];
            id<MTLTexture> other = [device newTextureWithDescriptor:descriptor];
            const auto view = [&](id<MTLTexture> storage, NSUInteger slice) {
                return [storage newTextureViewWithPixelFormat:format textureType:MTLTextureType2D
                    levels:NSMakeRange(0, 1) slices:NSMakeRange(slice, 1)];
            };
            id<MTLTexture> first = view(root, 0);
            id<MTLTexture> second = layout == 0 ? view(other, 0)
                : layout == 1 ? first : view(root, layout == 3 ? 1 : 0);
            CHECK(separate && first && second);
            if (!separate || !first || !second) continue;
            std::array<std::uint8_t, 64> initial{};
            for (unsigned pixel = 0; pixel < 16; ++pixel) {
                initial[pixel * 4] = 255;
                initial[pixel * 4 + 3] = 255;
            }
            for (id<MTLTexture> texture : {separate, first, second})
                [texture replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0
                    withBytes:initial.data() bytesPerRow:16];
            std::array<TextureState, 3> states{texture_state(93, 4, 4),
                texture_state(94, 4, 4), texture_state(95, 4, 4)};
            states[1].description.format = states[2].description.format =
                srgb ? NVN_FORMAT_RGBA8_SRGB : NVN_FORMAT_RGBA8;
            resources.textures[93] = separate;
            resources.textures[94] = first;
            resources.textures[95] = second;
            std::array<NVNtexture, 3> objects{texture_object(states[0]),
                texture_object(states[1]), texture_object(states[2])};
            const NVNtexture* targets[]{&objects[0], &objects[1], &objects[2]};
            Recorder recorder;
            NVNprogram program{}, next_program{};
            CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
            const NVNshaderData shaders[]{{8 + writer, nullptr}, {8 + writer, nullptr}};
            CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
            if (layout == 4) {
                CHECK(load<Entry::nvnProgramInitialize>()(&next_program, &recorder.device));
                const NVNshaderData next_shaders[]{{11 - writer, nullptr}, {11 - writer, nullptr}};
                CHECK(load<Entry::nvnProgramSetShaders>()(&next_program, 2, next_shaders));
            }
            load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
            load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 3, targets, nullptr, nullptr, nullptr);
            load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
            const std::array<float, 4> yellow{1, 1, 0, 1}, magenta{1, 0, 1, 1};
            if (layout == 5) {
                // Ordered clears through sibling views must survive detaching
                // either view before the next partial-screen draw.
                load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 1, yellow.data(), 15);
                load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 2, magenta.data(), 15);
            }
            if (layout == 4 || layout == 5)
                load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 2, 4);
            load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
            if (layout == 4) {
                load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &next_program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
                load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 2, 0, 2, 4);
                load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
            } else if (layout == 6) {
                // A clear after a draw supersedes that draw before the next
                // pass loads the same material through its retained view.
                load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 2, magenta.data(), 15);
                load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 2, 4);
                load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
            }
            const auto handle = recorder.finish();
            const auto fence = backend->submit({&handle, 1});
            CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
            CHECK(!backend->last_error());
            CHECK(backend->statistics().render_passes == (layout == 4 || layout == 6 ? 2 : 1));
            const bool aliases = layout != 0 && layout != 3;
            unsigned slot = 0;
            for (id<MTLTexture> texture : {separate, first, second}) {
                std::array<std::uint8_t, 64> pixels{};
                [texture getBytes:pixels.data() bytesPerRow:16
                    fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
                const std::array<std::uint8_t, 4> expected = slot == 0
                    ? std::array<std::uint8_t, 4>{0, 0, 255, 255}
                    : aliases || slot == writer ? std::array<std::uint8_t, 4>{0, 255, 0, 255}
                    : std::array<std::uint8_t, 4>{255, 0, 0, 255};
                unsigned mismatches = 0;
                for (unsigned pixel = 0; pixel < 16; ++pixel) {
                    auto wanted = expected;
                    if (pixel % 4 >= 2 && (layout == 5 || layout == 6)) {
                        if (slot) wanted = {255, 0, 255, 255};
                        else if (layout == 5) wanted = {255, 0, 0, 255};
                    }
                    mismatches += std::memcmp(pixels.data() + pixel * 4, wanted.data(), 4) != 0;
                }
                if (mismatches)
                    std::fprintf(stderr, "MRT preservation failed: srgb=%u writer=%u layout=%u slot=%u mismatches=%u\n",
                        unsigned(srgb), writer, layout, slot, mismatches);
                CHECK(mismatches == 0);
                ++slot;
            }
            load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
            load<Entry::nvnProgramFinalize>()(&program);
            if (layout == 4) load<Entry::nvnProgramFinalize>()(&next_program);
            load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
        }
}

void test_aliased_mrt_writer_choice(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    // Aliased slots with different formats are one subresource, and a draw
    // writing several slots of a group keeps the lowest written one.
    for (unsigned scenario = 0; scenario < 3; ++scenario) {
        TestResources resources(device);
        auto backend = replay::create_replay_backend(device, queue, resources);
        auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
            width:4 height:4 mipmapped:NO];
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
        id<MTLTexture> separate = [device newTextureWithDescriptor:descriptor];
        id<MTLTexture> root = [device newTextureWithDescriptor:descriptor];
        id<MTLTexture> first = [root newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
        id<MTLTexture> second = scenario == 2 ? first
            : [root newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm_sRGB];
        CHECK(separate && first && second);
        if (!separate || !first || !second) continue;
        std::array<std::uint8_t, 64> initial{};
        for (unsigned pixel = 0; pixel < 16; ++pixel) {
            initial[pixel * 4] = 255;
            initial[pixel * 4 + 3] = 255;
        }
        for (id<MTLTexture> texture : {separate, root})
            [texture replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0
                withBytes:initial.data() bytesPerRow:16];
        std::array<TextureState, 3> states{texture_state(96, 4, 4),
            texture_state(97, 4, 4), texture_state(98, 4, 4)};
        if (scenario != 2) states[2].description.format = NVN_FORMAT_RGBA8_SRGB;
        resources.textures[96] = separate;
        resources.textures[97] = first;
        resources.textures[98] = second;
        std::array<NVNtexture, 3> objects{texture_object(states[0]),
            texture_object(states[1]), texture_object(states[2])};
        const NVNtexture* targets[]{&objects[0], &objects[1], &objects[2]};
        Recorder recorder;
        NVNprogram program{};
        CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
        // Scenario 0 writes only slot 2, scenarios 1 and 2 write both slots.
        const std::uintptr_t shader = scenario == 0 ? 10 : 11;
        const NVNshaderData shaders[]{{shader, nullptr}, {shader, nullptr}};
        CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 3, targets, nullptr, nullptr, nullptr);
        load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        const auto handle = recorder.finish();
        const auto fence = backend->submit({&handle, 1});
        CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
        CHECK(!backend->last_error());
        CHECK(backend->statistics().render_passes == 1);
        unsigned slot = 0;
        for (id<MTLTexture> texture : {separate, root}) {
            std::array<std::uint8_t, 64> pixels{};
            [texture getBytes:pixels.data() bytesPerRow:16
                fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
            // The slot-2 writer exports green there; the two-slot writer
            // exports green to slot 1 and magenta to slot 2, and slot 1 stays.
            const std::array<std::uint8_t, 4> expected = slot == 0
                ? std::array<std::uint8_t, 4>{0, 0, 255, 255}
                : std::array<std::uint8_t, 4>{0, 255, 0, 255};
            unsigned mismatches = 0;
            for (unsigned pixel = 0; pixel < 16; ++pixel)
                mismatches += std::memcmp(pixels.data() + pixel * 4, expected.data(), 4) != 0;
            if (mismatches)
                std::fprintf(stderr, "MRT writer choice failed: scenario=%u slot=%u mismatches=%u\n",
                    scenario, slot, mismatches);
            CHECK(mismatches == 0);
            ++slot;
        }
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
        load<Entry::nvnProgramFinalize>()(&program);
        load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
    }
}

void test_uniform_prefix_growth(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    std::memset(resources.memory.contents, 0, resources.memory.length);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget;
    resources.textures[92] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(92, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram near_program{}, far_program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&near_program, &recorder.device));
    CHECK(load<Entry::nvnProgramInitialize>()(&far_program, &recorder.device));
    const NVNshaderData near_shaders[]{{5, nullptr}, {5, nullptr}};
    const NVNshaderData far_shaders[]{{8, nullptr}, {8, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&near_program, 2, near_shaders));
    CHECK(load<Entry::nvnProgramSetShaders>()(&far_program, 2, far_shaders));
    constexpr auto uniform = TestResources::base + 0x200;
    const std::array<float, 4> green{0, 1, 0, 1}, blue{0, 0, 1, 1};
    std::memcpy(static_cast<std::byte*>(resources.memory.contents) + 0x200,
                green.data(), sizeof green);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &near_program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, uniform, 128);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    // Create a version during the active pass, then materialize only its first
    // 16 bytes. The recorded far write is outside that readable prefix.
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 64, 16, blue.data());
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    // Grow the shader's window to 80 bytes without recording another update.
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &far_program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 2, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    const auto stats = backend->statistics();
    CHECK(stats.uniform_versions == 1 && stats.uniform_version_bytes == 0);
    CHECK(stats.render_passes == 1 && stats.synchronize_gpu_waits == 0);
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[92] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 4; ++x) {
            const auto offset = (y * 4 + x) * 4;
            CHECK(pixels[offset] == 0 && pixels[offset + 1] == (x < 2 ? 255 : 0) &&
                  pixels[offset + 2] == (x < 2 ? 0 : 255) && pixels[offset + 3] == 255);
        }
    CHECK(std::memcmp(static_cast<const std::byte*>(resources.memory.contents) + 0x240,
                      blue.data(), sizeof blue) == 0);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&near_program);
    load<Entry::nvnProgramFinalize>()(&far_program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A constant window reads zero past the bound size: a buffer smaller than the
// window binds a copy whose tail is zero, one that covers it binds in place.
// The synthetic shader reads the window's fifth vector without any check.
void test_constant_window_bound(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    std::memset(resources.memory.contents, 0, resources.memory.length);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget;
    resources.textures[92] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(92, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram far_program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&far_program, &recorder.device));
    const NVNshaderData far_shaders[]{{8, nullptr}, {8, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&far_program, 2, far_shaders));
    constexpr auto uniform = TestResources::base + 0x200;
    const std::array<float, 4> blue{0, 0, 1, 1};
    // The window's fifth vector lies past a 64-byte bound and inside a 128-byte one.
    std::memcpy(static_cast<std::byte*>(resources.memory.contents) + 0x240, blue.data(), sizeof blue);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &far_program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, uniform, 64);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, uniform, 128);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 2, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[92] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 4; ++x) {
            const auto offset = (y * 4 + x) * 4;
            const bool inside = x >= 2;
            CHECK(pixels[offset] == 0 && pixels[offset + 1] == 0 &&
                  pixels[offset + 2] == (inside ? 255 : 0) && pixels[offset + 3] == (inside ? 255 : 0));
        }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&far_program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A window copied for one draw is copied again after a GPU write to its
// range: a buffer smaller than the window is snapshotted, and the second draw
// must read what the copy wrote, not the first snapshot.
void test_constant_window_after_gpu_write(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    std::memset(resources.memory.contents, 0, resources.memory.length);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget;
    resources.textures[92] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(92, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{5, nullptr}, {5, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    constexpr auto uniform = TestResources::base + 0x200;
    constexpr auto source = TestResources::base + 0x300;
    // An 8-byte binding of a 16-byte window: red and green come from the
    // guest, blue and alpha read zero past the bound.
    const std::array<float, 2> red{1, 0}, green{0, 1};
    std::memcpy(static_cast<std::byte*>(resources.memory.contents) + 0x200, red.data(), sizeof red);
    std::memcpy(static_cast<std::byte*>(resources.memory.contents) + 0x300, green.data(), sizeof green);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, uniform, 8);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyBufferToBuffer>()(&recorder.commands, source, uniform, sizeof green, 0);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 2, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[92] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 4; ++x) {
            const auto offset = (y * 4 + x) * 4;
            const bool after = x >= 2;
            CHECK(pixels[offset] == (after ? 0 : 255) && pixels[offset + 1] == (after ? 255 : 0) &&
                  pixels[offset + 2] == 0 && pixels[offset + 3] == 0);
        }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_uniform_versions_and_gpu_indices(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[81] = [device newTextureWithDescriptor:descriptor];
    resources.textures[82] = [device newTextureWithDescriptor:descriptor];
    auto next_state = texture_state(82, 4, 4);
    auto next_target = texture_object(next_state);
    const NVNtexture* next_targets[]{&next_target};
    auto state = texture_state(81, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{5, nullptr}, {5, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    constexpr auto uniform = TestResources::base + 0x100;
    const std::array<float, 4> red{1, 0, 0, 1}, green{0, 1, 0, 1};
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    std::memcpy(resources.memory.contents, indices.data(), sizeof indices);
    // Leave the first update on a blocked GPU. The next submission must build
    // its uniform version from the captured update, not stale guest memory.
    id<MTLSharedEvent> gate = [device newSharedEvent];
    [queue waitForEvent:gate value:1];
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 0, 16, red.data());
    const auto preload = recorder.finish();
    CHECK(backend->submit({&preload, 1}) != 0);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, preload);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferCopyBufferToBuffer>()(&recorder.commands,
        TestResources::base, TestResources::base + 0x80, sizeof indices, 0);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands,
        NVN_SHADER_STAGE_FRAGMENT, 0, uniform, 128);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 64, 16, red.data());
    load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawElements>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES,
        NVN_INDEX_TYPE_UNSIGNED_SHORT, 3, TestResources::base + 0x80);
    // Updating beyond the prefix already read does not replace the version.
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 96, 16, red.data());
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 0, 16, green.data());
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 2, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawElements>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES,
        NVN_INDEX_TYPE_UNSIGNED_SHORT, 3, TestResources::base + 0x80);
    load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
        TestResources::base + 0x3000);
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands,
        &target, nullptr, &whole, TestResources::base + 0x1000, 0);
    const auto handle = recorder.finish();
    CHECK(backend->submit({&handle, 1}) != 0);
    // A third submission keeps a second visibility buffer live and continues
    // the counter. It also reads the second submission's latest uniform value.
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, next_targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 1, 4);
    load<Entry::nvnCommandBufferDrawElements>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES,
        NVN_INDEX_TYPE_UNSIGNED_SHORT, 3, TestResources::base + 0x80);
    load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
        TestResources::base + 0x3020);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands,
        &next_target, nullptr, &whole, TestResources::base + 0x2000, 0);
    const auto continuation = recorder.finish();
    const auto fence = backend->submit({&continuation, 1});
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (backend->statistics().submissions < 3 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(backend->statistics().submissions == 3);
    CHECK(backend->completed_fence() == 0);
    gate.signaledValue = 1;
    CHECK(backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    const auto stats = backend->statistics();
    CHECK(stats.skipped == 0 && stats.render_passes == 2 && stats.synchronize_gpu_waits == 0);
    CHECK(stats.uniform_versions == 2 && stats.uniform_version_bytes == 16);
    const auto* bytes = static_cast<const std::uint8_t*>(resources.memory.contents);
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 4; ++x) {
            const auto offset = 0x1000 + (4 * y + x) * 4;
            CHECK(bytes[offset] == (x < 2 ? 255 : 0));
            CHECK(bytes[offset + 1] == (x < 2 ? 0 : 255));
        }
    for (unsigned y = 0; y < 4; ++y) {
        CHECK(bytes[0x2000 + 4 * y * 4] == 0);
        CHECK(bytes[0x2000 + 4 * y * 4 + 1] == 255);
    }
    NVNcounterData first_samples{}, second_samples{};
    std::memcpy(&first_samples, bytes + 0x3000, sizeof first_samples);
    std::memcpy(&second_samples, bytes + 0x3020, sizeof second_samples);
    CHECK(first_samples.counter == 16 && second_samples.counter == 20);
    CHECK(std::memcmp(bytes + 0x100, green.data(), 16) == 0);
    CHECK(std::memcmp(bytes + 0x160, red.data(), 16) == 0);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, continuation);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A counter report into a versioned uniform range makes the draw's fill of
// that version wait for the GPU, which writes the version back and ends it.
// The draw then reads the guest buffer, without touching the ended version.
void test_uniform_fill_after_gpu_wait(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[83] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(83, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{8, nullptr}, {8, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    constexpr auto uniform = TestResources::base + 0x100;
    const std::array<float, 4> red{1, 0, 0, 1};
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    std::memcpy(static_cast<std::byte*>(resources.memory.contents) + 0x80, indices.data(), sizeof indices);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands,
        NVN_SHADER_STAGE_FRAGMENT, 0, uniform, 128);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, uniform, 128, 64, 16, red.data());
    load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
    load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
        uniform);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 4, 4);
    load<Entry::nvnCommandBufferDrawElements>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES,
        NVN_INDEX_TYPE_UNSIGNED_SHORT, 3, TestResources::base + 0x80);
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands,
        &target, nullptr, &whole, TestResources::base + 0x1000, 0);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence != 0);
    CHECK(backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    const auto stats = backend->statistics();
    CHECK(stats.skipped == 0 && stats.synchronize_gpu_waits == 1);
    const auto* bytes = static_cast<const std::uint8_t*>(resources.memory.contents);
    CHECK(std::memcmp(bytes + 0x140, red.data(), 16) == 0);
    CHECK(bytes[0x1000] == 255 && bytes[0x1001] == 0);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_upload_retirement(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    Recorder recorder;
    std::array<NVNcommandHandle, 7> handles{};
    std::uint64_t fence = 0;
    for (unsigned submission = 0; submission < handles.size(); ++submission) {
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        for (unsigned update = 0; update < 32; ++update) {
            const std::array<std::uint32_t, 4> values{
                submission, update, submission * 32 + update, 0x12345678};
            load<Entry::nvnCommandBufferUpdateUniformBuffer>()(
                &recorder.commands, TestResources::base + update * sizeof values,
                sizeof values, 0, sizeof values, values.data());
        }
        handles[submission] = recorder.finish();
        fence = backend->submit({&handles[submission], 1});
        CHECK(fence != 0);
    }
    CHECK(backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    const auto* words = static_cast<const std::uint32_t*>(resources.memory.contents);
    for (unsigned update = 0; update < 32; ++update) {
        CHECK(words[update * 4] == handles.size() - 1);
        CHECK(words[update * 4 + 1] == update);
        CHECK(words[update * 4 + 2] == (handles.size() - 1) * 32 + update);
        CHECK(words[update * 4 + 3] == 0x12345678);
    }
    const auto stats = backend->statistics();
    CHECK(stats.upload_arena_allocations >= 1 && stats.upload_arena_allocations <= 6);
    CHECK(stats.upload_arena_allocations + stats.upload_arena_reuses == handles.size());
    CHECK(stats.upload_bytes == handles.size() * 32 * 16);
    CHECK(stats.synchronize_gpu_waits == 0);
    CHECK(stats.compute_passes == handles.size());
    for (const auto handle : handles)
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A failed upload batch fails the segment whose fence covers it, and only that
// one: the segments before and after retire as successes.
void test_failed_upload_attribution(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    Recorder recorder;
    std::array<NVNcommandHandle, 3> handles{};
    resources.failing_upload = 2;
    for (unsigned submission = 0; submission < handles.size(); ++submission) {
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        const std::array<std::uint32_t, 4> values{submission, 1, 2, 3};
        load<Entry::nvnCommandBufferUpdateUniformBuffer>()(
            &recorder.commands, TestResources::base, sizeof values, 0, sizeof values, values.data());
        handles[submission] = recorder.finish();
        const auto fence = backend->submit({&handles[submission], 1});
        CHECK(fence != 0);
        CHECK(backend->wait_fence(fence, 10'000'000'000ull));
        const auto error = backend->last_error();
        if (submission == 1) {
            CHECK(error.code == replay::ReplayErrorCode::execution);
            CHECK(error.detail.find("texture upload failed") != std::string::npos);
        } else if (submission == 0) {
            CHECK(!error);
        }
    }
    for (const auto handle : handles)
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_cube_clear(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                           size:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    auto cube = [device newTextureWithDescriptor:descriptor];
    CHECK(cube != nil);
    auto state = texture_state(90, 4, 4);
    state.description.target = NVN_TEXTURE_TARGET_CUBEMAP;
    state.description.depth = 6;
    resources.textures[90] = cube;
    auto object = texture_object(state);
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 6};
    const float blue[]{0, 0, 1, 1};
    load<Entry::nvnCommandBufferClearTexture>()(&recorder.commands, &object, nullptr, &whole,
                                              blue, 0xf);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &object, nullptr,
                                                      &whole, TestResources::base, NVN_COPY_FLAGS_NONE);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    const auto* bytes = static_cast<const std::uint8_t*>(resources.memory.contents);
    for (unsigned pixel = 0; pixel < 6 * 4 * 4; ++pixel) {
        CHECK(bytes[4 * pixel] == 0 && bytes[4 * pixel + 1] == 0);
        CHECK(bytes[4 * pixel + 2] == 255 && bytes[4 * pixel + 3] == 255);
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A BGRA8 texture keeps the B, G, R, A byte order: a blue clear copied to a
// buffer reads 255, 0, 0, 255.
void test_bgra8_clear_copy(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[91] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(91, 4, 4);
    state.description.format = NVN_FORMAT_BGRA8;
    auto object = texture_object(state);
    Recorder recorder;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
    const float blue[]{0, 0, 1, 1};
    load<Entry::nvnCommandBufferClearTexture>()(&recorder.commands, &object, nullptr, &whole, blue, 0xf);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &object, nullptr,
                                                      &whole, TestResources::base, NVN_COPY_FLAGS_NONE);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error() && backend->statistics().skipped == 0);
    const auto* bytes = static_cast<const std::uint8_t*>(resources.memory.contents);
    for (unsigned pixel = 0; pixel < 4 * 4; ++pixel) {
        CHECK(bytes[4 * pixel] == 255 && bytes[4 * pixel + 1] == 0);
        CHECK(bytes[4 * pixel + 2] == 0 && bytes[4 * pixel + 3] == 255);
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// Dispatches in one transfer encoder go without a barrier only when their
// guest accesses are independent: a dispatch reading another's output still
// sees it.
void test_dispatch_hazards(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    NVNprogram writer{};
    NVNprogram chain{};
    Recorder recorder;
    CHECK(load<Entry::nvnProgramInitialize>()(&writer, &recorder.device) == 1);
    CHECK(load<Entry::nvnProgramInitialize>()(&chain, &recorder.device) == 1);
    const std::array<NVNshaderData, 1> writer_shaders{{{2, nullptr}}};
    const std::array<NVNshaderData, 1> chain_shaders{{{20, nullptr}}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&writer, 1, writer_shaders.data()) == 1);
    CHECK(load<Entry::nvnProgramSetShaders>()(&chain, 1, chain_shaders.data()) == 1);
    constexpr std::uint64_t first = TestResources::base + 0x1000;
    constexpr std::uint64_t second = TestResources::base + 0x2000;
    constexpr std::uint64_t chained = TestResources::base + 0x3000;
    auto* bytes = static_cast<std::uint8_t*>(resources.memory.contents);
    std::memset(bytes + 0x1000, 0, 0x3000);

    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &writer, NVN_SHADER_STAGE_COMPUTE_BIT);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0, first, 16);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    // Independent of the first: another range.
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0, second, 16);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    // Dependent: reads the first dispatch's output.
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &chain, NVN_SHADER_STAGE_COMPUTE_BIT);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0, first, 16);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 1, chained, 16);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence != 0);
    CHECK(backend->wait_fence(fence, 10'000'000'000ull));
    std::array<std::uint32_t, 4> words{};
    std::memcpy(words.data(), bytes + 0x2000, sizeof words);
    CHECK((words == std::array<std::uint32_t, 4>{10, 11, 12, 13}));
    std::memcpy(words.data(), bytes + 0x3000, sizeof words);
    CHECK((words == std::array<std::uint32_t, 4>{110, 111, 112, 113}));
    const auto stats = backend->statistics();
    CHECK(stats.dispatches == 3);
    CHECK(stats.transfer_barriers_skipped == 1);

    // An undeclared operation (a copy, the kind that also writes textures)
    // takes the barrier and leaves the encoder untracked, so the next
    // dispatch takes one too although its accesses are independent.
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &writer, NVN_SHADER_STAGE_COMPUTE_BIT);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0, first, 16);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    load<Entry::nvnCommandBufferCopyBufferToBuffer>()(&recorder.commands,
        TestResources::base + 0x4000, TestResources::base + 0x4100, 16, 0);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0, second, 16);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    const auto untracked = recorder.finish();
    const auto untracked_fence = backend->submit({&untracked, 1});
    CHECK(untracked_fence != 0);
    CHECK(backend->wait_fence(untracked_fence, 10'000'000'000ull));
    CHECK(backend->statistics().dispatches == 5);
    CHECK(backend->statistics().transfer_barriers_skipped == 1);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, untracked);

    // Write after read, reads of one range, and ranges that only touch:
    // reader of first -> second reader of first (skip) -> writer of the
    // range after first (touching, skip) -> writer of first (conflicts with
    // both readers: barrier).
    constexpr std::uint64_t other = TestResources::base + 0x3100;
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &chain, NVN_SHADER_STAGE_COMPUTE_BIT);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0, first, 16);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 1, chained, 16);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 1, other, 16);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &writer, NVN_SHADER_STAGE_COMPUTE_BIT);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0, first + 16, 16);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(&recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0, first, 16);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    const auto ordered = recorder.finish();
    const auto ordered_fence = backend->submit({&ordered, 1});
    CHECK(ordered_fence != 0);
    CHECK(backend->wait_fence(ordered_fence, 10'000'000'000ull));
    CHECK(backend->statistics().dispatches == 9);
    CHECK(backend->statistics().transfer_barriers_skipped == 3);
    std::memcpy(words.data(), bytes + 0x3100, sizeof words);
    CHECK((words == std::array<std::uint32_t, 4>{110, 111, 112, 113}));
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, ordered);

    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&writer);
    load<Entry::nvnProgramFinalize>()(&chain);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_clear_draw_copy_dispatch(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;

    auto* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
                       MTLTextureUsageShaderWrite;
    id<MTLTexture> first = [device newTextureWithDescriptor:descriptor];
    id<MTLTexture> second = [device newTextureWithDescriptor:descriptor];
    CHECK(first != nil && second != nil);
    TextureState first_state = texture_state(1, 4, 4);
    TextureState second_state = texture_state(2, 4, 4);
    resources.textures.emplace(1, first);
    resources.textures.emplace(2, second);
    resources.texture_ids.emplace(1, first);
    resources.texture_ids.emplace(2, second);
    NVNtexture first_object = texture_object(first_state);
    NVNtexture second_object = texture_object(second_state);

    NVNprogram graphics_program{};
    NVNprogram compute_program{};
    Recorder recorder;
    CHECK(load<Entry::nvnProgramInitialize>()(&graphics_program, &recorder.device) == 1);
    CHECK(load<Entry::nvnProgramInitialize>()(&compute_program, &recorder.device) == 1);
    const std::array<NVNshaderData, 2> graphics_descriptors{{{1, nullptr}, {1, nullptr}}};
    const std::array<NVNshaderData, 1> compute_descriptors{{{2, nullptr}}};
    CHECK(load<Entry::nvnProgramSetShaders>()(
        &graphics_program, graphics_descriptors.size(), graphics_descriptors.data()) == 1);
    CHECK(load<Entry::nvnProgramSetShaders>()(
        &compute_program, compute_descriptors.size(), compute_descriptors.data()) == 1);

    constexpr std::uint64_t upload = TestResources::base + 0x0000;
    constexpr std::uint64_t copied = TestResources::base + 0x1000;
    constexpr std::uint64_t readback = TestResources::base + 0x2000;
    constexpr std::uint64_t dispatch_output = TestResources::base + 0x3000;
    constexpr std::uint64_t counter = TestResources::base + 0x4000;
    auto* bytes = static_cast<std::uint8_t*>(resources.memory.contents);
    for (unsigned index = 0; index != 64; ++index) bytes[index] = std::uint8_t(index * 3);
    std::memset(bytes + 0x1000, 0, 0x4000);

    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferCopyBufferToBuffer>()(
        &recorder.commands, upload, copied, 64, NVN_COPY_FLAGS_NONE);
    load<Entry::nvnCommandBufferSetCopyRowStride>()(&recorder.commands, 16);
    load<Entry::nvnCommandBufferSetCopyImageStride>()(&recorder.commands, 64);
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
    load<Entry::nvnCommandBufferCopyBufferToTexture>()(
        &recorder.commands, copied, &first_object, nullptr, &whole, NVN_COPY_FLAGS_NONE);
    load<Entry::nvnCommandBufferCopyTextureToTexture>()(
        &recorder.commands, &first_object, nullptr, &whole,
        &second_object, nullptr, &whole, NVN_COPY_FLAGS_NONE);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(
        &recorder.commands, &second_object, nullptr, &whole, readback, NVN_COPY_FLAGS_NONE);
    const auto copy_handle = recorder.finish();
    const auto copy_fence = backend->submit({&copy_handle, 1});
    CHECK(copy_fence != 0);
    CHECK(backend->wait_fence(copy_fence, 10'000'000'000ull));
    CHECK(std::memcmp(bytes, bytes + 0x2000, 64) == 0);

    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    const NVNtexture* targets[]{&first_object};
    load<Entry::nvnCommandBufferSetRenderTargets>()(
        &recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 4, 4);
    const std::array<float, 4> clear{0.25f, 0.5f, 0.75f, 1.0f};
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, clear.data(), 0xf);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(
        &recorder.commands, &first_object, nullptr, &whole, readback, NVN_COPY_FLAGS_NONE);
    const auto clear_handle = recorder.finish();
    const auto clear_fence = backend->submit({&clear_handle, 1});
    CHECK(clear_fence != 0);
    CHECK(backend->wait_fence(clear_fence, 10'000'000'000ull));
    CHECK(bytes[0x2000] >= 63 && bytes[0x2000] <= 64);
    CHECK(bytes[0x2001] >= 127 && bytes[0x2001] <= 128);
    CHECK(bytes[0x2002] >= 191 && bytes[0x2002] <= 192);
    CHECK(bytes[0x2003] == 255);

    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(
        &recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(
        &recorder.commands, &graphics_program,
        NVN_SHADER_STAGE_VERTEX_BIT | NVN_SHADER_STAGE_FRAGMENT_BIT);
    load<Entry::nvnCommandBufferDrawArrays>()(
        &recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(
        &recorder.commands, &first_object, nullptr, &whole, readback, NVN_COPY_FLAGS_NONE);
    const auto draw_handle = recorder.finish();
    const auto draw_fence = backend->submit({&draw_handle, 1});
    CHECK(draw_fence != 0);
    CHECK(backend->wait_fence(draw_fence, 10'000'000'000ull));
    CHECK(bytes[0x2000] == 0 && bytes[0x2001] == 255 &&
          bytes[0x2002] == 0 && bytes[0x2003] == 255);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, draw_handle);

    // State and identical target updates stay inside one pass. The untouched
    // columns retain the clear carried into that pass's attachment load.
    const auto before_continuation = backend->statistics();
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    const std::array<float, 4> background{0.0f, 0.0f, 1.0f, 1.0f};
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, background.data(), 0xf);
    for (int i = 0; i < 12; ++i) {
        load<Entry::nvnCommandBufferSetRenderTargets>()(
            &recorder.commands, 1, targets, nullptr, nullptr, nullptr);
        load<Entry::nvnCommandBufferBindProgram>()(
            &recorder.commands, &graphics_program,
            NVN_SHADER_STAGE_VERTEX_BIT | NVN_SHADER_STAGE_FRAGMENT_BIT);
        load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, i % 2, 0, 1, 4);
        load<Entry::nvnCommandBufferDrawArrays>()(
            &recorder.commands, i % 2 ? NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP : NVN_DRAW_PRIMITIVE_TRIANGLES,
            0, 3);
    }
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(
        &recorder.commands, &first_object, nullptr, &whole, readback, NVN_COPY_FLAGS_NONE);
    const auto continuation_handle = recorder.finish();
    const auto continuation_fence = backend->submit({&continuation_handle, 1});
    CHECK(backend->wait_fence(continuation_fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    CHECK(backend->statistics().render_passes - before_continuation.render_passes == 1);
    CHECK(backend->statistics().draws - before_continuation.draws == 12);
    for (unsigned y = 0; y < 4; ++y) {
        for (unsigned x = 0; x < 4; ++x) {
            const auto at = 0x2000 + (y * 4 + x) * 4;
            CHECK(bytes[at] == 0 && bytes[at + 1] == (x < 2 ? 255 : 0) &&
                  bytes[at + 2] == (x < 2 ? 0 : 255) && bytes[at + 3] == 255);
        }
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, continuation_handle);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 4, 4);
    const auto restore_handle = recorder.finish();
    CHECK(backend->wait_fence(backend->submit({&restore_handle, 1}), 10'000'000'000ull));
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, restore_handle);

    // Queue state owns inherited attachments after the list that supplied
    // their immutable snapshots is explicitly finalized.
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    const std::array<float, 4> inherited_clear{0.0f, 0.0f, 1.0f, 1.0f};
    load<Entry::nvnCommandBufferClearColor>()(
        &recorder.commands, 0, inherited_clear.data(), 0xf);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(
        &recorder.commands, &first_object, nullptr, &whole, readback, NVN_COPY_FLAGS_NONE);
    const auto inherited_handle = recorder.finish();
    const auto inherited_fence = backend->submit({&inherited_handle, 1});
    CHECK(inherited_fence != 0);
    CHECK(backend->wait_fence(inherited_fence, 10'000'000'000ull));
    CHECK(bytes[0x2000] == 0 && bytes[0x2001] == 0 &&
          bytes[0x2002] == 255 && bytes[0x2003] == 255);

    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferBindProgram>()(
        &recorder.commands, &compute_program, NVN_SHADER_STAGE_COMPUTE_BIT);
    load<Entry::nvnCommandBufferBindStorageBuffer>()(
        &recorder.commands, NVN_SHADER_STAGE_COMPUTE, 0, dispatch_output, 16);
    // Empty grids do no work and must not reach the Metal encoder.
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 0, 1, 1);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 0);
    load<Entry::nvnCommandBufferDispatchCompute>()(&recorder.commands, 1, 1, 1);
    load<Entry::nvnCommandBufferReportCounter>()(
        &recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP, counter);
    NVNevent event{};
    std::uint32_t event_value = 0;
    reinterpret_cast<EventState*>(&event)->value = &event_value;
    load<Entry::nvnCommandBufferSignalEvent>()(
        &recorder.commands, &event, NVN_EVENT_SIGNAL_MODE_WRITE,
        NVN_EVENT_SIGNAL_LOCATION_BOTTOM, 0, 0x31415926);
    NVNsync sync{};
    load<Entry::nvnCommandBufferFenceSync>()(
        &recorder.commands, &sync, NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE, 0);
    const auto dispatch_handle = recorder.finish();
    const auto dispatch_fence = backend->submit({&dispatch_handle, 1});
    CHECK(dispatch_fence != 0);
    CHECK(reinterpret_cast<SyncState*>(&sync)->fence == dispatch_fence);
    CHECK(backend->wait_fence(dispatch_fence, 10'000'000'000ull));
    std::array<std::uint32_t, 4> compute_words{};
    std::memcpy(compute_words.data(), bytes + 0x3000, sizeof compute_words);
    CHECK((compute_words == std::array<std::uint32_t, 4>{10, 11, 12, 13}));
    CHECK(event_value == 0x31415926);
    NVNcounterData timestamp{};
    std::memcpy(&timestamp, bytes + 0x4000, sizeof timestamp);
    CHECK(timestamp.timestamp != 0 && timestamp.counter == 0);
    CHECK(backend->completed_fence() >= dispatch_fence);

    const auto stats = backend->statistics();
    CHECK(stats.draws == 13);
    CHECK(stats.dispatches == 1);
    CHECK(stats.empty_dispatches == 2);
    CHECK(stats.synchronize_gpu_waits == 0);
    CHECK(stats.copies >= 5);
    CHECK(stats.clears == 3);
    CHECK(resources.texture_writes >= 3);
    CHECK(resources.buffer_writes >= 4);

    for (const auto handle : {copy_handle, clear_handle, inherited_handle, dispatch_handle})
        load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&graphics_program);
    load<Entry::nvnProgramFinalize>()(&compute_program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A float32 attribute feeds an integer shader input as raw register bits.
// Preserve negative zero and a NaN payload, and keep the preceding draw's pass.
void test_vertex_input_bits(id<MTLDevice> device, id<MTL4CommandQueue> queue, bool signed_input) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget;
    resources.textures[11] = [device newTextureWithDescriptor:descriptor];
    auto state = texture_state(11, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    Recorder recorder;
    NVNprogram plain{}, bits{};
    CHECK(load<Entry::nvnProgramInitialize>()(&plain, &recorder.device));
    CHECK(load<Entry::nvnProgramInitialize>()(&bits, &recorder.device));
    const NVNshaderData plain_shaders[]{{1, nullptr}, {1, nullptr}};
    const NVNshaderData bit_shaders[]{{signed_input ? 22u : 21u, nullptr}, {1, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&plain, 2, plain_shaders));
    CHECK(load<Entry::nvnProgramSetShaders>()(&bits, 2, bit_shaders));
    const std::uint32_t vertices[]{
        0xbf800000, 0xbf800000, 0x80000000, 0x7fc12345,
        0x40400000, 0xbf800000, 0x80000000, 0x7fc12345,
        0xbf800000, 0x40400000, 0x80000000, 0x7fc12345};
    std::memcpy(resources.memory.contents, vertices, sizeof vertices);
    NVNvertexAttribState attribute{};
    NVNvertexStreamState stream{};
    load<Entry::nvnVertexAttribStateSetDefaults>()(&attribute);
    load<Entry::nvnVertexAttribStateSetFormat>()(&attribute, NVN_FORMAT_RGBA32F, 0);
    load<Entry::nvnVertexStreamStateSetDefaults>()(&stream);
    load<Entry::nvnVertexStreamStateSetStride>()(&stream, 16);
    const float black[]{0, 0, 0, 1};
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, black, 15);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &plain, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &bits, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindVertexAttribState>()(&recorder.commands, 1, &attribute);
    load<Entry::nvnCommandBufferBindVertexStreamState>()(&recorder.commands, 1, &stream);
    load<Entry::nvnCommandBufferBindVertexBuffer>()(&recorder.commands, 0, TestResources::base, sizeof vertices);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 2, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    CHECK(backend->statistics().skipped == 0 && backend->statistics().draws == 2);
    CHECK(backend->statistics().render_passes == 1);
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[11] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned pixel = 0; pixel < 16; ++pixel)
        CHECK(pixels[pixel * 4] == 0 && pixels[pixel * 4 + 1] == 255 &&
              pixels[pixel * 4 + 2] == 0 && pixels[pixel * 4 + 3] == 255);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&plain);
    load<Entry::nvnProgramFinalize>()(&bits);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// Integer-to-float vertex formats (the UI2F and I2F families) bind the
// integer layout; Metal converts the integers to the float input. Positions
// of -1 and 3 as 16-bit integers cover the whole target.
void test_scaled_vertex_format(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    resources.require_completion = true;
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> target = [device newTextureWithDescriptor:descriptor];
    auto target_state = texture_state(11, 4, 4);
    resources.textures = {{11, target}};
    auto target_object = texture_object(target_state);
    const NVNtexture* targets[]{&target_object};
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
    Recorder recorder;
    NVNprogram attributes{};
    CHECK(load<Entry::nvnProgramInitialize>()(&attributes, &recorder.device));
    const NVNshaderData attribute_shaders[]{{3, nullptr}, {3, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&attributes, 2, attribute_shaders));
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    const std::int16_t vertices[]{-1, -1, 0, 0, 3, -1, 0, 0, -1, 3, 0, 0};
    std::memcpy(bytes, vertices, sizeof vertices);
    NVNvertexAttribState attribute{};
    NVNvertexStreamState streams[16]{};
    load<Entry::nvnVertexAttribStateSetDefaults>()(&attribute);
    load<Entry::nvnVertexAttribStateSetFormat>()(&attribute, NVN_FORMAT_RGB16_I2F, 0);
    load<Entry::nvnVertexAttribStateSetStreamIndex>()(&attribute, 0);
    for (auto& stream : streams) load<Entry::nvnVertexStreamStateSetDefaults>()(&stream);
    load<Entry::nvnVertexStreamStateSetStride>()(&streams[0], 8);
    const float black[]{0, 0, 0, 1}, green[]{0, 1, 0, 1};
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, TestResources::base + 0x200,
        sizeof green, 0, sizeof green, green);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, black, 15);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &attributes, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindVertexAttribState>()(&recorder.commands, 1, &attribute);
    load<Entry::nvnCommandBufferBindVertexStreamState>()(&recorder.commands, 16, streams);
    load<Entry::nvnCommandBufferBindVertexBuffer>()(&recorder.commands, 0, TestResources::base, sizeof vertices);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        TestResources::base + 0x200, sizeof green);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x1000, 0);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    if (!fence) std::fprintf(stderr, "scaled vertex format replay: %s\n", backend->last_error().detail.c_str());
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(backend->statistics().skipped == 0);
    const auto* output = reinterpret_cast<const std::uint8_t*>(bytes);
    bool covered = true;
    for (unsigned pixel = 0; pixel < 16; ++pixel) {
        const auto* rgba = output + 0x1000 + pixel * 4;
        covered = covered && rgba[0] == 0 && rgba[1] == 255 && rgba[2] == 0 && rgba[3] == 255;
    }
    CHECK(covered);
}

// A vertex stage that fetches its own inputs: positions as 16-bit integers
// converted to float (RGB16_I2F) in stream 3 at offset 4, and a per-instance
// shift (RG32F, divisor 1) in stream 1. The first draw has no shift; the
// second, with a different layout and the same pipeline, moves the triangle
// off the target through the shift of instance 1, so only instance 0 covers.
void test_vertex_fetch_draw(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    resources.require_completion = true;
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> target = [device newTextureWithDescriptor:descriptor];
    auto target_state = texture_state(11, 4, 4);
    resources.textures = {{11, target}};
    auto target_object = texture_object(target_state);
    const NVNtexture* targets[]{&target_object};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{23, nullptr}, {3, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    // Stream 3: 12-byte vertices, the position at byte 4.
    const std::int16_t positions[]{0, 0, -1, -1, 0, 0, 0, 0, 3, -1, 0, 0, 0, 0, -1, 3, 0, 0};
    std::memcpy(bytes, positions, sizeof positions);
    // Stream 1: one shift per instance, instance 1 far off the target.
    const float shifts[]{0, 0, 8, 8};
    std::memcpy(bytes + 0x100, shifts, sizeof shifts);
    NVNvertexAttribState attributes[6]{};
    NVNvertexStreamState streams[16]{};
    for (auto& attribute : attributes) load<Entry::nvnVertexAttribStateSetDefaults>()(&attribute);
    load<Entry::nvnVertexAttribStateSetFormat>()(&attributes[2], NVN_FORMAT_RGB16_I2F, 4);
    load<Entry::nvnVertexAttribStateSetStreamIndex>()(&attributes[2], 3);
    for (auto& stream : streams) load<Entry::nvnVertexStreamStateSetDefaults>()(&stream);
    load<Entry::nvnVertexStreamStateSetStride>()(&streams[3], 12);
    const float black[]{0, 0, 0, 1}, green[]{0, 1, 0, 1};
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, TestResources::base + 0x200,
        sizeof green, 0, sizeof green, green);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, black, 15);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindVertexAttribState>()(&recorder.commands, 6, attributes);
    load<Entry::nvnCommandBufferBindVertexStreamState>()(&recorder.commands, 16, streams);
    load<Entry::nvnCommandBufferBindVertexBuffer>()(&recorder.commands, 3, TestResources::base, sizeof positions);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        TestResources::base + 0x200, sizeof green);
    // Attribute 5 disabled: the shift reads zeros.
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnVertexAttribStateSetFormat>()(&attributes[5], NVN_FORMAT_RG32F, 0);
    load<Entry::nvnVertexAttribStateSetStreamIndex>()(&attributes[5], 1);
    load<Entry::nvnVertexStreamStateSetStride>()(&streams[1], 8);
    load<Entry::nvnVertexStreamStateSetDivisor>()(&streams[1], 1);
    load<Entry::nvnCommandBufferBindVertexAttribState>()(&recorder.commands, 6, attributes);
    load<Entry::nvnCommandBufferBindVertexStreamState>()(&recorder.commands, 16, streams);
    load<Entry::nvnCommandBufferBindVertexBuffer>()(&recorder.commands, 1, TestResources::base + 0x100,
        sizeof shifts);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 2, 0, 2, 4);
    load<Entry::nvnCommandBufferDrawArraysInstanced>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3,
        0, 2);
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x1000, 0);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    if (!fence) std::fprintf(stderr, "vertex fetch replay: %s\n", backend->last_error().detail.c_str());
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(backend->statistics().skipped == 0 && backend->statistics().draws == 2);
    const auto* output = reinterpret_cast<const std::uint8_t*>(bytes);
    bool covered = true;
    for (unsigned pixel = 0; pixel < 16; ++pixel) {
        const auto* rgba = output + 0x1000 + pixel * 4;
        covered = covered && rgba[0] == 0 && rgba[1] == 255 && rgba[2] == 0 && rgba[3] == 255;
    }
    CHECK(covered);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// The fetch reads each component with one load of its size, so a draw whose
// attribute offset is not aligned to its 16-bit components is skipped as
// unsupported work with its reason reported, not drawn.
void test_vertex_fetch_misaligned(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    resources.require_completion = true;
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures = {{11, [device newTextureWithDescriptor:descriptor]}};
    auto target_state = texture_state(11, 4, 4);
    auto target_object = texture_object(target_state);
    const NVNtexture* targets[]{&target_object};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{23, nullptr}, {3, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    std::memset(resources.memory.contents, 0, resources.memory.length);
    NVNvertexAttribState attributes[6]{};
    NVNvertexStreamState streams[16]{};
    for (auto& attribute : attributes) load<Entry::nvnVertexAttribStateSetDefaults>()(&attribute);
    load<Entry::nvnVertexAttribStateSetFormat>()(&attributes[2], NVN_FORMAT_RGB16_I2F, 3);
    load<Entry::nvnVertexAttribStateSetStreamIndex>()(&attributes[2], 3);
    for (auto& stream : streams) load<Entry::nvnVertexStreamStateSetDefaults>()(&stream);
    load<Entry::nvnVertexStreamStateSetStride>()(&streams[3], 12);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindVertexAttribState>()(&recorder.commands, 6, attributes);
    load<Entry::nvnCommandBufferBindVertexStreamState>()(&recorder.commands, 16, streams);
    load<Entry::nvnCommandBufferBindVertexBuffer>()(&recorder.commands, 3, TestResources::base, 64);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    if (fence) backend->wait_fence(fence, 10'000'000'000ull);
    CHECK(backend->statistics().draws == 0 && backend->statistics().skipped == 1);
    bool reported = false;
    for (const auto& [reason, count] : backend->skip_reasons())
        reported = reported || (reason.find("not aligned") != std::string::npos && count == 1);
    CHECK(reported);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_state_and_deferred_bindings(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    resources.require_completion = true;
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> target = [device newTextureWithDescriptor:descriptor];
    id<MTLTexture> red = [device newTextureWithDescriptor:descriptor];
    id<MTLTexture> blue = [device newTextureWithDescriptor:descriptor];
    descriptor.pixelFormat = MTLPixelFormatDepth32Float;
    id<MTLTexture> depth = [device newTextureWithDescriptor:descriptor];
    auto target_state = texture_state(11, 4, 4), red_state = texture_state(12, 4, 4),
         blue_state = texture_state(13, 4, 4), depth_state = texture_state(14, 4, 4);
    depth_state.description.format = NVN_FORMAT_DEPTH32F;
    resources.textures = {{11, target}, {12, red}, {13, blue}, {14, depth}};
    resources.texture_ids = {{12, red}, {13, blue}};
    auto* sampler_descriptor = [MTLSamplerDescriptor new];
    sampler_descriptor.supportArgumentBuffers = YES;
    resources.samplers[1] = [device newSamplerStateWithDescriptor:sampler_descriptor];
    auto target_object = texture_object(target_state), red_object = texture_object(red_state),
         blue_object = texture_object(blue_state), depth_object = texture_object(depth_state);
    const NVNtexture* targets[]{&target_object};
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 1};
    Recorder recorder;
    NVNprogram attributes{}, sampled{};
    CHECK(load<Entry::nvnProgramInitialize>()(&attributes, &recorder.device));
    CHECK(load<Entry::nvnProgramInitialize>()(&sampled, &recorder.device));
    const NVNshaderData attribute_shaders[]{{3, nullptr}, {3, nullptr}};
    const NVNshaderData sampled_shaders[]{{4, nullptr}, {4, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&attributes, 2, attribute_shaders));
    CHECK(load<Entry::nvnProgramSetShaders>()(&sampled, 2, sampled_shaders));
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    const float vertices[]{-1, -1, 0.5f, 3, -1, 0.5f, -1, 3, 0.5f};
    std::memcpy(bytes, vertices, sizeof vertices);
    const std::uint8_t indices[]{0, 1, 2, 0xff, 0, 1, 2};
    std::memcpy(bytes + 0x100, indices, sizeof indices);
    NVNvertexAttribState attribute{};
    NVNvertexStreamState streams[16]{};
    load<Entry::nvnVertexAttribStateSetDefaults>()(&attribute);
    load<Entry::nvnVertexAttribStateSetFormat>()(&attribute, NVN_FORMAT_RGB32F, 0);
    load<Entry::nvnVertexAttribStateSetStreamIndex>()(&attribute, 15);
    for (auto& stream : streams) load<Entry::nvnVertexStreamStateSetDefaults>()(&stream);
    load<Entry::nvnVertexStreamStateSetStride>()(&streams[15], 12);
    NVNdepthStencilState depth_test{};
    load<Entry::nvnDepthStencilStateSetDefaults>()(&depth_test);
    load<Entry::nvnDepthStencilStateSetDepthTestEnable>()(&depth_test, 1);
    load<Entry::nvnDepthStencilStateSetDepthFunc>()(&depth_test, NVN_DEPTH_FUNC_LEQUAL);
    NVNchannelMaskState channels{};
    load<Entry::nvnChannelMaskStateSetDefaults>()(&channels);
    NVNcolorState color{};
    load<Entry::nvnColorStateSetDefaults>()(&color);
    NVNblendState blend{};
    load<Entry::nvnBlendStateSetDefaults>()(&blend);
    load<Entry::nvnBlendStateSetBlendFunc>()(&blend, NVN_BLEND_FUNC_CONSTANT_COLOR,
        NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_COLOR, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO);
    const float black[]{0, 0, 0, 1}, red_value[]{1, 0, 0, 1}, blue_value[]{0, 0, 1, 1};
    const float green[]{0, 1, 0, 1}, half[]{0.5f, 0.5f, 0.5f, 0.5f};
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, TestResources::base + 0x200,
        sizeof red_value, 0, sizeof red_value, red_value);
    const auto preload = recorder.finish();
    CHECK(backend->submit({&preload, 1}) != 0);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, &depth_object, nullptr);
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, black, 15);
    load<Entry::nvnCommandBufferClearDepthStencil>()(&recorder.commands, 1, 1, 0, 0);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &attributes, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindVertexAttribState>()(&recorder.commands, 1, &attribute);
    load<Entry::nvnCommandBufferBindVertexStreamState>()(&recorder.commands, 16, streams);
    load<Entry::nvnCommandBufferBindVertexBuffer>()(&recorder.commands, 15, TestResources::base, sizeof vertices);
    load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &depth_test);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        TestResources::base + 0x200, sizeof red_value);
    load<Entry::nvnCommandBufferResetCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED);
    load<Entry::nvnCommandBufferSetPrimitiveRestart>()(&recorder.commands, 1, 0xff);
    load<Entry::nvnCommandBufferDrawElementsInstanced>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES,
        NVN_INDEX_TYPE_UNSIGNED_BYTE, 7, TestResources::base + 0x100, 0, 5, 1);
    load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_SAMPLES_PASSED,
        TestResources::base + 0x300);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x1000, 0);
    // A later uniform update must not change the earlier draw, and a new
    // blend/scissor state must be used with the same resolved program.
    load<Entry::nvnColorStateSetBlendEnable>()(&color, 0, 1);
    load<Entry::nvnCommandBufferBindColorState>()(&recorder.commands, &color);
    load<Entry::nvnCommandBufferBindBlendState>()(&recorder.commands, &blend);
    load<Entry::nvnCommandBufferSetBlendColor>()(&recorder.commands, half);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, -1, 0, 3, 4);
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, TestResources::base + 0x200,
        sizeof green, 0, sizeof green, green);
    const MTLDrawPrimitivesIndirectArguments arguments{3, 1, 0, 7};
    load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, TestResources::base + 0x400,
        sizeof arguments, 0, sizeof arguments, &arguments);
    load<Entry::nvnCommandBufferClearBuffer>()(&recorder.commands, TestResources::base + 0x500, 4, 2);
    load<Entry::nvnCommandBufferMultiDrawArraysIndirectCount>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES,
        TestResources::base + 0x400, TestResources::base + 0x500, 1, sizeof arguments);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x2000, 0);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    if (!fence) std::fprintf(stderr, "state replay: %s\n", backend->last_error().detail.c_str());
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    const auto* output = reinterpret_cast<const std::uint8_t*>(bytes);
    CHECK(output[0x1000] == 255 && output[0x1001] == 0);
    CHECK(output[0x2000] >= 127 && output[0x2000] <= 128);
    CHECK(output[0x2001] >= 127 && output[0x2001] <= 128);
    CHECK(output[0x2008] == 255 && output[0x2009] == 0);
    NVNcounterData samples{};
    std::memcpy(&samples, bytes + 0x300, sizeof samples);
    CHECK(samples.counter == 32);
    CHECK(backend->statistics().graphics_pipelines == 2);
    // The blend change after the first draw makes a second pipeline; it was
    // started when the first one missed, before its draw was reached.
    CHECK(backend->statistics().predicted_pipelines == 1);

    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 4, 4);
    load<Entry::nvnCommandBufferSetRenderEnableConditional>()(&recorder.commands,
        NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL, TestResources::base + 0x300);
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, blue_value, 15);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x4000, 0);
    load<Entry::nvnCommandBufferReportCounter>()(&recorder.commands, NVN_COUNTER_TYPE_TIMESTAMP_TOP,
        TestResources::base + 0x300);
    load<Entry::nvnCommandBufferSetRenderEnableConditional>()(&recorder.commands,
        NVN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL, TestResources::base + 0x300);
    load<Entry::nvnCommandBufferClearColor>()(&recorder.commands, 0, blue_value, 15);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x5000, 0);
    const auto conditional = recorder.finish();
    const auto conditional_fence = backend->submit({&conditional, 1});
    CHECK(conditional_fence && backend->wait_fence(conditional_fence, 10'000'000'000ull));
    CHECK(output[0x4000] >= 127 && output[0x4000] <= 128 && output[0x4002] == 0);
    CHECK(output[0x5000] == 0 && output[0x5002] == 255);

    // Deferred handles are snapshotted before GPU work changes their source.
    auto* handles = reinterpret_cast<NVNtextureHandle*>(bytes + 0x6000);
    *handles = combined_handle(12, 1);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetScissor>()(&recorder.commands, 0, 0, 4, 4);
    load<Entry::nvnCommandBufferClearTexture>()(&recorder.commands, &red_object, nullptr, &whole, red_value, 15);
    load<Entry::nvnCommandBufferClearTexture>()(&recorder.commands, &blue_object, nullptr, &whole, blue_value, 15);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &sampled, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnColorStateSetDefaults>()(&color);
    load<Entry::nvnCommandBufferBindColorState>()(&recorder.commands, &color);
    load<Entry::nvnDepthStencilStateSetDefaults>()(&depth_test);
    load<Entry::nvnCommandBufferBindDepthStencilState>()(&recorder.commands, &depth_test);
    load<Entry::nvnCommandBufferClearBuffer>()(&recorder.commands, TestResources::base + 0x6000, 4,
        static_cast<std::uint32_t>(combined_handle(13, 1)));
    load<Entry::nvnCommandBufferBindTexturesDeferred>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, 1, handles);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x3000, 0);
    const auto deferred = recorder.finish();
    const auto deferred_fence = backend->submit({&deferred, 1});
    if (!deferred_fence) std::fprintf(stderr, "deferred replay: %s\n", backend->last_error().detail.c_str());
    CHECK(deferred_fence && backend->wait_fence(deferred_fence, 10'000'000'000ull));
    CHECK(output[0x3000] == 255 && output[0x3002] == 0);
    CHECK(handle_texture_id(*handles) == 13);
    CHECK(resources.coherence_waits > 0);
    // Empty and unavailable sampler slots use the cached default descriptor.
    const auto skipped_before = backend->statistics().skipped;
    resources.samplers.erase(1);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        combined_handle(13, 4095));
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x7000, 0);
    load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        combined_handle(12, 0));
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x8000, 0);
    const auto empty_samplers = recorder.finish();
    CHECK(backend->wait_fence(backend->submit({&empty_samplers, 1}), 10'000'000'000ull));
    CHECK(backend->statistics().skipped == skipped_before);
    CHECK(output[0x7000] == 0 && output[0x7002] == 255);
    CHECK(output[0x8000] == 255 && output[0x8002] == 0);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, empty_samplers);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, deferred);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, conditional);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, preload);
    load<Entry::nvnProgramFinalize>()(&attributes);
    load<Entry::nvnProgramFinalize>()(&sampled);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_mip_chain_levels(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    // A pass renders mip level 1 of a storage while sampling a view of all its
    // levels, as mip chains do: the sampler's level-of-detail clamp selects
    // the level above (an upsample) or below (a downsample). Both must read
    // their level, and the rendered colour must land in level 1 afterwards.
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:YES];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> storage = [device newTextureWithDescriptor:descriptor];
    std::array<std::uint32_t, 16> texels{};
    texels.fill(0xff0000ffu);  // red
    [storage replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels.data() bytesPerRow:16];
    texels.fill(0xffff0000u);  // blue
    [storage replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:1 withBytes:texels.data() bytesPerRow:8];
    texels.fill(0xff00ff00u);  // green
    [storage replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:2 withBytes:texels.data() bytesPerRow:4];
    id<MTLTexture> level_one = [storage newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
        textureType:MTLTextureType2D levels:NSMakeRange(1, 1) slices:NSMakeRange(0, 1)];
    resources.textures = {{21, level_one}};
    resources.texture_ids = {{22, storage}};
    for (const auto& [slot, level] : {std::pair{1, 2.0f}, std::pair{2, 0.0f}}) {
        auto* sampler_descriptor = [MTLSamplerDescriptor new];
        sampler_descriptor.mipFilter = MTLSamplerMipFilterNearest;
        sampler_descriptor.lodMinClamp = level;
        sampler_descriptor.lodMaxClamp = level;
        sampler_descriptor.supportArgumentBuffers = YES;
        resources.samplers[slot] = [device newSamplerStateWithDescriptor:sampler_descriptor];
    }
    auto target_state = texture_state(21, 2, 2);
    auto target_object = texture_object(target_state);
    const NVNtexture* targets[]{&target_object};
    const NVNcopyRegion whole{0, 0, 0, 2, 2, 1};
    Recorder recorder;
    NVNprogram mip{};
    CHECK(load<Entry::nvnProgramInitialize>()(&mip, &recorder.device));
    const NVNshaderData shaders[]{{12, nullptr}, {12, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&mip, 2, shaders));
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &mip, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        combined_handle(22, 1));
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x1000, 0);
    load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        combined_handle(22, 2));
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x2000, 0);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    if (!fence) std::fprintf(stderr, "mip chain replay: %s\n", backend->last_error().detail.c_str());
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    const auto* output = reinterpret_cast<const std::uint8_t*>(bytes);
    for (unsigned texel = 0; texel < 4; ++texel) {
        const auto* up = output + 0x1000 + texel * 4;
        const auto* down = output + 0x2000 + texel * 4;
        CHECK(up[0] == 0 && up[1] == 255 && up[2] == 0);      // level 2 above
        CHECK(down[0] == 255 && down[1] == 0 && down[2] == 0);  // level 0 below
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&mip);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A game may register one sampler ID again between the draws of one command
// buffer. Each draw samples with the parameters registered before its binding
// was recorded, not with the last registration before the replay.
// An in-stream update rebinds a draw stage's uniform buffers; its textures and
// storage buffers resolve again only when the update lands on the word their
// handle or descriptor is read from. Each case draws twice in one pass with an
// update between, and the final pixels show which bindings the second draw used.
enum class RebindCase { texture_handle, storage_descriptor, plain_data };

void run_uniform_only_rebind(id<MTLDevice> device, id<MTL4CommandQueue> queue, RebindCase which) {
    TestResources resources(device);
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    resources.textures[96] = [device newTextureWithDescriptor:descriptor];
    std::array<std::uint32_t, 16> texels{};
    id<MTLTexture> red = [device newTextureWithDescriptor:descriptor];
    id<MTLTexture> green = [device newTextureWithDescriptor:descriptor];
    texels.fill(0xff0000ffu);
    [red replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels.data() bytesPerRow:16];
    texels.fill(0xff00ff00u);
    [green replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels.data() bytesPerRow:16];
    resources.texture_ids = {{31, red}, {32, green}};
    auto state = texture_state(96, 4, 4);
    auto target = texture_object(state);
    const NVNtexture* targets[]{&target};
    constexpr auto first = TestResources::base + 0x200, second = TestResources::base + 0x300;
    const std::array<float, 4> white{1, 1, 1, 1}, blue{0, 0, 1, 1};
    const auto handle = [](unsigned id) { return static_cast<std::uint32_t>(combined_handle(id, 0)); };
    const bool storage = which == RebindCase::storage_descriptor;
    // Storage buffers: the first holds 16 (the reader answers green), the second zero.
    const std::uint32_t sixteen = 16;
    std::memcpy(bytes + 0x400, &sixteen, sizeof sixteen);
    struct Descriptor { std::uint64_t address; std::uint32_t size; } before{TestResources::base + 0x400, 16},
        after{TestResources::base + 0x500, 16};
    if (storage) std::memcpy(bytes + 0x200, &before, 12);
    else std::memcpy(bytes + 0x200, std::array{which == RebindCase::plain_data ? handle(32) : handle(31)}.data(), 4);
    std::memcpy(bytes + 0x300, white.data(), sizeof white);
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{storage ? 16u : 15u, nullptr}, {storage ? 16u : 15u, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0, first, 64);
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 1, second, 64);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    switch (which) {
    case RebindCase::texture_handle: {
        const auto next = handle(32);
        load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, first, 64, 0, 4, &next);
        break;
    }
    case RebindCase::storage_descriptor:
        load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, first, 64, 0, 12, &after);
        break;
    case RebindCase::plain_data:
        load<Entry::nvnCommandBufferUpdateUniformBuffer>()(&recorder.commands, second, 64, 0, 16, blue.data());
        break;
    }
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    const auto commands = recorder.finish();
    const auto fence = backend->submit({&commands, 1});
    if (!fence) std::fprintf(stderr, "uniform-only rebind replay: %s\n", backend->last_error().detail.c_str());
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    CHECK(!backend->last_error());
    const auto stats = backend->statistics();
    CHECK(stats.draws == 2 && stats.render_passes == 1);
    // The vertex stage has nothing a update could stale, so its second bind is always
    // uniform-only; the fragment stage's is, unless the update hit its handle or descriptor.
    CHECK(stats.stage_uniform_binds == (which == RebindCase::plain_data ? 2u : 1u));
    std::array<std::uint8_t, 64> pixels{};
    [resources.textures[96] getBytes:pixels.data() bytesPerRow:16
        fromRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0];
    for (unsigned texel = 0; texel < 16; ++texel) {
        const auto* pixel = &pixels[texel * 4];
        switch (which) {
        case RebindCase::texture_handle:       // the second texture, green
            CHECK(pixel[0] == 0 && pixel[1] == 255 && pixel[2] == 0); break;
        case RebindCase::storage_descriptor:   // the second buffer reads zero: red
            CHECK(pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 0); break;
        case RebindCase::plain_data:           // green sampled under a blue tint: black
            CHECK(pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0); break;
        }
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, commands);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_uniform_only_rebind(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    run_uniform_only_rebind(device, queue, RebindCase::texture_handle);
    run_uniform_only_rebind(device, queue, RebindCase::storage_descriptor);
    run_uniform_only_rebind(device, queue, RebindCase::plain_data);
}

void test_sampler_registration_order(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:YES];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> storage = [device newTextureWithDescriptor:descriptor];
    std::array<std::uint32_t, 16> texels{};
    texels.fill(0xff0000ffu);  // red
    [storage replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels.data() bytesPerRow:16];
    texels.fill(0xffff0000u);  // blue
    [storage replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:1 withBytes:texels.data() bytesPerRow:8];
    texels.fill(0xff00ff00u);  // green
    [storage replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:2 withBytes:texels.data() bytesPerRow:4];
    id<MTLTexture> level_one = [storage newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
        textureType:MTLTextureType2D levels:NSMakeRange(1, 1) slices:NSMakeRange(0, 1)];
    resources.textures = {{21, level_one}};
    resources.texture_ids = {{22, storage}};
    const auto clamped = [&](float level) {
        auto* sampler_descriptor = [MTLSamplerDescriptor new];
        sampler_descriptor.mipFilter = MTLSamplerMipFilterNearest;
        sampler_descriptor.lodMinClamp = level;
        sampler_descriptor.lodMaxClamp = level;
        sampler_descriptor.supportArgumentBuffers = YES;
        return [device newSamplerStateWithDescriptor:sampler_descriptor];
    };
    const auto base = nvn::sampler_registrations.load() + 1;
    resources.sampler_versions[3] = {{base, clamped(2.0f)}, {base + 1, clamped(0.0f)}, {base + 2, clamped(1.0f)}};
    auto target_state = texture_state(21, 2, 2);
    auto target_object = texture_object(target_state);
    const NVNtexture* targets[]{&target_object};
    const NVNcopyRegion whole{0, 0, 0, 2, 2, 1};
    Recorder recorder;
    NVNprogram mip{};
    CHECK(load<Entry::nvnProgramInitialize>()(&mip, &recorder.device));
    const NVNshaderData shaders[]{{12, nullptr}, {12, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&mip, 2, shaders));
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &mip, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    nvn::sampler_registrations.store(base);  // sampler 3 registered with level 2
    load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        combined_handle(22, 3));
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x1000, 0);
    nvn::sampler_registrations.store(base + 1);  // registered again with level 0
    load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        combined_handle(22, 3));
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x2000, 0);
    const auto handle = recorder.finish();
    nvn::sampler_registrations.store(base + 2);  // and again before the submission
    const auto fence = backend->submit({&handle, 1});
    if (!fence) std::fprintf(stderr, "sampler order replay: %s\n", backend->last_error().detail.c_str());
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    const auto* output = reinterpret_cast<const std::uint8_t*>(bytes);
    for (unsigned texel = 0; texel < 4; ++texel) {
        const auto* first = output + 0x1000 + texel * 4;
        const auto* second = output + 0x2000 + texel * 4;
        CHECK(first[0] == 0 && first[1] == 255 && first[2] == 0);    // level 2
        CHECK(second[0] == 255 && second[1] == 0 && second[2] == 0); // level 0
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&mip);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A sampler registered again between the binding and the draw applies to the
// draw: the draw stamps the registrations made before it.
void test_sampler_draw_order(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:YES];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> storage = [device newTextureWithDescriptor:descriptor];
    std::array<std::uint32_t, 16> texels{};
    texels.fill(0xff0000ffu);  // red
    [storage replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels.data() bytesPerRow:16];
    texels.fill(0xffff0000u);  // blue
    [storage replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:1 withBytes:texels.data() bytesPerRow:8];
    texels.fill(0xff00ff00u);  // green
    [storage replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:2 withBytes:texels.data() bytesPerRow:4];
    id<MTLTexture> level_one = [storage newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
        textureType:MTLTextureType2D levels:NSMakeRange(1, 1) slices:NSMakeRange(0, 1)];
    resources.textures = {{21, level_one}};
    resources.texture_ids = {{22, storage}};
    const auto clamped = [&](float level) {
        auto* sampler_descriptor = [MTLSamplerDescriptor new];
        sampler_descriptor.mipFilter = MTLSamplerMipFilterNearest;
        sampler_descriptor.lodMinClamp = level;
        sampler_descriptor.lodMaxClamp = level;
        sampler_descriptor.supportArgumentBuffers = YES;
        return [device newSamplerStateWithDescriptor:sampler_descriptor];
    };
    auto target_state = texture_state(21, 2, 2);
    auto target_object = texture_object(target_state);
    const NVNtexture* targets[]{&target_object};
    const NVNcopyRegion whole{0, 0, 0, 2, 2, 1};
    NVNprogram mip{};
    Recorder recorder;
    CHECK(load<Entry::nvnProgramInitialize>()(&mip, &recorder.device));
    const NVNshaderData shaders[]{{12, nullptr}, {12, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&mip, 2, shaders));
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    const auto record = [&](std::uint64_t output, auto&& before_draw) {
        load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
        load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
        load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &mip, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
            combined_handle(22, 3));
        before_draw();
        load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
        load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
            TestResources::base + output, 0);
        return recorder.finish();
    };
    const auto run = [&](NVNcommandHandle handle) {
        const auto fence = backend->submit({&handle, 1});
        if (!fence) std::fprintf(stderr, "sampler draw order replay: %s\n", backend->last_error().detail.c_str());
        CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    };
    const auto texel = [&](std::uint64_t output) { return reinterpret_cast<const std::uint8_t*>(bytes) + output; };
    std::memset(bytes, 0, resources.memory.length);

    // Bind, register again, draw: the draw samples with the later registration.
    const auto base = nvn::sampler_registrations.load() + 1;
    resources.sampler_versions[3] = {{base, clamped(2.0f)}, {base + 1, clamped(0.0f)}};
    nvn::sampler_registrations.store(base);
    const auto between = record(0x1000, [&] { nvn::sampler_registrations.store(base + 1); });
    run(between);
    CHECK(texel(0x1000)[0] == 255 && texel(0x1000)[1] == 0);  // level 0

    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, between);
    load<Entry::nvnProgramFinalize>()(&mip);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// A GPU synchronization in the middle of an open pass closes and reopens it:
// the second draw, after the reopen, must still reach the target, at level 0
// and at a mip level.
void test_mip_alias_reopened_pass(id<MTLDevice> device, id<MTL4CommandQueue> queue, bool level_one) {
    TestResources resources(device);
    resources.require_completion = true;
    resources.on_texture_write = [&](const TextureState&) { resources.pending_write = true; };
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:YES];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> storage = [device newTextureWithDescriptor:descriptor];
    std::array<std::uint32_t, 16> texels{};
    texels.fill(0xff0000ffu);
    [storage replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels.data() bytesPerRow:16];
    texels.fill(0xffff0000u);
    [storage replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:1 withBytes:texels.data() bytesPerRow:8];
    // The sampled level-0 copy is separate, so level 0 does not sample its own target.
    descriptor.mipmapLevelCount = 1;
    id<MTLTexture> source = [device newTextureWithDescriptor:descriptor];
    texels.fill(0xff0000ffu);
    [source replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels.data() bytesPerRow:16];
    const NSUInteger level = level_one ? 1 : 0;
    const std::uint32_t size = level_one ? 2 : 4;
    id<MTLTexture> target = [storage newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
        textureType:MTLTextureType2D levels:NSMakeRange(level, 1) slices:NSMakeRange(0, 1)];
    resources.textures = {{21, target}};
    resources.texture_ids = {{22, source}};
    auto* sampler_descriptor = [MTLSamplerDescriptor new];
    sampler_descriptor.supportArgumentBuffers = YES;
    resources.samplers[2] = [device newSamplerStateWithDescriptor:sampler_descriptor];
    auto target_state = texture_state(21, size, size);
    auto target_object = texture_object(target_state);
    const NVNtexture* targets[]{&target_object};
    const NVNcopyRegion whole{0, 0, 0, int(size), int(size), 1};
    Recorder recorder;
    NVNprogram mip{}, uniform{};
    CHECK(load<Entry::nvnProgramInitialize>()(&mip, &recorder.device));
    CHECK(load<Entry::nvnProgramInitialize>()(&uniform, &recorder.device));
    const NVNshaderData mip_shaders[]{{12, nullptr}, {12, nullptr}};
    const NVNshaderData uniform_shaders[]{{5, nullptr}, {5, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&mip, 2, mip_shaders));
    CHECK(load<Entry::nvnProgramSetShaders>()(&uniform, 2, uniform_shaders));
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    const float green[]{0, 1, 0, 1};
    std::memcpy(bytes + 0x200, green, sizeof green);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 1, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &mip, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferBindTexture>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        combined_handle(22, 2));
    load<Entry::nvnCommandBufferBindUniformBuffer>()(&recorder.commands, NVN_SHADER_STAGE_FRAGMENT, 0,
        TestResources::base + 0x200, sizeof green);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &uniform, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &target_object, nullptr, &whole,
        TestResources::base + 0x1000, 0);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    const auto* output = reinterpret_cast<const std::uint8_t*>(bytes) + 0x1000;
    CHECK(output[0] == 0 && output[1] == 255 && output[2] == 0);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&mip);
    load<Entry::nvnProgramFinalize>()(&uniform);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

// The same subresource attached at two colour slots, with the draw writing
// only slot 0: the written colour must survive, at level 0 and at a mip level.
void test_mip_alias_shared_subresource(id<MTLDevice> device, id<MTL4CommandQueue> queue, bool level_one) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:YES];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> storage = [device newTextureWithDescriptor:descriptor];
    std::array<std::uint32_t, 16> texels{};
    texels.fill(0xff0000ffu);  // red everywhere
    [storage replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:texels.data() bytesPerRow:16];
    [storage replaceRegion:MTLRegionMake2D(0, 0, 2, 2) mipmapLevel:1 withBytes:texels.data() bytesPerRow:8];
    const NSUInteger level = level_one ? 1 : 0;
    const std::uint32_t size = level_one ? 2 : 4;
    id<MTLTexture> first = [storage newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
        textureType:MTLTextureType2D levels:NSMakeRange(level, 1) slices:NSMakeRange(0, 1)];
    id<MTLTexture> second = [storage newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm
        textureType:MTLTextureType2D levels:NSMakeRange(level, 1) slices:NSMakeRange(0, 1)];
    resources.textures = {{31, first}, {32, second}};
    auto first_state = texture_state(31, size, size), second_state = texture_state(32, size, size);
    auto first_object = texture_object(first_state), second_object = texture_object(second_state);
    const NVNtexture* targets[]{&first_object, &second_object};
    const NVNcopyRegion whole{0, 0, 0, int(size), int(size), 1};
    Recorder recorder;
    NVNprogram program{};
    CHECK(load<Entry::nvnProgramInitialize>()(&program, &recorder.device));
    const NVNshaderData shaders[]{{10, nullptr}, {10, nullptr}};
    CHECK(load<Entry::nvnProgramSetShaders>()(&program, 2, shaders));
    auto* bytes = static_cast<std::byte*>(resources.memory.contents);
    std::memset(bytes, 0, resources.memory.length);
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferSetRenderTargets>()(&recorder.commands, 2, targets, nullptr, nullptr, nullptr);
    load<Entry::nvnCommandBufferBindProgram>()(&recorder.commands, &program, NVN_SHADER_STAGE_ALL_GRAPHICS_BITS);
    load<Entry::nvnCommandBufferDrawArrays>()(&recorder.commands, NVN_DRAW_PRIMITIVE_TRIANGLES, 0, 3);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &first_object, nullptr, &whole,
        TestResources::base + 0x1000, 0);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    const auto* output = reinterpret_cast<const std::uint8_t*>(bytes) + 0x1000;
    CHECK(output[0] == 0 && output[1] == 0 && output[2] == 255);
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnProgramFinalize>()(&program);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

void test_resolve_and_partial_clear(id<MTLDevice> device, id<MTL4CommandQueue> queue) {
    TestResources resources(device);
    auto backend = replay::create_replay_backend(device, queue, resources);
    CHECK(backend != nullptr);
    if (!backend) return;
    auto* descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
        width:4 height:4 mipmapped:NO];
    descriptor.storageMode = MTLStorageModePrivate;
    descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    auto resolved = [device newTextureWithDescriptor:descriptor];
    descriptor.textureType = MTLTextureType2DMultisample;
    descriptor.sampleCount = 4;
    auto multisample = [device newTextureWithDescriptor:descriptor];
    CHECK(resolved && multisample);
    if (!resolved || !multisample) return;
    auto resolved_state = texture_state(21, 4, 4), multisample_state = texture_state(22, 4, 4);
    multisample_state.description.samples = 4;
    resources.textures = {{21, resolved}, {22, multisample}};
    auto resolved_object = texture_object(resolved_state), multisample_object = texture_object(multisample_state);
    Recorder recorder;
    const NVNcopyRegion whole{0, 0, 0, 4, 4, 1}, middle{1, 1, 0, 2, 2, 1};
    const float blue[]{0, 0, 1, 1}, red[]{1, 0, 0, 1};
    load<Entry::nvnCommandBufferBeginRecording>()(&recorder.commands);
    load<Entry::nvnCommandBufferClearTexture>()(&recorder.commands, &multisample_object, nullptr, &whole, blue, 15);
    load<Entry::nvnCommandBufferDownsample>()(&recorder.commands, &multisample_object, &resolved_object);
    load<Entry::nvnCommandBufferClearTexture>()(&recorder.commands, &resolved_object, nullptr, &middle, red, 1);
    load<Entry::nvnCommandBufferCopyTextureToBuffer>()(&recorder.commands, &resolved_object, nullptr, &whole,
        TestResources::base, 0);
    const auto handle = recorder.finish();
    const auto fence = backend->submit({&handle, 1});
    if (!fence) std::fprintf(stderr, "resolve replay: %s\n", backend->last_error().detail.c_str());
    CHECK(fence && backend->wait_fence(fence, 10'000'000'000ull));
    const auto* output = static_cast<const std::uint8_t*>(resources.memory.contents);
    for (unsigned y = 0; y != 4; ++y) for (unsigned x = 0; x != 4; ++x) {
        const auto offset = (y * 4 + x) * 4;
        CHECK(output[offset] == (x >= 1 && x < 3 && y >= 1 && y < 3 ? 255 : 0));
        CHECK(output[offset + 1] == 0 && output[offset + 2] == 255 && output[offset + 3] == 255);
    }
    load<Entry::nvnDeviceFinalizeCommandHandle>()(&recorder.device, handle);
    load<Entry::nvnCommandBufferFinalize>()(&recorder.commands);
}

} // namespace

int main() {
    @autoreleasepool {
        test_draw_tracking_contracts();
        test_draw_table_contracts();
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || ![device supportsFamily:MTLGPUFamilyMetal4] || !device.hasUnifiedMemory) {
            std::fprintf(stderr, "unified-memory Metal 4 device is unavailable\n");
            return 1;
        }
        id<MTL4CommandQueue> queue = [device newMTL4CommandQueue];
        CHECK(queue != nil);
        const auto library = build_library(R"msl(
#include <metal_stdlib>
using namespace metal;
struct RasterOut { float4 position [[position]]; };
vertex RasterOut synthetic_vertex(uint vertex_id [[vertex_id]]) {
    float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    return {float4(positions[vertex_id], 0.0, 1.0)};
}
struct UnsignedBitsIn { uint4 value [[attribute(0)]]; };
vertex RasterOut report_vertex(UnsignedBitsIn input [[stage_in]], uint vertex_id [[vertex_id]]) {
    float2 positions[3] = {float2(-1, -1), float2(3, -1), float2(-1, 3)};
    return {float4(input.value.x == 0u ? positions[vertex_id] : float2(2), 0, 1)};
}
struct SignedBitsIn { int4 value [[attribute(0)]]; };
RasterOut bit_position(uint4 value) {
    // Non-numeric payloads detect conversion and NaN canonicalization.
    bool intact = value.z == 0x80000000u && value.w == 0x7fc12345u;
    return {float4(intact ? as_type<float2>(value.xy) : float2(0), 0, 1)};
}
vertex RasterOut unsigned_bits_vertex(UnsignedBitsIn input [[stage_in]]) {
    return bit_position(input.value);
}
vertex RasterOut signed_bits_vertex(SignedBitsIn input [[stage_in]]) {
    return bit_position(as_type<uint4>(input.value));
}
struct AttributeIn { float3 position [[attribute(0)]]; };
vertex RasterOut attribute_vertex(AttributeIn input [[stage_in]], uint instance [[instance_id]],
    constant uint& base [[buffer(0)]]) {
    return {float4(input.position.xy, instance == base ? input.position.z : 0.9f, 1.0f)};
}
fragment float4 uniform_fragment(constant float4& color [[buffer(0)]]) { return color; }
fragment float4 far_uniform_fragment(constant float4* colors [[buffer(0)]]) { return colors[4]; }
struct MrtFirst { float4 separate [[color(0)]]; float4 material [[color(1)]]; };
struct MrtLast { float4 separate [[color(0)]]; float4 material [[color(2)]]; };
fragment MrtFirst mrt_first() { return {float4(0, 0, 1, 1), float4(0, 1, 0, 1)}; }
fragment MrtLast mrt_last() { return {float4(0, 0, 1, 1), float4(0, 1, 0, 1)}; }
struct MrtBoth { float4 separate [[color(0)]]; float4 first [[color(1)]]; float4 last [[color(2)]]; };
fragment MrtBoth mrt_both() { return {float4(0, 0, 1, 1), float4(0, 1, 0, 1), float4(1, 0, 1, 1)}; }
fragment float4 sampled_fragment(texture2d<float> image [[texture(0)]], sampler sample [[sampler(0)]]) {
    return image.sample(sample, float2(0.5f));
}
fragment float4 report_buffer_fragment(texture_buffer<uint, access::read> image [[texture(0)]]) {
    return image.read(image.get_width() - 1).x == 99u ? float4(1, 0, 0, 1) : float4(0, 1, 0, 1);
}
fragment float4 report_texture_fragment(texture2d<uint, access::read> image [[texture(0)]]) {
    return image.read(uint2(image.get_width() - 1, 0)).x == 99u ? float4(1, 0, 0, 1) : float4(0, 1, 0, 1);
}
fragment float4 report_samples_fragment(texture_buffer<uint, access::read> image [[texture(0)]]) {
    return image.read(0u).x == 16u ? float4(0, 1, 0, 1) : float4(1, 0, 0, 1);
}
fragment float4 mip_fragment(RasterOut input [[stage_in]], texture2d<float> image [[texture(0)]],
    sampler sample [[sampler(0)]]) {
    // The level comes from the sampler's level-of-detail clamp.
    return image.sample(sample, input.position.xy / 2.0f);
}
fragment float4 handle_fragment(texture2d<float> image [[texture(0)]], constant float4& tint [[buffer(0)]]) {
    constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
    return image.sample(nearest, float2(0.5f)) * tint;
}
vertex RasterOut writer_vertex(uint vertex_id [[vertex_id]], device uint* output [[buffer(0)]]) {
    output[0] = 16u;
    float2 positions[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    return {float4(positions[vertex_id], 0.0, 1.0)};
}
fragment float4 writer_fragment(device atomic_uint* output [[buffer(0)]]) {
    atomic_fetch_add_explicit(output, 1u, memory_order_relaxed);
    return float4(1, 0, 0, 1);
}
fragment float4 reader_fragment(const device uint* input [[buffer(0)]]) {
    return input[0] == 16u ? float4(0, 1, 0, 1) : float4(1, 0, 0, 1);
}
fragment float4 depth_reader_fragment(depth2d<float, access::read> image [[texture(0)]]) {
    return image.read(uint2(1, 1)) == 0.0f ? float4(0, 1, 0, 1) : float4(1, 0, 0, 1);
}
fragment float4 synthetic_fragment() { return float4(0.0, 1.0, 0.0, 1.0); }
kernel void synthetic_compute(device uint* output [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    output[id] = 10 + id;
}
kernel void chain_compute(const device uint* input [[buffer(0)]], device uint* output [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    output[id] = input[id] + 100;
}
)msl" + std::string(shader::vertex_fetch_msl()) + R"msl(
vertex RasterOut fetch_vertex(uint vertex_id [[vertex_id]], uint instance_id [[instance_id]],
    uint base_instance [[base_instance]], constant ArmigoVertexFetch& fetch [[buffer(0)]]) {
    const float4 position = armigo_vertex_fetch_float(fetch, 2u, vertex_id, instance_id, base_instance);
    const float4 shift = armigo_vertex_fetch_float(fetch, 5u, vertex_id, instance_id, base_instance);
    return {float4(position.xy + shift.xy, 0.0f, 1.0f)};
}
)msl");
        CHECK(!library.empty());
        if (queue && !library.empty()) {
            Stages stages;
            auto vertex = std::make_shared<platform::CompiledGraphicsStage>();
            vertex->metallib = library;
            vertex->stage = shader_input::Stage::Vertex;
            vertex->entry_point = "synthetic_vertex";
            auto fragment = std::make_shared<platform::CompiledGraphicsStage>();
            fragment->metallib = library;
            fragment->stage = shader_input::Stage::Fragment;
            fragment->entry_point = "synthetic_fragment";
            auto compute = std::make_shared<platform::CompiledGraphicsStage>();
            compute->metallib = library;
            compute->stage = shader_input::Stage::Compute;
            compute->entry_point = "synthetic_compute";
            compute->compute_local_size = {4, 1, 1};
            // Descriptors from the compute storage table (driver data 0x310 + 16 n).
            compute->storage_buffers.push_back({0, 0x310, 0, platform::CompiledShaderResourceAccess::ReadWrite, 4});
            auto chain_compute = std::make_shared<platform::CompiledGraphicsStage>(*compute);
            chain_compute->entry_point = "chain_compute";
            chain_compute->storage_buffers = {{0, 0x310, 0, platform::CompiledShaderResourceAccess::Read, 4},
                                              {0, 0x320, 1, platform::CompiledShaderResourceAccess::ReadWrite, 4}};
            stages.chain_compute = chain_compute;
            for (unsigned index = 0; index < stages.bit_vertices.size(); ++index) {
                auto bits = std::make_shared<platform::CompiledGraphicsStage>(*vertex);
                bits->entry_point = index ? "signed_bits_vertex" : "unsigned_bits_vertex";
                bits->vertex_class_inputs.push_back({0, index
                    ? graphics::VertexFormatClass::sint_class : graphics::VertexFormatClass::uint_class});
                stages.bit_vertices[index] = bits;
            }
            auto attribute_vertex = std::make_shared<platform::CompiledGraphicsStage>(*vertex);
            auto report_vertex = std::make_shared<platform::CompiledGraphicsStage>(*vertex);
            report_vertex->entry_point = "report_vertex";
            report_vertex->vertex_class_inputs.push_back({0, graphics::VertexFormatClass::uint_class});
            stages.report_vertex = report_vertex;
            attribute_vertex->entry_point = "attribute_vertex";
            attribute_vertex->vertex_attributes.push_back({0, 0, shader::graphics_msl::VertexFormat::Float32x3});
            attribute_vertex->draw_support_buffer_index = 0;
            auto uniform_fragment = std::make_shared<platform::CompiledGraphicsStage>(*fragment);
            uniform_fragment->entry_point = "uniform_fragment";
            // Hardware constant buffer 3 is NVN uniform slot 0.
            uniform_fragment->constant_buffer_windows.push_back({3, 16, 0});
            auto far_uniform_fragment = std::make_shared<platform::CompiledGraphicsStage>(*fragment);
            far_uniform_fragment->entry_point = "far_uniform_fragment";
            far_uniform_fragment->constant_buffer_windows.push_back({3, 80, 0});
            auto sampled_fragment = std::make_shared<platform::CompiledGraphicsStage>(*fragment);
            sampled_fragment->entry_point = "sampled_fragment";
            sampled_fragment->textures.push_back({0, 0, 0});
            for (unsigned kind = 0; kind != 3; ++kind) {
                auto stage = std::make_shared<platform::CompiledGraphicsStage>(*fragment);
                stage->entry_point = kind == 0 ? "report_buffer_fragment"
                    : kind == 1 ? "report_texture_fragment" : "report_samples_fragment";
                stage->textures.push_back({0, 0, shader::graphics_msl::TextureBinding::no_sampler});
                stages.report_texture_fragments[kind] = stage;
            }
            auto writer_fragment = std::make_shared<platform::CompiledGraphicsStage>(*fragment);
            writer_fragment->entry_point = "writer_fragment";
            writer_fragment->storage_buffers.push_back({0, 0x510, 0,
                platform::CompiledShaderResourceAccess::ReadWrite, 4});
            auto reader_fragment = std::make_shared<platform::CompiledGraphicsStage>(*writer_fragment);
            reader_fragment->entry_point = "reader_fragment";
            reader_fragment->storage_buffers[0].access = platform::CompiledShaderResourceAccess::Read;
            // The texture handle is a word of uniform buffer 0 (hardware slot 3); the tint is
            // uniform buffer 1 (slot 4).
            auto handle_fragment = std::make_shared<platform::CompiledGraphicsStage>(*fragment);
            handle_fragment->entry_point = "handle_fragment";
            platform::CompiledShaderResource handle_resource;
            handle_resource.array_length = handle_resource.location.count = 1;
            handle_resource.cbuf_slot = 3;
            handle_resource.handle_byte_offset = 0;
            handle_fragment->resources.push_back(handle_resource);
            handle_fragment->constant_buffer_windows.push_back({4, 16, 0});
            stages.handle_fragment = handle_fragment;
            // The storage buffer's descriptor is the first twelve bytes of uniform buffer 0.
            auto descriptor_fragment = std::make_shared<platform::CompiledGraphicsStage>(*reader_fragment);
            descriptor_fragment->storage_buffers = {{3, 0, 0, platform::CompiledShaderResourceAccess::Read, 4}};
            stages.descriptor_fragment = descriptor_fragment;
            stages.writer_fragment = writer_fragment;
            stages.reader_fragment = reader_fragment;
            auto writer_vertex = std::make_shared<platform::CompiledGraphicsStage>(*vertex);
            writer_vertex->entry_point = "writer_vertex";
            writer_vertex->storage_buffers.push_back({0, 0x110, 0,
                platform::CompiledShaderResourceAccess::ReadWrite, 4});
            stages.writer_vertex = writer_vertex;
            stages.attribute_vertex = attribute_vertex;
            auto fetch_vertex = std::make_shared<platform::CompiledGraphicsStage>(*vertex);
            fetch_vertex->entry_point = "fetch_vertex";
            fetch_vertex->vertex_class_inputs = {{2, graphics::VertexFormatClass::float_class},
                                                 {5, graphics::VertexFormatClass::float_class}};
            fetch_vertex->vertex_fetch_buffer_index = 0;
            stages.fetch_vertex = fetch_vertex;
            stages.uniform_fragment = uniform_fragment;
            stages.far_uniform_fragment = far_uniform_fragment;
            for (unsigned index = 0; index < stages.mrt_fragments.size(); ++index) {
                auto stage = std::make_shared<platform::CompiledGraphicsStage>(*fragment);
                stage->entry_point = index == 0 ? "mrt_first" : index == 1 ? "mrt_last" : "mrt_both";
                for (unsigned slot = 0; slot < 3; ++slot) {
                    if (slot && index < 2 && slot != index + 1) continue;
                    stage->interface.color_output_mask |= 15u << (slot * 4);
                    stage->color_outputs.push_back({static_cast<std::uint8_t>(slot), 15,
                        platform::ColorOutputScalar::f32, 4});
                }
                stages.mrt_fragments[index] = stage;
            }
            stages.sampled_fragment = sampled_fragment;
            auto mip_fragment = std::make_shared<platform::CompiledGraphicsStage>(*sampled_fragment);
            mip_fragment->entry_point = "mip_fragment";
            stages.mip_fragment = mip_fragment;
            auto depth_reader_fragment = std::make_shared<platform::CompiledGraphicsStage>(*fragment);
            depth_reader_fragment->entry_point = "depth_reader_fragment";
            depth_reader_fragment->textures.push_back({0, 0,
                shader::graphics_msl::TextureBinding::no_sampler});
            stages.depth_reader_fragments[0] = depth_reader_fragment;
            auto reflected_depth_reader = std::make_shared<platform::CompiledGraphicsStage>(*depth_reader_fragment);
            reflected_depth_reader->textures.clear();
            platform::CompiledShaderResource depth_resource;
            depth_resource.array_length = depth_resource.location.count = 1;
            depth_resource.cbuf_slot = 2;
            depth_resource.handle_byte_offset = 0x20;
            reflected_depth_reader->resources.push_back(depth_resource);
            stages.depth_reader_fragments[1] = reflected_depth_reader;
            stages.vertex = vertex;
            stages.fragment = fragment;
            stages.compute = compute;

            select_api_version({55, 15});
            set_program_resolver(Stages::resolve, &stages);
            register_command_entries();
            if (std::getenv("ARMIGO_NVN_STENCIL_CONTRACT_TESTS")) {
                for (const bool write_depth : {false, true})
                    for (const bool stencil_matches : {false, true})
                        for (const bool masked_clear : {false, true})
                            test_inactive_packed_stencil(device, queue, write_depth, stencil_matches, masked_clear);
                for (const bool matches : {false, true}) {
                    test_inactive_packed_stencil(device, queue, false, matches, false, true, false, true);
                    test_inactive_packed_stencil(device, queue, false, matches, false, false, true, true);
                    test_inactive_packed_stencil(device, queue, false, matches, false, false, false, true, true);
                }
                set_program_resolver(nullptr, nullptr);
                std::fprintf(stderr, "Metal stencil contract failures: %u\n", failures);
                return failures ? 1 : 0;
            }
            if (std::getenv("ARMIGO_NVN_MRT_PROBE_ONLY")) {
                test_single_writer_mrt_alias(device, queue);
                test_aliased_mrt_writer_choice(device, queue);
                set_program_resolver(nullptr, nullptr);
                std::fprintf(stderr, "Metal MRT probe failures: %u\n", failures);
                return failures ? 1 : 0;
            }
            if (std::getenv("ARMIGO_NVN_BATCHED_QUERY_CONTRACT_TESTS")) {
                test_predicate_ordering_stress(device, queue, 8);
                test_predicate_ordering_stress(device, queue, 8, true);
                test_predicate_ordering_stress(device, queue, 8, true, true);
                set_program_resolver(nullptr, nullptr);
                std::fprintf(stderr, "Metal batched query contract failures: %u\n", failures);
                return failures ? 1 : 0;
            }
            if (std::getenv("ARMIGO_NVN_CONDITIONAL_CONTRACT_TESTS")) {
                for (const auto which : {ConditionalContract::ordinary, ConditionalContract::occluded,
                        ConditionalContract::report_in_block, ConditionalContract::report_after_block,
                        ConditionalContract::storage_write, ConditionalContract::nonzero_reference,
                        ConditionalContract::two_reports, ConditionalContract::pending_reference_copy,
                        ConditionalContract::stencil_depth_fail, ConditionalContract::depth_write})
                    for (const bool visible : {true, false})
                        test_gpu_conditional_render(device, queue, visible, 3, which);
                std::fprintf(stderr, "Metal conditional contract failures: %u\n", failures);
                // Also cover equality, reused addresses, cross-submission
                // predicates, and the GPU ordering stress below.
            }
            test_segmented_uniforms_and_effects(device, queue);
            test_stencil_clear_without_stencil(device, queue);
            test_depth_bounds_draw(device, queue);
            test_counters_inside_pass(device, queue);
            for (const auto which : {ConditionalCase::if_equal, ConditionalCase::clear_in_block,
                                     ConditionalCase::reused_address, ConditionalCase::across_submissions,
                                     ConditionalCase::stale_after_drain, ConditionalCase::copy_after_condition})
                for (const bool visible : {true, false}) test_conditional_case(device, queue, which, visible);
            test_predicate_ordering_stress(device, queue, 8);
            test_predicate_ordering_stress(device, queue, 8, true);
            test_predicate_ordering_stress(device, queue, 8, true, true);
            test_gpu_conditional_render(device, queue, true, 3);
            test_gpu_conditional_render(device, queue, false, 3);
            test_gpu_conditional_render(device, queue, true, 130);
            test_gpu_conditional_render(device, queue, false, 130);
            test_vertex_only_depth_draw(device, queue);
            for (const bool write_depth : {false, true})
                for (const bool stencil_matches : {false, true})
                    for (const bool masked_clear : {false, true})
                        test_inactive_packed_stencil(device, queue, write_depth, stencil_matches, masked_clear);
            for (const bool matches : {false, true}) {
                test_inactive_packed_stencil(device, queue, false, matches, false, true, false, true);
                test_inactive_packed_stencil(device, queue, false, matches, false, false, true, true);
                test_inactive_packed_stencil(device, queue, false, matches, false, false, false, true, true);
            }
            test_clear_pipeline_record(device, queue);
            test_pipeline_archive(device, queue);
            test_failed_pending_stage(device, queue, stages);
            test_segmented_failed_prefix(device, queue);
            test_depth_barrier_continuation(device, queue, false);
            test_depth_barrier_continuation(device, queue, true);
            test_depth_barrier_continuation(device, queue, true, true);
            test_depth_barrier_continuation(device, queue, false, false, true);
            test_depth_barrier_continuation(device, queue, false, false, false, true);
            test_binding_generations(device, queue);
            test_draw_write_order_and_submission_bindings(device, queue);
            test_submission_slot_reuse(device, queue);
            test_report_vertex_alias(device, queue);
            test_report_attachment_alias(device, queue);
            for (unsigned mode = 0; mode != 5; ++mode) test_report_isolation_probe(device, queue, mode);
            test_reused_report_destination(device, queue, false);
            test_reused_report_destination(device, queue, true);
            for (unsigned kind = 0; kind != 3; ++kind) {
                test_report_texture_alias(device, queue, kind, false, false, true);
                test_report_texture_alias(device, queue, kind, true, false, true);
                test_report_texture_alias(device, queue, kind, true, true, true);
                test_report_texture_alias(device, queue, kind, true, false, false);
                if (kind != 0) {
                    test_report_texture_alias(device, queue, kind, true, false, true, true);
                    test_report_texture_alias(device, queue, kind, true, false, false, true);
                }
            }
            test_report_texture_alias(device, queue, 0, true, false, true, false, true);
            test_report_texture_alias(device, queue, 0, true, false, false, false, true);
            test_shutdown_waits_for_feedback_logging_tail(device, queue);
            test_submission_shutdown_joins_and_retries(device, queue);
            test_submission_scheduling_period(device, queue);
            test_submission_encode_wait_budget(device, queue);
            test_scheduling_failures_and_encode_policy(device, queue);
            test_guest_present_scheduling(device, queue);
            test_upload_retirement(device, queue);
            test_failed_upload_attribution(device, queue);
            test_uniform_versions_and_gpu_indices(device, queue);
            test_uniform_fill_after_gpu_wait(device, queue);
            test_overlapping_uniform_update_order(device, queue);
            test_uniform_prefix_growth(device, queue);
            test_constant_window_bound(device, queue);
            test_constant_window_after_gpu_write(device, queue);
            test_single_writer_mrt_alias(device, queue);
            test_aliased_mrt_writer_choice(device, queue);
            test_cube_clear(device, queue);
            test_bgra8_clear_copy(device, queue);
            test_clear_draw_copy_dispatch(device, queue);
            test_dispatch_hazards(device, queue);
            test_state_and_deferred_bindings(device, queue);
            test_scaled_vertex_format(device, queue);
            test_vertex_fetch_draw(device, queue);
            test_vertex_fetch_misaligned(device, queue);
            test_vertex_input_bits(device, queue, false);
            test_vertex_input_bits(device, queue, true);
            test_mip_chain_levels(device, queue);
            test_uniform_only_rebind(device, queue);
            test_sampler_registration_order(device, queue);
            test_sampler_draw_order(device, queue);
            for (const bool level_one : {false, true}) {
                test_mip_alias_reopened_pass(device, queue, level_one);
                test_mip_alias_shared_subresource(device, queue, level_one);
            }
            test_resolve_and_partial_clear(device, queue);
            set_program_resolver(nullptr, nullptr);
        }
    }
    std::printf("Metal NVN replay failures: %u\n", failures);
    return failures ? 1 : 0;
}
