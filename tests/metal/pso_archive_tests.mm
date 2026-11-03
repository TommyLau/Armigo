// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#import <Metal/Metal.h>

#include "pso_archive.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {

using namespace armigo::platform::metal::nvn_driver;

int failures = 0;
#define CHECK(condition) do { if (!(condition)) { ++failures; std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #condition); } } while (0)

MTL4RenderPipelineDescriptor* descriptor(id<MTLDevice> device, int salt) {
    NSString* source = [NSString stringWithFormat:
        @"#include <metal_stdlib>\nusing namespace metal;\n"
         "vertex float4 vs(uint i [[vertex_id]]) { return float4(float(i) * %d.0, 0, 0, 1); }\n"
         "fragment float4 fs() { return float4(%d.5, 0, 0, 1); }\n", salt + 1, salt];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
    auto* vertex = [MTL4LibraryFunctionDescriptor new];
    vertex.library = library;
    vertex.name = @"vs";
    auto* fragment = [MTL4LibraryFunctionDescriptor new];
    fragment.library = library;
    fragment.name = @"fs";
    auto* result = [MTL4RenderPipelineDescriptor new];
    result.vertexFunctionDescriptor = vertex;
    result.fragmentFunctionDescriptor = fragment;
    result.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    return result;
}

} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { std::printf("no Metal device\n"); return 77; }
        const auto dir = std::filesystem::temp_directory_path() / ("pso_archive_tests." + std::to_string(getpid()));
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);

        // An empty directory, or a missing one, is simply no archives.
        CHECK(open_archives(device, dir).archives.count == 0);
        CHECK(open_archives(device, dir / "missing").archives.count == 0);

        // Write a real archive holding one pipeline.
        NSError* error = nil;
        auto* serializer_descriptor = [MTL4PipelineDataSetSerializerDescriptor new];
        serializer_descriptor.configuration = MTL4PipelineDataSetSerializerConfigurationCaptureBinaries;
        id<MTL4PipelineDataSetSerializer> serializer =
            [device newPipelineDataSetSerializerWithDescriptor:serializer_descriptor];
        auto* compiler_descriptor = [MTL4CompilerDescriptor new];
        compiler_descriptor.pipelineDataSetSerializer = serializer;
        id<MTL4Compiler> compiler = [device newCompilerWithDescriptor:compiler_descriptor error:&error];
        CHECK(compiler != nil);
        CHECK([compiler newRenderPipelineStateWithDescriptor:descriptor(device, 41)
                                         compilerTaskOptions:nil error:&error] != nil);
        const auto written = dir / "pipelines-1.partial";
        CHECK([serializer serializeAsArchiveAndFlushToURL:[NSURL fileURLWithPath:@(written.c_str())] error:&error]);
        // A file still being written is not an archive yet.
        CHECK(open_archives(device, dir).archives.count == 0);
        const auto good = seal_archive(written, "pipelines-1");
        CHECK(!good.empty() && good.extension() == ".marchive" && archive_intact(good));

        // Damaged and foreign files are skipped, the good archive still loads.
        // (Metal itself crashes on some of these, so none reaches it.)
        { std::ofstream(dir / "pipelines-2.marchive") << "not an archive"; }
        { std::ofstream(dir / "pipelines-3.marchive", std::ios::binary) << std::string(4096, '\0'); }
        { std::ofstream(dir / "notes.txt") << "unrelated"; }
        {
            // The good archive with its first bytes damaged keeps its old name.
            std::filesystem::copy_file(good, dir / ("pipelines-4-" + good.stem().string().substr(good.stem().string().rfind('-') + 1) + ".marchive"));
            std::fstream damaged(dir / ("pipelines-4-" + good.stem().string().substr(good.stem().string().rfind('-') + 1) + ".marchive"),
                                 std::ios::in | std::ios::out | std::ios::binary);
            damaged.write("ABCD", 4);
        }
        const auto set = open_archives(device, dir);
        CHECK(set.archives.count == 1);
        CHECK(set.loaded.size() == 1 && set.loaded[0] == good);
        CHECK(set.rejected.size() == 3);

        // A pipeline the archive holds comes from it; one it lacks (another
        // shader, as after a package or OS change) falls back to compiling and
        // still succeeds.
        auto* options = [MTL4CompilerTaskOptions new];
        options.lookupArchives = set.archives;
        id<MTL4Compiler> reader = [device newCompilerWithDescriptor:[MTL4CompilerDescriptor new] error:&error];
        CHECK([reader newRenderPipelineStateWithDescriptor:descriptor(device, 41)
                                       compilerTaskOptions:options error:&error] != nil);
        CHECK([reader newRenderPipelineStateWithDescriptor:descriptor(device, 4242)
                                       compilerTaskOptions:options error:&error] != nil);

        // Looked up directly, the archive returns the pipelines it holds and
        // nothing else, so a miss is known without compiling.
        id<MTL4Archive> archive = set.archives[0];
        CHECK([archive newRenderPipelineStateWithDescriptor:descriptor(device, 41) error:&error] != nil);
        CHECK([archive newRenderPipelineStateWithDescriptor:descriptor(device, 4242) error:&error] == nil);

        // A pipeline with its color attachment left unspecialized (format,
        // blending, write mask) is archived once and specialized for each
        // attachment configuration a draw uses.
        auto* open_state = descriptor(device, 7);
        for (NSUInteger index = 0; index != 8; ++index) {
            open_state.colorAttachments[index].pixelFormat = MTLPixelFormatUnspecialized;
            open_state.colorAttachments[index].writeMask = MTLColorWriteMaskUnspecialized;
            open_state.colorAttachments[index].blendingState = MTL4BlendStateUnspecialized;
        }
        CHECK([compiler newRenderPipelineStateWithDescriptor:open_state compilerTaskOptions:nil error:&error] != nil);
        const auto open_written = dir / "pipelines-5.partial";
        CHECK([serializer serializeAsArchiveAndFlushToURL:[NSURL fileURLWithPath:@(open_written.c_str())]
                                                    error:&error]);
        const auto open_sealed = seal_archive(open_written, "pipelines-5");
        id<MTL4Archive> open_archive = [device newArchiveWithURL:[NSURL fileURLWithPath:@(open_sealed.c_str())]
                                                           error:&error];
        CHECK(open_archive != nil);
        id<MTLRenderPipelineState> unspecialized =
            [open_archive newRenderPipelineStateWithDescriptor:open_state error:&error];
        CHECK(unspecialized != nil);
        auto* specialized = descriptor(device, 7);
        specialized.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA16Float;
        specialized.colorAttachments[0].writeMask = MTLColorWriteMaskRed | MTLColorWriteMaskAlpha;
        specialized.colorAttachments[0].blendingState = MTL4BlendStateEnabled;
        specialized.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        specialized.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        const auto specialize_begin = std::chrono::steady_clock::now();
        CHECK(unspecialized && [reader newRenderPipelineStateBySpecializationWithDescriptor:specialized
                                                                                   pipeline:unspecialized
                                                                                      error:&error] != nil);
        std::printf("specialization %.3f ms\n", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - specialize_begin).count());
        std::filesystem::remove(open_sealed);

        // Nothing but damaged files: no archives, so compiles carry no lookup list.
        std::filesystem::remove(good);
        CHECK(open_archives(device, dir).archives.count == 0);
        std::filesystem::remove_all(dir);
    }
    std::printf("pso archive failures: %d\n", failures);
    return failures ? 1 : 0;
}
