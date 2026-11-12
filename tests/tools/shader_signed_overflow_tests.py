#!/usr/bin/env python3
# ============================================================================
# Armigo Tests - Native Runtime Verification
# Copyright (c) 2026 Tommy Lau.
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
#
# This file is licensed under GPL-3.0-only or, by separate written
# agreement, an Armigo proprietary license.
# See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
# ============================================================================

"""Execute synthetic wrapping arithmetic with each production compile policy."""
import ast
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BANNER = """// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
// This file is licensed under GPL-3.0-only or, by separate written agreement,
// an Armigo proprietary license. See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
"""
KERNEL = BANNER + """#include <metal_stdlib>
using namespace metal;
kernel void wrap_values(device const uint* words [[buffer(0)]],
                        device uint* results [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    int value = as_type<int>(words[index]);
    int sum = value + 11;
    int difference = value - 23;
    int product = value * 5;
    results[index * 6] = as_type<uint>(sum);
    results[index * 6 + 1] = as_type<uint>(difference);
    results[index * 6 + 2] = as_type<uint>(product);
    results[index * 6 + 3] = sum < value ? 1u : 0u;
    results[index * 6 + 4] = as_type<uint>((value * 8) / 8);
    results[index * 6 + 5] = as_type<uint>((value * 4096) >> 12);
}
"""
RUNNER = BANNER + r'''#import <Metal/Metal.h>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
int main(int argc, char** argv) { @autoreleasepool {
    if (argc != 2) return 1;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 2;
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    id<MTLComputePipelineState> pipeline = library ? [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"wrap_values"] error:&error] : nil;
    if (!pipeline) { std::fprintf(stderr, "%s\n", error.localizedDescription.UTF8String); return 1; }
    constexpr unsigned count = 16384;
    id<MTLBuffer> input = [device newBufferWithLength:count * 4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:count * 24 options:MTLResourceStorageModeShared];
    if (!input || !output) return 1;
    auto* words = static_cast<std::uint32_t*>(input.contents);
    constexpr std::array<std::uint32_t, 13> edges{0, 1, 0xffffffffu, 0x7fffffffu, 0x80000000u,
        0x7ffffff4u, 0x7ffffff5u, 0x80000016u, 0x80000017u, 0x1234abcdu, 0x40000000u, 0xc0000000u, 0x80000u};
    std::uint32_t state = 0x615a7c23u;
    for (unsigned i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        words[i] = i < edges.size() ? edges[i] : state;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0]; [encoder setBuffer:output offset:0 atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(count / 64, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding]; [commands commit]; [commands waitUntilCompleted];
    if (commands.status == MTLCommandBufferStatusError) return 1;
    const auto* actual = static_cast<const std::uint32_t*>(output.contents);
    for (unsigned i = 0; i < count; ++i) {
        const auto raw = words[i];
        const auto value = std::bit_cast<std::int32_t>(raw);
        const auto wrapped_sum = std::bit_cast<std::int32_t>(raw + 11u);
        const std::array<std::uint32_t, 6> expected{raw + 11u, raw - 23u, raw * 5u,
            wrapped_sum < value ? 1u : 0u,
            static_cast<std::uint32_t>(std::bit_cast<std::int32_t>(raw * 8u) / 8),
            static_cast<std::uint32_t>(std::bit_cast<std::int32_t>(raw * 4096u) >> 12)};
        for (unsigned component = 0; component < expected.size(); ++component)
            if (actual[i * 6 + component] != expected[component]) {
                std::fprintf(stderr, "Wrapping arithmetic mismatch at input %08x component %u: %08x expected %08x\n",
                             raw, component, actual[i * 6 + component], expected[component]);
                return 1;
            }
    }
    std::printf("Checked %u wrapping arithmetic words\n", count * 6);
    return 0;
} }
'''


def python_flags():
    tree = ast.parse((ROOT / "src/shader/tools/build_package.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "METAL_COMPILE_FLAGS" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise AssertionError("Python package compile policy not found")


def cpp_flags(relative):
    text = (ROOT / relative).read_text()
    match = re.search(r"\bcompile_flags\s*\{([^}]+)\}", text)
    if not match:
        raise AssertionError("C++ package compile policy not found")
    return re.findall(r'"([^"\n]+)"', match[1])


class SignedOverflow(unittest.TestCase):
    def test_production_compiler_policies(self):
        with tempfile.TemporaryDirectory(prefix="armigo-wrap-") as temporary:
            root = Path(temporary)
            source, runner_source = root / "values.metal", root / "runner.mm"
            source.write_text(KERNEL); runner_source.write_text(RUNNER)
            runner = root / "runner"
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O2", "-fobjc-arc", str(runner_source),
                            "-framework", "Foundation", "-framework", "Metal", "-o", str(runner)], check=True, capture_output=True)
            routes = [("python-macos", "macosx", python_flags()),
                      ("cpp-macos", "macosx", cpp_flags("src/shader/package.cpp")),
                      ("native-macos", "macosx", cpp_flags("src/pack/cli/native_shaders.cpp")),
                      ("cpp-ios", "iphoneos", cpp_flags("src/shader/package_ios.cpp"))]
            for name, sdk, flags in routes:
                with self.subTest(route=name):
                    air, library = root / (name + ".air"), root / (name + ".metallib")
                    subprocess.run(["xcrun", "--sdk", sdk, "metal", *flags, "-c", str(source), "-o", str(air)],
                                   check=True, capture_output=True)
                    if sdk == "macosx":
                        subprocess.run(["xcrun", "--sdk", sdk, "metallib", str(air), "-o", str(library)],
                                       check=True, capture_output=True)
                        result = subprocess.run([str(runner), str(library)], capture_output=True, text=True)
                        if result.returncode == 2: self.skipTest("No Metal device")
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("-fwrapv", flags)


if __name__ == "__main__":
    unittest.main()
