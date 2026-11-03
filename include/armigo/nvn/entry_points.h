// ============================================================================
// Armigo NVN - API Entry Points
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// NVN API entry points by name, with their signatures over the types of
// api_types.h. ARMIGO_NVN_ENTRY_POINTS(X) expands X(name, return type, (parameters)) for
// every entry point with a known signature; ARMIGO_NVN_UNTYPED_ENTRY_POINTS(X)
// expands X(name) for newer entry points whose signature is not described
// yet. Booleans are passed as NVNboolean; callers set only the low 8 bits.

#pragma once

#include <armigo/nvn/api_types.h>

// clang-format off
#define ARMIGO_NVN_ENTRY_POINTS(X) \
    X(nvnDeviceBuilderSetDefaults, void, (NVNdeviceBuilder*)) \
    X(nvnDeviceBuilderGetFlags, int, (const NVNdeviceBuilder*)) \
    X(nvnDeviceBuilderSetFlags, void, (NVNdeviceBuilder*, int)) \
    X(nvnDeviceInitialize, NVNboolean, (NVNdevice*, const NVNdeviceBuilder*)) \
    X(nvnDeviceFinalize, void, (NVNdevice*)) \
    X(nvnDeviceSetDebugLabel, void, (NVNdevice*, const char*)) \
    X(nvnDeviceGetProcAddress, PFNNVNGENERICFUNCPTRPROC, (const NVNdevice*, const char*)) \
    X(nvnDeviceGetInteger, void, (const NVNdevice*, NVNdeviceInfo, int*)) \
    X(nvnDeviceGetCurrentTimestampInNanoseconds, std::uint64_t, (const NVNdevice*)) \
    X(nvnDeviceSetIntermediateShaderCache, void, (NVNdevice*, int)) \
    X(nvnDeviceGetTextureHandle, NVNtextureHandle, (const NVNdevice*, int, int)) \
    X(nvnDeviceGetTexelFetchHandle, NVNtextureHandle, (const NVNdevice*, int)) \
    X(nvnDeviceGetImageHandle, NVNimageHandle, (const NVNdevice*, int)) \
    X(nvnDeviceInstallDebugCallback, void, (NVNdevice*, const PFNNVNDEBUGCALLBACKPROC, void*, std::uint32_t)) \
    X(nvnDeviceGenerateDebugDomainId, NVNdebugDomainId, (const NVNdevice*, const char*)) \
    X(nvnDeviceSetWindowOriginMode, void, (NVNdevice*, NVNwindowOriginMode)) \
    X(nvnDeviceSetDepthMode, void, (NVNdevice*, NVNdepthMode)) \
    X(nvnDeviceRegisterFastClearColor, NVNboolean, (NVNdevice*, const float*, NVNformat)) \
    X(nvnDeviceRegisterFastClearColori, NVNboolean, (NVNdevice*, const int*, NVNformat)) \
    X(nvnDeviceRegisterFastClearColorui, NVNboolean, (NVNdevice*, const std::uint32_t*, NVNformat)) \
    X(nvnDeviceRegisterFastClearDepth, NVNboolean, (NVNdevice*, float)) \
    X(nvnDeviceGetWindowOriginMode, NVNwindowOriginMode, (const NVNdevice*)) \
    X(nvnDeviceGetDepthMode, NVNdepthMode, (const NVNdevice*)) \
    X(nvnDeviceGetTimestampInNanoseconds, std::uint64_t, (const NVNdevice*, const NVNcounterData*)) \
    X(nvnDeviceApplyDeferredFinalizes, void, (NVNdevice*, int)) \
    X(nvnDeviceFinalizeCommandHandle, void, (NVNdevice*, NVNcommandHandle)) \
    X(nvnDeviceWalkDebugDatabase, void, (const NVNdevice*, NVNdebugObjectType, PFNNVNWALKDEBUGDATABASECALLBACKPROC, void*)) \
    X(nvnDeviceGetSeparateTextureHandle, NVNseparateTextureHandle, (const NVNdevice*, int)) \
    X(nvnDeviceGetSeparateSamplerHandle, NVNseparateSamplerHandle, (const NVNdevice*, int)) \
    X(nvnDeviceIsExternalDebuggerAttached, NVNboolean, (const NVNdevice*)) \
    X(nvnQueueGetError, NVNqueueGetErrorResult, (NVNqueue*, NVNqueueErrorInfo*)) \
    X(nvnQueueGetTotalCommandMemoryUsed, std::size_t, (NVNqueue*)) \
    X(nvnQueueGetTotalControlMemoryUsed, std::size_t, (NVNqueue*)) \
    X(nvnQueueGetTotalComputeMemoryUsed, std::size_t, (NVNqueue*)) \
    X(nvnQueueResetMemoryUsageCounts, void, (NVNqueue*)) \
    X(nvnQueueBuilderSetDevice, void, (NVNqueueBuilder*, NVNdevice*)) \
    X(nvnQueueBuilderSetDefaults, void, (NVNqueueBuilder*)) \
    X(nvnQueueBuilderSetFlags, void, (NVNqueueBuilder*, int)) \
    X(nvnQueueBuilderSetCommandMemorySize, void, (NVNqueueBuilder*, std::size_t)) \
    X(nvnQueueBuilderSetComputeMemorySize, void, (NVNqueueBuilder*, std::size_t)) \
    X(nvnQueueBuilderSetControlMemorySize, void, (NVNqueueBuilder*, std::size_t)) \
    X(nvnQueueBuilderGetQueueMemorySize, std::size_t, (const NVNqueueBuilder*)) \
    X(nvnQueueBuilderSetQueueMemory, void, (NVNqueueBuilder*, void*, std::size_t)) \
    X(nvnQueueBuilderSetCommandFlushThreshold, void, (NVNqueueBuilder*, std::size_t)) \
    X(nvnQueueInitialize, NVNboolean, (NVNqueue*, const NVNqueueBuilder*)) \
    X(nvnQueueFinalize, void, (NVNqueue*)) \
    X(nvnQueueSetDebugLabel, void, (NVNqueue*, const char*)) \
    X(nvnQueueSubmitCommands, void, (NVNqueue*, int, const NVNcommandHandle*)) \
    X(nvnQueueFlush, void, (NVNqueue*)) \
    X(nvnQueueFinish, void, (NVNqueue*)) \
    X(nvnQueuePresentTexture, void, (NVNqueue*, NVNwindow*, int)) \
    X(nvnQueueAcquireTexture, NVNqueueAcquireTextureResult, (NVNqueue*, NVNwindow*, int*)) \
    X(nvnWindowBuilderSetDevice, void, (NVNwindowBuilder*, NVNdevice*)) \
    X(nvnWindowBuilderSetDefaults, void, (NVNwindowBuilder*)) \
    X(nvnWindowBuilderSetNativeWindow, void, (NVNwindowBuilder*, NVNnativeWindow)) \
    X(nvnWindowBuilderSetTextures, void, (NVNwindowBuilder*, int, NVNtexture* const*)) \
    X(nvnWindowBuilderSetPresentInterval, void, (NVNwindowBuilder*, int)) \
    X(nvnWindowBuilderGetNativeWindow, NVNnativeWindow, (const NVNwindowBuilder*)) \
    X(nvnWindowBuilderGetPresentInterval, int, (const NVNwindowBuilder*)) \
    X(nvnWindowInitialize, NVNboolean, (NVNwindow*, const NVNwindowBuilder*)) \
    X(nvnWindowFinalize, void, (NVNwindow*)) \
    X(nvnWindowSetDebugLabel, void, (NVNwindow*, const char*)) \
    X(nvnWindowAcquireTexture, NVNwindowAcquireTextureResult, (NVNwindow*, NVNsync*, int*)) \
    X(nvnWindowGetNativeWindow, NVNnativeWindow, (const NVNwindow*)) \
    X(nvnWindowGetPresentInterval, int, (const NVNwindow*)) \
    X(nvnWindowSetPresentInterval, void, (NVNwindow*, int)) \
    X(nvnWindowSetCrop, void, (NVNwindow*, int, int, int, int)) \
    X(nvnWindowGetCrop, void, (const NVNwindow*, NVNrectangle*)) \
    X(nvnProgramInitialize, NVNboolean, (NVNprogram*, NVNdevice*)) \
    X(nvnProgramFinalize, void, (NVNprogram*)) \
    X(nvnProgramSetDebugLabel, void, (NVNprogram*, const char*)) \
    X(nvnProgramSetShaders, NVNboolean, (NVNprogram*, int, const NVNshaderData*)) \
    X(nvnProgramSetShadersExt, NVNboolean, (NVNprogram*, int, const NVNshaderDataExt*)) \
    X(nvnMemoryPoolBuilderSetDevice, void, (NVNmemoryPoolBuilder*, NVNdevice*)) \
    X(nvnMemoryPoolBuilderSetDefaults, void, (NVNmemoryPoolBuilder*)) \
    X(nvnMemoryPoolBuilderGetDevice, NVNdevice*, (const NVNmemoryPoolBuilder*)) \
    X(nvnMemoryPoolBuilderSetStorage, void, (NVNmemoryPoolBuilder*, void*, std::size_t)) \
    X(nvnMemoryPoolBuilderSetFlags, void, (NVNmemoryPoolBuilder*, int)) \
    X(nvnMemoryPoolBuilderGetMemory, void*, (const NVNmemoryPoolBuilder*)) \
    X(nvnMemoryPoolBuilderGetSize, std::size_t, (const NVNmemoryPoolBuilder*)) \
    X(nvnMemoryPoolBuilderGetFlags, NVNmemoryPoolFlags, (const NVNmemoryPoolBuilder*)) \
    X(nvnMemoryPoolInitialize, NVNboolean, (NVNmemoryPool*, const NVNmemoryPoolBuilder*)) \
    X(nvnMemoryPoolSetDebugLabel, void, (NVNmemoryPool*, const char*)) \
    X(nvnMemoryPoolFinalize, void, (NVNmemoryPool*)) \
    X(nvnMemoryPoolMap, void*, (const NVNmemoryPool*)) \
    X(nvnMemoryPoolFlushMappedRange, void, (const NVNmemoryPool*, ptrdiff_t, std::size_t)) \
    X(nvnMemoryPoolInvalidateMappedRange, void, (const NVNmemoryPool*, ptrdiff_t, std::size_t)) \
    X(nvnMemoryPoolGetBufferAddress, NVNbufferAddress, (const NVNmemoryPool*)) \
    X(nvnMemoryPoolMapVirtual, NVNboolean, (NVNmemoryPool*, int, const NVNmappingRequest*)) \
    X(nvnMemoryPoolGetSize, std::size_t, (const NVNmemoryPool*)) \
    X(nvnMemoryPoolGetFlags, NVNmemoryPoolFlags, (const NVNmemoryPool*)) \
    X(nvnTexturePoolInitialize, NVNboolean, (NVNtexturePool*, const NVNmemoryPool*, ptrdiff_t, int)) \
    X(nvnTexturePoolSetDebugLabel, void, (NVNtexturePool*, const char*)) \
    X(nvnTexturePoolFinalize, void, (NVNtexturePool*)) \
    X(nvnTexturePoolRegisterTexture, void, (const NVNtexturePool*, int, const NVNtexture*, const NVNtextureView*)) \
    X(nvnTexturePoolRegisterImage, void, (const NVNtexturePool*, int, const NVNtexture*, const NVNtextureView*)) \
    X(nvnTexturePoolGetMemoryPool, const NVNmemoryPool*, (const NVNtexturePool*)) \
    X(nvnTexturePoolGetMemoryOffset, ptrdiff_t, (const NVNtexturePool*)) \
    X(nvnTexturePoolGetSize, int, (const NVNtexturePool*)) \
    X(nvnSamplerPoolInitialize, NVNboolean, (NVNsamplerPool*, const NVNmemoryPool*, ptrdiff_t, int)) \
    X(nvnSamplerPoolSetDebugLabel, void, (NVNsamplerPool*, const char*)) \
    X(nvnSamplerPoolFinalize, void, (NVNsamplerPool*)) \
    X(nvnSamplerPoolRegisterSampler, void, (const NVNsamplerPool*, int, const NVNsampler*)) \
    X(nvnSamplerPoolRegisterSamplerBuilder, void, (const NVNsamplerPool*, int, const NVNsamplerBuilder*)) \
    X(nvnSamplerPoolGetMemoryPool, const NVNmemoryPool*, (const NVNsamplerPool*)) \
    X(nvnSamplerPoolGetMemoryOffset, ptrdiff_t, (const NVNsamplerPool*)) \
    X(nvnSamplerPoolGetSize, int, (const NVNsamplerPool*)) \
    X(nvnBufferBuilderSetDevice, void, (NVNbufferBuilder*, NVNdevice*)) \
    X(nvnBufferBuilderSetDefaults, void, (NVNbufferBuilder*)) \
    X(nvnBufferBuilderSetStorage, void, (NVNbufferBuilder*, NVNmemoryPool*, ptrdiff_t, std::size_t)) \
    X(nvnBufferBuilderGetDevice, NVNdevice*, (const NVNbufferBuilder*)) \
    X(nvnBufferBuilderGetMemoryPool, NVNmemoryPool*, (const NVNbufferBuilder*)) \
    X(nvnBufferBuilderGetMemoryOffset, ptrdiff_t, (const NVNbufferBuilder*)) \
    X(nvnBufferBuilderGetSize, std::size_t, (const NVNbufferBuilder*)) \
    X(nvnBufferInitialize, NVNboolean, (NVNbuffer*, const NVNbufferBuilder*)) \
    X(nvnBufferSetDebugLabel, void, (NVNbuffer*, const char*)) \
    X(nvnBufferFinalize, void, (NVNbuffer*)) \
    X(nvnBufferMap, void*, (const NVNbuffer*)) \
    X(nvnBufferGetAddress, NVNbufferAddress, (const NVNbuffer*)) \
    X(nvnBufferFlushMappedRange, void, (const NVNbuffer*, ptrdiff_t, std::size_t)) \
    X(nvnBufferInvalidateMappedRange, void, (const NVNbuffer*, ptrdiff_t, std::size_t)) \
    X(nvnBufferGetMemoryPool, NVNmemoryPool*, (const NVNbuffer*)) \
    X(nvnBufferGetMemoryOffset, ptrdiff_t, (const NVNbuffer*)) \
    X(nvnBufferGetSize, std::size_t, (const NVNbuffer*)) \
    X(nvnBufferGetDebugID, std::uint64_t, (const NVNbuffer*)) \
    X(nvnTextureBuilderSetDevice, void, (NVNtextureBuilder*, NVNdevice*)) \
    X(nvnTextureBuilderSetDefaults, void, (NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetDevice, NVNdevice*, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderSetFlags, void, (NVNtextureBuilder*, int)) \
    X(nvnTextureBuilderSetTarget, void, (NVNtextureBuilder*, NVNtextureTarget)) \
    X(nvnTextureBuilderSetWidth, void, (NVNtextureBuilder*, int)) \
    X(nvnTextureBuilderSetHeight, void, (NVNtextureBuilder*, int)) \
    X(nvnTextureBuilderSetDepth, void, (NVNtextureBuilder*, int)) \
    X(nvnTextureBuilderSetSize1D, void, (NVNtextureBuilder*, int)) \
    X(nvnTextureBuilderSetSize2D, void, (NVNtextureBuilder*, int, int)) \
    X(nvnTextureBuilderSetSize3D, void, (NVNtextureBuilder*, int, int, int)) \
    X(nvnTextureBuilderSetLevels, void, (NVNtextureBuilder*, int)) \
    X(nvnTextureBuilderSetFormat, void, (NVNtextureBuilder*, NVNformat)) \
    X(nvnTextureBuilderSetSamples, void, (NVNtextureBuilder*, int)) \
    X(nvnTextureBuilderSetSwizzle, void, (NVNtextureBuilder*, NVNtextureSwizzle, NVNtextureSwizzle, NVNtextureSwizzle, NVNtextureSwizzle)) \
    X(nvnTextureBuilderSetDepthStencilMode, void, (NVNtextureBuilder*, NVNtextureDepthStencilMode)) \
    X(nvnTextureBuilderGetStorageSize, std::size_t, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetStorageAlignment, std::size_t, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderSetStorage, void, (NVNtextureBuilder*, NVNmemoryPool*, ptrdiff_t)) \
    X(nvnTextureBuilderSetPackagedTextureData, void, (NVNtextureBuilder*, const void*)) \
    X(nvnTextureBuilderSetPackagedTextureLayout, void, (NVNtextureBuilder*, const NVNpackagedTextureLayout*)) \
    X(nvnTextureBuilderSetStride, void, (NVNtextureBuilder*, ptrdiff_t)) \
    X(nvnTextureBuilderSetGLTextureName, void, (NVNtextureBuilder*, std::uint32_t)) \
    X(nvnTextureBuilderGetStorageClass, NVNstorageClass, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetFlags, NVNtextureFlags, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetTarget, NVNtextureTarget, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetWidth, int, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetHeight, int, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetDepth, int, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetLevels, int, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetFormat, NVNformat, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetSamples, int, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetSwizzle, void, (const NVNtextureBuilder*, NVNtextureSwizzle*, NVNtextureSwizzle*, NVNtextureSwizzle*, NVNtextureSwizzle*)) \
    X(nvnTextureBuilderGetDepthStencilMode, NVNtextureDepthStencilMode, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetPackagedTextureData, const void*, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetStride, ptrdiff_t, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetSparseTileLayout, void, (const NVNtextureBuilder*, NVNtextureSparseTileLayout*)) \
    X(nvnTextureBuilderGetGLTextureName, std::uint32_t, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetZCullStorageSize, std::size_t, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetMemoryPool, NVNmemoryPool*, (const NVNtextureBuilder*)) \
    X(nvnTextureBuilderGetMemoryOffset, ptrdiff_t, (const NVNtextureBuilder*)) \
    X(nvnTextureViewSetDefaults, void, (NVNtextureView*)) \
    X(nvnTextureViewSetLevels, void, (NVNtextureView*, int, int)) \
    X(nvnTextureViewSetLayers, void, (NVNtextureView*, int, int)) \
    X(nvnTextureViewSetFormat, void, (NVNtextureView*, NVNformat)) \
    X(nvnTextureViewSetSwizzle, void, (NVNtextureView*, NVNtextureSwizzle, NVNtextureSwizzle, NVNtextureSwizzle, NVNtextureSwizzle)) \
    X(nvnTextureViewSetDepthStencilMode, void, (NVNtextureView*, NVNtextureDepthStencilMode)) \
    X(nvnTextureViewSetTarget, void, (NVNtextureView*, NVNtextureTarget)) \
    X(nvnTextureViewGetLevels, NVNboolean, (const NVNtextureView*, int*, int*)) \
    X(nvnTextureViewGetLayers, NVNboolean, (const NVNtextureView*, int*, int*)) \
    X(nvnTextureViewGetFormat, NVNboolean, (const NVNtextureView*, NVNformat*)) \
    X(nvnTextureViewGetSwizzle, NVNboolean, (const NVNtextureView*, NVNtextureSwizzle*, NVNtextureSwizzle*, NVNtextureSwizzle*, NVNtextureSwizzle*)) \
    X(nvnTextureViewGetDepthStencilMode, NVNboolean, (const NVNtextureView*, NVNtextureDepthStencilMode*)) \
    X(nvnTextureViewGetTarget, NVNboolean, (const NVNtextureView*, NVNtextureTarget*)) \
    X(nvnTextureViewCompare, NVNboolean, (const NVNtextureView*, const NVNtextureView*)) \
    X(nvnTextureInitialize, NVNboolean, (NVNtexture*, const NVNtextureBuilder*)) \
    X(nvnTextureGetZCullStorageSize, std::size_t, (const NVNtexture*)) \
    X(nvnTextureFinalize, void, (NVNtexture*)) \
    X(nvnTextureSetDebugLabel, void, (NVNtexture*, const char*)) \
    X(nvnTextureGetStorageClass, NVNstorageClass, (const NVNtexture*)) \
    X(nvnTextureGetViewOffset, ptrdiff_t, (const NVNtexture*, const NVNtextureView*)) \
    X(nvnTextureGetFlags, NVNtextureFlags, (const NVNtexture*)) \
    X(nvnTextureGetTarget, NVNtextureTarget, (const NVNtexture*)) \
    X(nvnTextureGetWidth, int, (const NVNtexture*)) \
    X(nvnTextureGetHeight, int, (const NVNtexture*)) \
    X(nvnTextureGetDepth, int, (const NVNtexture*)) \
    X(nvnTextureGetLevels, int, (const NVNtexture*)) \
    X(nvnTextureGetFormat, NVNformat, (const NVNtexture*)) \
    X(nvnTextureGetSamples, int, (const NVNtexture*)) \
    X(nvnTextureGetSwizzle, void, (const NVNtexture*, NVNtextureSwizzle*, NVNtextureSwizzle*, NVNtextureSwizzle*, NVNtextureSwizzle*)) \
    X(nvnTextureGetDepthStencilMode, NVNtextureDepthStencilMode, (const NVNtexture*)) \
    X(nvnTextureGetStride, ptrdiff_t, (const NVNtexture*)) \
    X(nvnTextureGetTextureAddress, NVNtextureAddress, (const NVNtexture*)) \
    X(nvnTextureGetSparseTileLayout, void, (const NVNtexture*, NVNtextureSparseTileLayout*)) \
    X(nvnTextureWriteTexels, void, (const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const void*)) \
    X(nvnTextureWriteTexelsStrided, void, (const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const void*, ptrdiff_t, ptrdiff_t)) \
    X(nvnTextureReadTexels, void, (const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, void*)) \
    X(nvnTextureReadTexelsStrided, void, (const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, void*, ptrdiff_t, ptrdiff_t)) \
    X(nvnTextureFlushTexels, void, (const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*)) \
    X(nvnTextureInvalidateTexels, void, (const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*)) \
    X(nvnTextureGetMemoryPool, NVNmemoryPool*, (const NVNtexture*)) \
    X(nvnTextureGetMemoryOffset, ptrdiff_t, (const NVNtexture*)) \
    X(nvnTextureGetStorageSize, int, (const NVNtexture*)) \
    X(nvnTextureCompare, NVNboolean, (const NVNtexture*, const NVNtexture*)) \
    X(nvnTextureGetDebugID, std::uint64_t, (const NVNtexture*)) \
    X(nvnSamplerBuilderSetDevice, void, (NVNsamplerBuilder*, NVNdevice*)) \
    X(nvnSamplerBuilderSetDefaults, void, (NVNsamplerBuilder*)) \
    X(nvnSamplerBuilderGetDevice, NVNdevice*, (const NVNsamplerBuilder*)) \
    X(nvnSamplerBuilderSetMinMagFilter, void, (NVNsamplerBuilder*, NVNminFilter, NVNmagFilter)) \
    X(nvnSamplerBuilderSetWrapMode, void, (NVNsamplerBuilder*, NVNwrapMode, NVNwrapMode, NVNwrapMode)) \
    X(nvnSamplerBuilderSetLodClamp, void, (NVNsamplerBuilder*, float, float)) \
    X(nvnSamplerBuilderSetLodBias, void, (NVNsamplerBuilder*, float)) \
    X(nvnSamplerBuilderSetCompare, void, (NVNsamplerBuilder*, NVNcompareMode, NVNcompareFunc)) \
    X(nvnSamplerBuilderSetBorderColor, void, (NVNsamplerBuilder*, const float*)) \
    X(nvnSamplerBuilderSetBorderColori, void, (NVNsamplerBuilder*, const int*)) \
    X(nvnSamplerBuilderSetBorderColorui, void, (NVNsamplerBuilder*, const std::uint32_t*)) \
    X(nvnSamplerBuilderSetMaxAnisotropy, void, (NVNsamplerBuilder*, float)) \
    X(nvnSamplerBuilderSetReductionFilter, void, (NVNsamplerBuilder*, NVNsamplerReduction)) \
    X(nvnSamplerBuilderSetLodSnap, void, (NVNsamplerBuilder*, float)) \
    X(nvnSamplerBuilderGetMinMagFilter, void, (const NVNsamplerBuilder*, NVNminFilter*, NVNmagFilter*)) \
    X(nvnSamplerBuilderGetWrapMode, void, (const NVNsamplerBuilder*, NVNwrapMode*, NVNwrapMode*, NVNwrapMode*)) \
    X(nvnSamplerBuilderGetLodClamp, void, (const NVNsamplerBuilder*, float*, float*)) \
    X(nvnSamplerBuilderGetLodBias, float, (const NVNsamplerBuilder*)) \
    X(nvnSamplerBuilderGetCompare, void, (const NVNsamplerBuilder*, NVNcompareMode*, NVNcompareFunc*)) \
    X(nvnSamplerBuilderGetBorderColor, void, (const NVNsamplerBuilder*, float*)) \
    X(nvnSamplerBuilderGetBorderColori, void, (const NVNsamplerBuilder*, int*)) \
    X(nvnSamplerBuilderGetBorderColorui, void, (const NVNsamplerBuilder*, std::uint32_t*)) \
    X(nvnSamplerBuilderGetMaxAnisotropy, float, (const NVNsamplerBuilder*)) \
    X(nvnSamplerBuilderGetReductionFilter, NVNsamplerReduction, (const NVNsamplerBuilder*)) \
    X(nvnSamplerBuilderGetLodSnap, float, (const NVNsamplerBuilder*)) \
    X(nvnSamplerInitialize, NVNboolean, (NVNsampler*, const NVNsamplerBuilder*)) \
    X(nvnSamplerFinalize, void, (NVNsampler*)) \
    X(nvnSamplerSetDebugLabel, void, (NVNsampler*, const char*)) \
    X(nvnSamplerGetMinMagFilter, void, (const NVNsampler*, NVNminFilter*, NVNmagFilter*)) \
    X(nvnSamplerGetWrapMode, void, (const NVNsampler*, NVNwrapMode*, NVNwrapMode*, NVNwrapMode*)) \
    X(nvnSamplerGetLodClamp, void, (const NVNsampler*, float*, float*)) \
    X(nvnSamplerGetLodBias, float, (const NVNsampler*)) \
    X(nvnSamplerGetCompare, void, (const NVNsampler*, NVNcompareMode*, NVNcompareFunc*)) \
    X(nvnSamplerGetBorderColor, void, (const NVNsampler*, float*)) \
    X(nvnSamplerGetBorderColori, void, (const NVNsampler*, int*)) \
    X(nvnSamplerGetBorderColorui, void, (const NVNsampler*, std::uint32_t*)) \
    X(nvnSamplerGetMaxAnisotropy, float, (const NVNsampler*)) \
    X(nvnSamplerGetReductionFilter, NVNsamplerReduction, (const NVNsampler*)) \
    X(nvnSamplerCompare, NVNboolean, (const NVNsampler*, const NVNsampler*)) \
    X(nvnSamplerGetDebugID, std::uint64_t, (const NVNsampler*)) \
    X(nvnBlendStateSetDefaults, void, (NVNblendState*)) \
    X(nvnBlendStateSetBlendTarget, void, (NVNblendState*, int)) \
    X(nvnBlendStateSetBlendFunc, void, (NVNblendState*, NVNblendFunc, NVNblendFunc, NVNblendFunc, NVNblendFunc)) \
    X(nvnBlendStateSetBlendEquation, void, (NVNblendState*, NVNblendEquation, NVNblendEquation)) \
    X(nvnBlendStateSetAdvancedMode, void, (NVNblendState*, NVNblendAdvancedMode)) \
    X(nvnBlendStateSetAdvancedOverlap, void, (NVNblendState*, NVNblendAdvancedOverlap)) \
    X(nvnBlendStateSetAdvancedPremultipliedSrc, void, (NVNblendState*, std::uint32_t)) \
    X(nvnBlendStateSetAdvancedNormalizedDst, void, (NVNblendState*, std::uint32_t)) \
    X(nvnBlendStateGetBlendTarget, int, (const NVNblendState*)) \
    X(nvnBlendStateGetBlendFunc, void, (const NVNblendState*, NVNblendFunc*, NVNblendFunc*, NVNblendFunc*, NVNblendFunc*)) \
    X(nvnBlendStateGetBlendEquation, void, (const NVNblendState*, NVNblendEquation*, NVNblendEquation*)) \
    X(nvnBlendStateGetAdvancedMode, NVNblendAdvancedMode, (const NVNblendState*)) \
    X(nvnBlendStateGetAdvancedOverlap, NVNblendAdvancedOverlap, (const NVNblendState*)) \
    X(nvnBlendStateGetAdvancedPremultipliedSrc, NVNboolean, (const NVNblendState*)) \
    X(nvnBlendStateGetAdvancedNormalizedDst, NVNboolean, (const NVNblendState*)) \
    X(nvnColorStateSetDefaults, void, (NVNcolorState*)) \
    X(nvnColorStateSetBlendEnable, void, (NVNcolorState*, int, std::uint32_t)) \
    X(nvnColorStateSetLogicOp, void, (NVNcolorState*, NVNlogicOp)) \
    X(nvnColorStateSetAlphaTest, void, (NVNcolorState*, NVNalphaFunc)) \
    X(nvnColorStateGetBlendEnable, NVNboolean, (const NVNcolorState*, int)) \
    X(nvnColorStateGetLogicOp, NVNlogicOp, (const NVNcolorState*)) \
    X(nvnColorStateGetAlphaTest, NVNalphaFunc, (const NVNcolorState*)) \
    X(nvnChannelMaskStateSetDefaults, void, (NVNchannelMaskState*)) \
    X(nvnChannelMaskStateSetChannelMask, void, (NVNchannelMaskState*, int, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t)) \
    X(nvnChannelMaskStateGetChannelMask, void, (const NVNchannelMaskState*, int, NVNboolean*, NVNboolean*, NVNboolean*, NVNboolean*)) \
    X(nvnMultisampleStateSetDefaults, void, (NVNmultisampleState*)) \
    X(nvnMultisampleStateSetMultisampleEnable, void, (NVNmultisampleState*, std::uint32_t)) \
    X(nvnMultisampleStateSetSamples, void, (NVNmultisampleState*, int)) \
    X(nvnMultisampleStateSetAlphaToCoverageEnable, void, (NVNmultisampleState*, std::uint32_t)) \
    X(nvnMultisampleStateSetAlphaToCoverageDither, void, (NVNmultisampleState*, std::uint32_t)) \
    X(nvnMultisampleStateGetMultisampleEnable, NVNboolean, (const NVNmultisampleState*)) \
    X(nvnMultisampleStateGetSamples, int, (const NVNmultisampleState*)) \
    X(nvnMultisampleStateGetAlphaToCoverageEnable, NVNboolean, (const NVNmultisampleState*)) \
    X(nvnMultisampleStateGetAlphaToCoverageDither, NVNboolean, (const NVNmultisampleState*)) \
    X(nvnMultisampleStateSetRasterSamples, void, (NVNmultisampleState*, int)) \
    X(nvnMultisampleStateGetRasterSamples, int, (NVNmultisampleState*)) \
    X(nvnMultisampleStateSetCoverageModulationMode, void, (NVNmultisampleState*, NVNcoverageModulationMode)) \
    X(nvnMultisampleStateGetCoverageModulationMode, NVNcoverageModulationMode, (const NVNmultisampleState*)) \
    X(nvnMultisampleStateSetCoverageToColorEnable, void, (NVNmultisampleState*, std::uint32_t)) \
    X(nvnMultisampleStateGetCoverageToColorEnable, NVNboolean, (const NVNmultisampleState*)) \
    X(nvnMultisampleStateSetCoverageToColorOutput, void, (NVNmultisampleState*, int)) \
    X(nvnMultisampleStateGetCoverageToColorOutput, int, (const NVNmultisampleState*)) \
    X(nvnMultisampleStateSetSampleLocationsEnable, void, (NVNmultisampleState*, std::uint32_t)) \
    X(nvnMultisampleStateGetSampleLocationsEnable, NVNboolean, (const NVNmultisampleState*)) \
    X(nvnMultisampleStateGetSampleLocationsGrid, void, (NVNmultisampleState*, int*, int*)) \
    X(nvnMultisampleStateSetSampleLocationsGridEnable, void, (NVNmultisampleState*, std::uint32_t)) \
    X(nvnMultisampleStateGetSampleLocationsGridEnable, NVNboolean, (const NVNmultisampleState*)) \
    X(nvnMultisampleStateSetSampleLocations, void, (NVNmultisampleState*, int, int, const float*)) \
    X(nvnPolygonStateSetDefaults, void, (NVNpolygonState*)) \
    X(nvnPolygonStateSetCullFace, void, (NVNpolygonState*, NVNface)) \
    X(nvnPolygonStateSetFrontFace, void, (NVNpolygonState*, NVNfrontFace)) \
    X(nvnPolygonStateSetPolygonMode, void, (NVNpolygonState*, NVNpolygonMode)) \
    X(nvnPolygonStateSetPolygonOffsetEnables, void, (NVNpolygonState*, int)) \
    X(nvnPolygonStateGetCullFace, NVNface, (const NVNpolygonState*)) \
    X(nvnPolygonStateGetFrontFace, NVNfrontFace, (const NVNpolygonState*)) \
    X(nvnPolygonStateGetPolygonMode, NVNpolygonMode, (const NVNpolygonState*)) \
    X(nvnPolygonStateGetPolygonOffsetEnables, NVNpolygonOffsetEnable, (const NVNpolygonState*)) \
    X(nvnDepthStencilStateSetDefaults, void, (NVNdepthStencilState*)) \
    X(nvnDepthStencilStateSetDepthTestEnable, void, (NVNdepthStencilState*, std::uint32_t)) \
    X(nvnDepthStencilStateSetDepthWriteEnable, void, (NVNdepthStencilState*, std::uint32_t)) \
    X(nvnDepthStencilStateSetDepthFunc, void, (NVNdepthStencilState*, NVNdepthFunc)) \
    X(nvnDepthStencilStateSetStencilTestEnable, void, (NVNdepthStencilState*, std::uint32_t)) \
    X(nvnDepthStencilStateSetStencilFunc, void, (NVNdepthStencilState*, NVNface, NVNstencilFunc)) \
    X(nvnDepthStencilStateSetStencilOp, void, (NVNdepthStencilState*, NVNface, NVNstencilOp, NVNstencilOp, NVNstencilOp)) \
    X(nvnDepthStencilStateGetDepthTestEnable, NVNboolean, (const NVNdepthStencilState*)) \
    X(nvnDepthStencilStateGetDepthWriteEnable, NVNboolean, (const NVNdepthStencilState*)) \
    X(nvnDepthStencilStateGetDepthFunc, NVNdepthFunc, (const NVNdepthStencilState*)) \
    X(nvnDepthStencilStateGetStencilTestEnable, NVNboolean, (const NVNdepthStencilState*)) \
    X(nvnDepthStencilStateGetStencilFunc, NVNstencilFunc, (const NVNdepthStencilState*, NVNface)) \
    X(nvnDepthStencilStateGetStencilOp, void, (const NVNdepthStencilState*, NVNface, NVNstencilOp*, NVNstencilOp*, NVNstencilOp*)) \
    X(nvnVertexAttribStateSetDefaults, void, (NVNvertexAttribState*)) \
    X(nvnVertexAttribStateSetFormat, void, (NVNvertexAttribState*, NVNformat, ptrdiff_t)) \
    X(nvnVertexAttribStateSetStreamIndex, void, (NVNvertexAttribState*, int)) \
    X(nvnVertexAttribStateGetFormat, void, (const NVNvertexAttribState*, NVNformat*, ptrdiff_t*)) \
    X(nvnVertexAttribStateGetStreamIndex, int, (const NVNvertexAttribState*)) \
    X(nvnVertexStreamStateSetDefaults, void, (NVNvertexStreamState*)) \
    X(nvnVertexStreamStateSetStride, void, (NVNvertexStreamState*, ptrdiff_t)) \
    X(nvnVertexStreamStateSetDivisor, void, (NVNvertexStreamState*, int)) \
    X(nvnVertexStreamStateGetStride, ptrdiff_t, (const NVNvertexStreamState*)) \
    X(nvnVertexStreamStateGetDivisor, int, (const NVNvertexStreamState*)) \
    X(nvnCommandBufferInitialize, NVNboolean, (NVNcommandBuffer*, NVNdevice*)) \
    X(nvnCommandBufferFinalize, void, (NVNcommandBuffer*)) \
    X(nvnCommandBufferSetDebugLabel, void, (NVNcommandBuffer*, const char*)) \
    X(nvnCommandBufferSetMemoryCallback, void, (NVNcommandBuffer*, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC)) \
    X(nvnCommandBufferSetMemoryCallbackData, void, (NVNcommandBuffer*, void*)) \
    X(nvnCommandBufferAddCommandMemory, void, (NVNcommandBuffer*, const NVNmemoryPool*, ptrdiff_t, std::size_t)) \
    X(nvnCommandBufferAddControlMemory, void, (NVNcommandBuffer*, void*, std::size_t)) \
    X(nvnCommandBufferGetCommandMemorySize, std::size_t, (const NVNcommandBuffer*)) \
    X(nvnCommandBufferGetCommandMemoryUsed, std::size_t, (const NVNcommandBuffer*)) \
    X(nvnCommandBufferGetCommandMemoryFree, std::size_t, (const NVNcommandBuffer*)) \
    X(nvnCommandBufferGetControlMemorySize, std::size_t, (const NVNcommandBuffer*)) \
    X(nvnCommandBufferGetControlMemoryUsed, std::size_t, (const NVNcommandBuffer*)) \
    X(nvnCommandBufferGetControlMemoryFree, std::size_t, (const NVNcommandBuffer*)) \
    X(nvnCommandBufferBeginRecording, void, (NVNcommandBuffer*)) \
    X(nvnCommandBufferEndRecording, NVNcommandHandle, (NVNcommandBuffer*)) \
    X(nvnCommandBufferCallCommands, void, (NVNcommandBuffer*, int, const NVNcommandHandle*)) \
    X(nvnCommandBufferCopyCommands, void, (NVNcommandBuffer*, int, const NVNcommandHandle*)) \
    X(nvnCommandBufferBindBlendState, void, (NVNcommandBuffer*, const NVNblendState*)) \
    X(nvnCommandBufferBindChannelMaskState, void, (NVNcommandBuffer*, const NVNchannelMaskState*)) \
    X(nvnCommandBufferBindColorState, void, (NVNcommandBuffer*, const NVNcolorState*)) \
    X(nvnCommandBufferBindMultisampleState, void, (NVNcommandBuffer*, const NVNmultisampleState*)) \
    X(nvnCommandBufferBindPolygonState, void, (NVNcommandBuffer*, const NVNpolygonState*)) \
    X(nvnCommandBufferBindDepthStencilState, void, (NVNcommandBuffer*, const NVNdepthStencilState*)) \
    X(nvnCommandBufferBindVertexAttribState, void, (NVNcommandBuffer*, int, const NVNvertexAttribState*)) \
    X(nvnCommandBufferBindVertexStreamState, void, (NVNcommandBuffer*, int, const NVNvertexStreamState*)) \
    X(nvnCommandBufferBindProgram, void, (NVNcommandBuffer*, const NVNprogram*, int)) \
    X(nvnCommandBufferBindVertexBuffer, void, (NVNcommandBuffer*, int, NVNbufferAddress, std::size_t)) \
    X(nvnCommandBufferBindVertexBuffers, void, (NVNcommandBuffer*, int, int, const NVNbufferRange*)) \
    X(nvnCommandBufferBindUniformBuffer, void, (NVNcommandBuffer*, NVNshaderStage, int, NVNbufferAddress, std::size_t)) \
    X(nvnCommandBufferBindUniformBuffers, void, (NVNcommandBuffer*, NVNshaderStage, int, int, const NVNbufferRange*)) \
    X(nvnCommandBufferBindTransformFeedbackBuffer, void, (NVNcommandBuffer*, int, NVNbufferAddress, std::size_t)) \
    X(nvnCommandBufferBindTransformFeedbackBuffers, void, (NVNcommandBuffer*, int, int, const NVNbufferRange*)) \
    X(nvnCommandBufferBindStorageBuffer, void, (NVNcommandBuffer*, NVNshaderStage, int, NVNbufferAddress, std::size_t)) \
    X(nvnCommandBufferBindStorageBuffers, void, (NVNcommandBuffer*, NVNshaderStage, int, int, const NVNbufferRange*)) \
    X(nvnCommandBufferBindTexture, void, (NVNcommandBuffer*, NVNshaderStage, int, NVNtextureHandle)) \
    X(nvnCommandBufferBindTextures, void, (NVNcommandBuffer*, NVNshaderStage, int, int, const NVNtextureHandle*)) \
    X(nvnCommandBufferBindImage, void, (NVNcommandBuffer*, NVNshaderStage, int, NVNimageHandle)) \
    X(nvnCommandBufferBindImages, void, (NVNcommandBuffer*, NVNshaderStage, int, int, const NVNimageHandle*)) \
    X(nvnCommandBufferSetPatchSize, void, (NVNcommandBuffer*, int)) \
    X(nvnCommandBufferSetInnerTessellationLevels, void, (NVNcommandBuffer*, const float*)) \
    X(nvnCommandBufferSetOuterTessellationLevels, void, (NVNcommandBuffer*, const float*)) \
    X(nvnCommandBufferSetPrimitiveRestart, void, (NVNcommandBuffer*, std::uint32_t, int)) \
    X(nvnCommandBufferBeginTransformFeedback, void, (NVNcommandBuffer*, NVNbufferAddress)) \
    X(nvnCommandBufferEndTransformFeedback, void, (NVNcommandBuffer*, NVNbufferAddress)) \
    X(nvnCommandBufferPauseTransformFeedback, void, (NVNcommandBuffer*, NVNbufferAddress)) \
    X(nvnCommandBufferResumeTransformFeedback, void, (NVNcommandBuffer*, NVNbufferAddress)) \
    X(nvnCommandBufferDrawTransformFeedback, void, (NVNcommandBuffer*, NVNdrawPrimitive, NVNbufferAddress)) \
    X(nvnCommandBufferDrawArrays, void, (NVNcommandBuffer*, NVNdrawPrimitive, int, int)) \
    X(nvnCommandBufferDrawElements, void, (NVNcommandBuffer*, NVNdrawPrimitive, NVNindexType, int, NVNbufferAddress)) \
    X(nvnCommandBufferDrawElementsBaseVertex, void, (NVNcommandBuffer*, NVNdrawPrimitive, NVNindexType, int, NVNbufferAddress, int)) \
    X(nvnCommandBufferDrawArraysInstanced, void, (NVNcommandBuffer*, NVNdrawPrimitive, int, int, int, int)) \
    X(nvnCommandBufferDrawElementsInstanced, void, (NVNcommandBuffer*, NVNdrawPrimitive, NVNindexType, int, NVNbufferAddress, int, int, int)) \
    X(nvnCommandBufferDrawArraysIndirect, void, (NVNcommandBuffer*, NVNdrawPrimitive, NVNbufferAddress)) \
    X(nvnCommandBufferDrawElementsIndirect, void, (NVNcommandBuffer*, NVNdrawPrimitive, NVNindexType, NVNbufferAddress, NVNbufferAddress)) \
    X(nvnCommandBufferMultiDrawArraysIndirectCount, void, (NVNcommandBuffer*, NVNdrawPrimitive, NVNbufferAddress, NVNbufferAddress, int, ptrdiff_t)) \
    X(nvnCommandBufferMultiDrawElementsIndirectCount, void, (NVNcommandBuffer*, NVNdrawPrimitive, NVNindexType, NVNbufferAddress, NVNbufferAddress, NVNbufferAddress, int, ptrdiff_t)) \
    X(nvnCommandBufferClearColor, void, (NVNcommandBuffer*, int, const float*, int)) \
    X(nvnCommandBufferClearColori, void, (NVNcommandBuffer*, int, const int*, int)) \
    X(nvnCommandBufferClearColorui, void, (NVNcommandBuffer*, int, const std::uint32_t*, int)) \
    X(nvnCommandBufferClearDepthStencil, void, (NVNcommandBuffer*, float, std::uint32_t, int, int)) \
    X(nvnCommandBufferDispatchCompute, void, (NVNcommandBuffer*, int, int, int)) \
    X(nvnCommandBufferDispatchComputeIndirect, void, (NVNcommandBuffer*, NVNbufferAddress)) \
    X(nvnCommandBufferSetViewport, void, (NVNcommandBuffer*, int, int, int, int)) \
    X(nvnCommandBufferSetViewports, void, (NVNcommandBuffer*, int, int, const float*)) \
    X(nvnCommandBufferSetViewportSwizzles, void, (NVNcommandBuffer*, int, int, const NVNviewportSwizzle*)) \
    X(nvnCommandBufferSetScissor, void, (NVNcommandBuffer*, int, int, int, int)) \
    X(nvnCommandBufferSetScissors, void, (NVNcommandBuffer*, int, int, const int*)) \
    X(nvnCommandBufferSetDepthRange, void, (NVNcommandBuffer*, float, float)) \
    X(nvnCommandBufferSetDepthBounds, void, (NVNcommandBuffer*, std::uint32_t, float, float)) \
    X(nvnCommandBufferSetDepthRanges, void, (NVNcommandBuffer*, int, int, const float*)) \
    X(nvnCommandBufferSetTiledCacheAction, void, (NVNcommandBuffer*, NVNtiledCacheAction)) \
    X(nvnCommandBufferSetTiledCacheTileSize, void, (NVNcommandBuffer*, int, int)) \
    X(nvnCommandBufferBindSeparateTexture, void, (NVNcommandBuffer*, NVNshaderStage, int, NVNseparateTextureHandle)) \
    X(nvnCommandBufferBindSeparateSampler, void, (NVNcommandBuffer*, NVNshaderStage, int, NVNseparateSamplerHandle)) \
    X(nvnCommandBufferBindSeparateTextures, void, (NVNcommandBuffer*, NVNshaderStage, int, int, const NVNseparateTextureHandle*)) \
    X(nvnCommandBufferBindSeparateSamplers, void, (NVNcommandBuffer*, NVNshaderStage, int, int, const NVNseparateSamplerHandle*)) \
    X(nvnCommandBufferSetStencilValueMask, void, (NVNcommandBuffer*, NVNface, int)) \
    X(nvnCommandBufferSetStencilMask, void, (NVNcommandBuffer*, NVNface, int)) \
    X(nvnCommandBufferSetStencilRef, void, (NVNcommandBuffer*, NVNface, int)) \
    X(nvnCommandBufferSetBlendColor, void, (NVNcommandBuffer*, const float*)) \
    X(nvnCommandBufferSetPointSize, void, (NVNcommandBuffer*, float)) \
    X(nvnCommandBufferSetLineWidth, void, (NVNcommandBuffer*, float)) \
    X(nvnCommandBufferSetPolygonOffsetClamp, void, (NVNcommandBuffer*, float, float, float)) \
    X(nvnCommandBufferSetAlphaRef, void, (NVNcommandBuffer*, float)) \
    X(nvnCommandBufferSetSampleMask, void, (NVNcommandBuffer*, int)) \
    X(nvnCommandBufferSetRasterizerDiscard, void, (NVNcommandBuffer*, std::uint32_t)) \
    X(nvnCommandBufferSetDepthClamp, void, (NVNcommandBuffer*, std::uint32_t)) \
    X(nvnCommandBufferSetConservativeRasterEnable, void, (NVNcommandBuffer*, std::uint32_t)) \
    X(nvnCommandBufferSetConservativeRasterDilate, void, (NVNcommandBuffer*, float)) \
    X(nvnCommandBufferSetSubpixelPrecisionBias, void, (NVNcommandBuffer*, int, int)) \
    X(nvnCommandBufferCopyBufferToTexture, void, (NVNcommandBuffer*, NVNbufferAddress, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, int)) \
    X(nvnCommandBufferCopyTextureToBuffer, void, (NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, NVNbufferAddress, int)) \
    X(nvnCommandBufferCopyTextureToTexture, void, (NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, int)) \
    X(nvnCommandBufferCopyBufferToBuffer, void, (NVNcommandBuffer*, NVNbufferAddress, NVNbufferAddress, std::size_t, int)) \
    X(nvnCommandBufferClearBuffer, void, (NVNcommandBuffer*, NVNbufferAddress, std::size_t, std::uint32_t)) \
    X(nvnCommandBufferClearTexture, void, (NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const float*, int)) \
    X(nvnCommandBufferClearTexturei, void, (NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const int*, int)) \
    X(nvnCommandBufferClearTextureui, void, (NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNcopyRegion*, const std::uint32_t*, int)) \
    X(nvnCommandBufferUpdateUniformBuffer, void, (NVNcommandBuffer*, NVNbufferAddress, std::size_t, ptrdiff_t, std::size_t, const void*)) \
    X(nvnCommandBufferReportCounter, void, (NVNcommandBuffer*, NVNcounterType, NVNbufferAddress)) \
    X(nvnCommandBufferResetCounter, void, (NVNcommandBuffer*, NVNcounterType)) \
    X(nvnCommandBufferReportValue, void, (NVNcommandBuffer*, std::uint32_t, NVNbufferAddress)) \
    X(nvnCommandBufferSetRenderEnable, void, (NVNcommandBuffer*, std::uint32_t)) \
    X(nvnCommandBufferSetRenderEnableConditional, void, (NVNcommandBuffer*, NVNconditionalRenderMode, NVNbufferAddress)) \
    X(nvnCommandBufferSetRenderTargets, void, (NVNcommandBuffer*, int, const NVNtexture* const*, const NVNtextureView* const*, const NVNtexture*, const NVNtextureView*)) \
    X(nvnCommandBufferDiscardColor, void, (NVNcommandBuffer*, int)) \
    X(nvnCommandBufferDiscardDepthStencil, void, (NVNcommandBuffer*)) \
    X(nvnCommandBufferDownsample, void, (NVNcommandBuffer*, const NVNtexture*, const NVNtexture*)) \
    X(nvnCommandBufferTiledDownsample, void, (NVNcommandBuffer*, const NVNtexture*, const NVNtexture*)) \
    X(nvnCommandBufferDownsampleTextureView, void, (NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNtexture*, const NVNtextureView*)) \
    X(nvnCommandBufferTiledDownsampleTextureView, void, (NVNcommandBuffer*, const NVNtexture*, const NVNtextureView*, const NVNtexture*, const NVNtextureView*)) \
    X(nvnCommandBufferBarrier, void, (NVNcommandBuffer*, int)) \
    X(nvnCommandBufferWaitSync, void, (NVNcommandBuffer*, const NVNsync*)) \
    X(nvnCommandBufferFenceSync, void, (NVNcommandBuffer*, NVNsync*, NVNsyncCondition, int)) \
    X(nvnCommandBufferSetTexturePool, void, (NVNcommandBuffer*, const NVNtexturePool*)) \
    X(nvnCommandBufferSetSamplerPool, void, (NVNcommandBuffer*, const NVNsamplerPool*)) \
    X(nvnCommandBufferSetShaderScratchMemory, void, (NVNcommandBuffer*, const NVNmemoryPool*, ptrdiff_t, std::size_t)) \
    X(nvnCommandBufferSaveZCullData, void, (NVNcommandBuffer*, NVNbufferAddress, std::size_t)) \
    X(nvnCommandBufferRestoreZCullData, void, (NVNcommandBuffer*, NVNbufferAddress, std::size_t)) \
    X(nvnCommandBufferSetCopyRowStride, void, (NVNcommandBuffer*, ptrdiff_t)) \
    X(nvnCommandBufferSetCopyImageStride, void, (NVNcommandBuffer*, ptrdiff_t)) \
    X(nvnCommandBufferGetCopyRowStride, ptrdiff_t, (const NVNcommandBuffer*)) \
    X(nvnCommandBufferGetCopyImageStride, ptrdiff_t, (const NVNcommandBuffer*)) \
    X(nvnCommandBufferDrawTexture, void, (NVNcommandBuffer*, NVNtextureHandle, const NVNdrawTextureRegion*, const NVNdrawTextureRegion*)) \
    X(nvnProgramSetSubroutineLinkage, NVNboolean, (NVNprogram*, int, const NVNsubroutineLinkageMapPtr*)) \
    X(nvnCommandBufferSetProgramSubroutines, void, (NVNcommandBuffer*, NVNprogram*, NVNshaderStage, const int, const int, const int*)) \
    X(nvnCommandBufferBindCoverageModulationTable, void, (NVNcommandBuffer*, const float*)) \
    X(nvnCommandBufferResolveDepthBuffer, void, (NVNcommandBuffer*)) \
    X(nvnCommandBufferPushDebugGroupStatic, void, (NVNcommandBuffer*, std::uint32_t, const char*)) \
    X(nvnCommandBufferPushDebugGroupDynamic, void, (NVNcommandBuffer*, std::uint32_t, const char*)) \
    X(nvnCommandBufferPushDebugGroup, void, (NVNcommandBuffer*, std::uint32_t, const char*)) \
    X(nvnCommandBufferPopDebugGroup, void, (NVNcommandBuffer*)) \
    X(nvnCommandBufferPopDebugGroupId, void, (NVNcommandBuffer*, std::uint32_t)) \
    X(nvnCommandBufferInsertDebugMarkerStatic, void, (NVNcommandBuffer*, std::uint32_t, const char*)) \
    X(nvnCommandBufferInsertDebugMarkerDynamic, void, (NVNcommandBuffer*, std::uint32_t, const char*)) \
    X(nvnCommandBufferInsertDebugMarker, void, (NVNcommandBuffer*, const char*)) \
    X(nvnCommandBufferGetMemoryCallback, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC, (const NVNcommandBuffer*)) \
    X(nvnCommandBufferGetMemoryCallbackData, void*, (const NVNcommandBuffer*)) \
    X(nvnCommandBufferIsRecording, NVNboolean, (const NVNcommandBuffer*)) \
    X(nvnSyncInitialize, NVNboolean, (NVNsync*, NVNdevice*)) \
    X(nvnSyncFinalize, void, (NVNsync*)) \
    X(nvnSyncSetDebugLabel, void, (NVNsync*, const char*)) \
    X(nvnQueueFenceSync, void, (NVNqueue*, NVNsync*, NVNsyncCondition, int)) \
    X(nvnSyncWait, NVNsyncWaitResult, (const NVNsync*, std::uint64_t)) \
    X(nvnQueueWaitSync, NVNboolean, (NVNqueue*, const NVNsync*)) \
    X(nvnEventBuilderSetDefaults, void, (NVNeventBuilder*)) \
    X(nvnEventBuilderSetStorage, void, (NVNeventBuilder*, const NVNmemoryPool*, std::int64_t)) \
    X(nvnEventInitialize, NVNboolean, (NVNevent*, const NVNeventBuilder*)) \
    X(nvnEventFinalize, void, (NVNevent*)) \
    X(nvnEventGetValue, std::uint32_t, (const NVNevent*)) \
    X(nvnEventSignal, void, (NVNevent*, NVNeventSignalMode, std::uint32_t)) \
    X(nvnCommandBufferWaitEvent, void, (NVNcommandBuffer*, const NVNevent*, NVNeventWaitMode, std::uint32_t)) \
    X(nvnCommandBufferSignalEvent, void, (NVNcommandBuffer*, const NVNevent*, NVNeventSignalMode, NVNeventSignalLocation, int, std::uint32_t)) \
    X(nvnBootstrapLoader, PFNNVNGENERICFUNCPTRPROC, (const char*)) \
    X(nvnCommandBufferBindTexturesDeferred, void, (NVNcommandBuffer*, NVNshaderStage, int, int, const NVNtextureHandle*))

#define ARMIGO_NVN_UNTYPED_ENTRY_POINTS(X) \
    X(nvnQueueBuilderSetQueuePriority) \
    X(nvnQueueBuilderGetQueuePriority) \
    X(nvnQueueBuilderGetDevice) \
    X(nvnQueueBuilderGetFlags) \
    X(nvnQueueBuilderGetCommandMemorySize) \
    X(nvnQueueBuilderGetComputeMemorySize) \
    X(nvnQueueBuilderGetControlMemorySize) \
    X(nvnQueueBuilderGetCommandFlushThreshold) \
    X(nvnQueueBuilderGetMemorySize) \
    X(nvnQueueBuilderGetMemory) \
    X(nvnWindowBuilderSetNumActiveTextures) \
    X(nvnWindowBuilderGetDevice) \
    X(nvnWindowBuilderGetNumTextures) \
    X(nvnWindowBuilderGetTexture) \
    X(nvnWindowBuilderGetNumActiveTextures) \
    X(nvnWindowSetNumActiveTextures) \
    X(nvnWindowGetNumActiveTextures) \
    X(nvnWindowGetNumTextures) \
    X(nvnProgramSetSampleShading) \
    X(nvnTextureBuilderGetPackagedTextureLayout) \
    X(nvnTextureBuilderGetRawStorageClass) \
    X(nvnTextureGetRawStorageClass) \
    X(nvnCommandBufferSetCommandMemoryCallbackEnabled) \
    X(nvnCommandBufferSetColorReductionEnable) \
    X(nvnCommandBufferSetColorReductionThresholds) \
    X(nvnSyncInitializeFromFencedGLSync) \
    X(nvnSyncCreateGLSync) \
    X(nvnEventBuilderGetStorage) \
    X(nvnEventBuilderGetMemoryPool) \
    X(nvnEventBuilderGetMemoryOffset) \
    X(nvnEventGetMemoryPool) \
    X(nvnEventGetMemoryOffset) \
    X(nvnCommandBufferSetStencilCullCriteria) \
    X(nvnCommandBufferInsertCheckpointStatic) \
    X(nvnCommandBufferInsertCheckpointDynamic) \
    X(nvnDeviceSetAutomaticCheckpointsEnable) \
    X(nvnDeviceSetAutomaticCheckpointsSamplingInterval) \
    X(nvnDeviceSetAutomaticCheckpointsByDebugGroup) \
    X(nvnDeviceSetCheckpointCallStacksEnable) \
    X(nvnDeviceSetCheckpointMaxCallStacksDepth) \
    X(nvnCommandBufferBindImagesDeferred) \
    X(nvnCommandBufferBindSeparateSamplersDeferred) \
    X(nvnCommandBufferBindSeparateTexturesDeferred) \
    X(nvnCommandBufferBindStorageBuffersDeferred) \
    X(nvnCommandBufferBindUniformBuffersDeferred) \
    X(nvnCommandBufferOverrideSubtileSize) \
    X(nvnCommandBufferSetControlMemoryCallbackEnabled) \
    X(nvnCommandBufferSetProvokingVertexMode) \
    X(nvnCommandBufferSetZCullZF32CompressionEnable) \
    X(nvnDeviceSetAutomaticCheckpointsMask) \
    X(nvnPrivate0) \
    X(nvnProgramGetSubtileSize) \
    X(nvnSyncGetNvRmSyncNVX) \
    X(nvnSyncInitializeFromNvRmSyncNVX) \
    X(nvnTextureBuilderGetViewOffset) \
    X(nvnTextureGetNvRmSurfaceNVX)
// clang-format on
