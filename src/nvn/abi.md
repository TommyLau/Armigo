# NVN direct-call ABI

NVN entry points cross directly from guest AArch64 code into functions compiled
for Apple arm64. The two calling conventions use the same integer, SIMD, result,
link, and intra-procedure-call registers for the signatures in the current
entry table, but they differ in two relevant areas:

- Apple arm64 requires callers to extend integer arguments narrower than 32
  bits. AAPCS64 permits the unused high bits to remain unspecified.
- AAPCS64 gives stack arguments eight-byte slots. Apple arm64 packs stack
  arguments according to their natural size and alignment.
- AAPCS64 rounds an integer-register argument with 16-byte alignment to an even
  register, while Apple arm64 permits it to start in an odd register.
- Apple arm64 may omit empty record arguments. The current table has none.

The typed host contract therefore represents boolean inputs as `std::uint32_t`.
Implementations must use `guest_bool()` and may not test the full word. Boolean
results remain `NVNboolean`: both sides consume the low byte of the result, and
the C++ implementation produces a value converted to that result type.

## Typed-signature audit

The table contains 481 typed entries and 60 explicitly untyped entries.
`EntryAbi` checks every typed function at compile time. No typed entry has an
integer or enumeration parameter narrower than 32 bits, an over-aligned
integer-bank parameter, or an empty record parameter.

The boolean-input entries represented with widened parameters are:

- `nvnDeviceInstallDebugCallback`
- `nvnBlendStateSetAdvancedPremultipliedSrc`
- `nvnBlendStateSetAdvancedNormalizedDst`
- `nvnColorStateSetBlendEnable`
- `nvnChannelMaskStateSetChannelMask`
- `nvnMultisampleStateSetMultisampleEnable`
- `nvnMultisampleStateSetAlphaToCoverageEnable`
- `nvnMultisampleStateSetAlphaToCoverageDither`
- `nvnMultisampleStateSetCoverageToColorEnable`
- `nvnMultisampleStateSetSampleLocationsEnable`
- `nvnMultisampleStateSetSampleLocationsGridEnable`
- `nvnDepthStencilStateSetDepthTestEnable`
- `nvnDepthStencilStateSetDepthWriteEnable`
- `nvnDepthStencilStateSetStencilTestEnable`
- `nvnCommandBufferSetPrimitiveRestart`
- `nvnCommandBufferClearDepthStencil`
- `nvnCommandBufferSetDepthBounds`
- `nvnCommandBufferSetRasterizerDiscard`
- `nvnCommandBufferSetDepthClamp`
- `nvnCommandBufferSetConservativeRasterEnable`
- `nvnCommandBufferSetRenderEnable`

The 45 narrow-result entries are:

- `nvnDeviceInitialize`
- `nvnDeviceRegisterFastClearColor`
- `nvnDeviceRegisterFastClearColori`
- `nvnDeviceRegisterFastClearColorui`
- `nvnDeviceRegisterFastClearDepth`
- `nvnDeviceIsExternalDebuggerAttached`
- `nvnQueueInitialize`
- `nvnWindowInitialize`
- `nvnProgramInitialize`
- `nvnProgramSetShaders`
- `nvnProgramSetShadersExt`
- `nvnMemoryPoolInitialize`
- `nvnMemoryPoolMapVirtual`
- `nvnTexturePoolInitialize`
- `nvnSamplerPoolInitialize`
- `nvnBufferInitialize`
- `nvnTextureViewGetLevels`
- `nvnTextureViewGetLayers`
- `nvnTextureViewGetFormat`
- `nvnTextureViewGetSwizzle`
- `nvnTextureViewGetDepthStencilMode`
- `nvnTextureViewGetTarget`
- `nvnTextureViewCompare`
- `nvnTextureInitialize`
- `nvnTextureCompare`
- `nvnSamplerInitialize`
- `nvnSamplerCompare`
- `nvnBlendStateGetAdvancedPremultipliedSrc`
- `nvnBlendStateGetAdvancedNormalizedDst`
- `nvnColorStateGetBlendEnable`
- `nvnMultisampleStateGetMultisampleEnable`
- `nvnMultisampleStateGetAlphaToCoverageEnable`
- `nvnMultisampleStateGetAlphaToCoverageDither`
- `nvnMultisampleStateGetCoverageToColorEnable`
- `nvnMultisampleStateGetSampleLocationsEnable`
- `nvnMultisampleStateGetSampleLocationsGridEnable`
- `nvnDepthStencilStateGetDepthTestEnable`
- `nvnDepthStencilStateGetDepthWriteEnable`
- `nvnDepthStencilStateGetStencilTestEnable`
- `nvnCommandBufferInitialize`
- `nvnProgramSetSubroutineLinkage`
- `nvnCommandBufferIsRecording`
- `nvnSyncInitialize`
- `nvnQueueWaitSync`
- `nvnEventInitialize`

No typed signature passes an argument on the stack. Three entries use all eight
integer argument registers and are the current boundary cases:

- `nvnCommandBufferDrawElementsInstanced`
- `nvnCommandBufferMultiDrawElementsIndirectCount`
- `nvnCommandBufferCopyTextureToTexture`

All functions have fixed parameter lists. Scalar and enumeration results use
the same result registers on both sides. The two eight-byte handle-record
results use `x0`, and the one large object-record result uses the common `x8`
indirect-result convention.

Consequently, the current install table needs no stack-layout assembly shim and
routes every implementation directly. A future typed entry that exhausts either
register bank fails the `EntryAbi` assertion. It must receive a static assembly
shim that copies AAPCS64 stack slots into Apple arm64 layout, and installation
must route that entry through the shim before the assertion is relaxed.

Untyped entries cannot be audited until their signatures are added. Their
unimplemented targets are safe because they do not consume arguments.

## Counting thunks

When `ARMIGO_NVN_COUNT` is exactly `1`, the loader returns one fixed-size static
assembly thunk per entry. A thunk uses only `x16` and `x17`, performs a relaxed
atomic increment, acquire-loads the current installed target, and tail-branches
to it. It does not change `x0`-`x7`, `v0`-`v7`, `x30`, or `sp`, so register and
stack arguments arrive unchanged. Updating an installed target also updates an
already returned counting thunk.

`call_counts()` returns the total entry count and fills as much of its output
span as fits. `sample_call_count_frame()` keeps a caller-owned snapshot, selects
the largest per-frame deltas without allocation, records the full frame total
in `CallCountFrameState::total_calls`, and sends the sorted selection
to a hook. `print_call_count_frame()` is a ready-made hook for diagnostics.
