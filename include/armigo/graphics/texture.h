// ============================================================================
// Armigo Graphics - GPU Texture Descriptors
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Texture descriptors decode TIC format words, swizzles, sRGB selection,
// targets, mip ranges, layers, and layout directly from the guest ABI.
// Sampler descriptors likewise decode coordinate normalization, filtering,
// trilinear optimization, and packed border-color fields from guest words.

#pragma once

#include <armigo/graphics/engines.h>
#include <armigo/graphics/resource_memory.h>
#include <armigo/graphics/texture_format.h>
#include <armigo/graphics/texture_layout.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace armigo::graphics {

inline constexpr std::size_t texture_descriptor_size = 32;
inline constexpr std::size_t max_texture_backing_size = 256u * 1024u * 1024u;
// Sampled 2D textures are admitted up to the per-axis Metal texture limit;
// the resolver rejects larger axes before staging or allocating anything.
inline constexpr std::uint32_t max_sampled_texture_axis = 16384;

enum class TextureErrorCode : std::uint8_t {
    DescriptorTooSmall,
    UnsupportedTextureFormat,
    UnsupportedComponentType,
    UnsupportedSwizzle,
    UnsupportedLayout,
    UnsupportedTextureType,
    UnsupportedTextureState,
    InvalidDimensions,
    ExtentOverflow,
    ExtentLimit,
    UnsupportedSamplerState,
    InvalidLodRange,
    NotDrawStop,
    StageOutOfRange,
    ConstantBufferSlotOutOfRange,
    ConstantBufferDisabled,
    HandleOutsideConstantBuffer,
    AddressOverflow,
    TextureIndexOutOfRange,
    SamplerIndexOutOfRange,
    MemoryRead,
};

enum class TextureField : std::uint8_t {
    None,
    Format,
    ComponentType,
    Swizzle,
    HeaderVersion,
    LoadStoreHint,
    BlockWidth,
    BlockHeight,
    BlockDepth,
    TileWidthSpacing,
    HeaderOptimization,
    DepthTexture,
    MipCount,
    TextureSrgb,
    TextureType,
    Depth,
    Sparse,
    NormalizedCoordinates,
    BaseLayer,
    ResourceMipRange,
    MultisampleMode,
    TextureMinLod,
    WrapMode,
    DepthCompare,
    MagnificationFilter,
    MinificationFilter,
    MipmapFilter,
    ReductionFilter,
    Anisotropy,
    LodBias,
    FloatCoordinateNormalization,
    TrilinearOptimization,
    BorderColor,
    LodRange,
    ConstantBuffer,
    HandleOffset,
    PoolAddress,
    PoolLimit,
    BackingRange,
};

struct TextureError {
    TextureErrorCode code = TextureErrorCode::DescriptorTooSmall;
    TextureField field = TextureField::None;
    std::uint32_t word = 0;
    std::uint64_t value = 0;
    std::uint64_t expected = 0;
    MemoryErrorCode memory_error = MemoryErrorCode::None;

    [[nodiscard]] const char* message() const noexcept;
};

// Admitted formats are the entries of the guest TIC format table
// (texture_format.h): a descriptor admits a format when its word-0 format
// index and component types, plus the word-4 sRGB flag, name a table entry.
// The descriptor's srgb flag selects the sampling transfer function where the
// format defines both encodings.
// Storage kinds of a TIC descriptor: block-linear tiling or the linear pitch
// layout (descriptor type Linear). The color-key variants and the buffer
// layout stay structured rejections.
enum class TextureLayout : std::uint8_t { BlockLinear, Pitch };
// Admitted TIC texture targets. Texture buffer has no staging path and no
// enumerator; a descriptor naming it returns a structured error.
enum class TextureType : std::uint8_t {
    Texture1D,
    Texture2D,
    Texture3D,
    Cubemap,
    Texture1DArray,
    Texture2DArray,
    Texture2DRect,
    CubemapArray,
};
// TIC swizzle sources; OneSInt and OneFloat are the integer-one and
// float-one variants distinguished by the guest encoding. Both map to Metal's One.
enum class TextureSwizzle : std::uint8_t {
    Zero,
    Red,
    Green,
    Blue,
    Alpha,
    OneSInt,
    OneFloat,
};
// TSC field encodings carry the guest enum values, so a decoded field
// is validated by switch against these enumerators and hardware codes outside
// them are reserved encodings.
enum class SamplerWrap : std::uint8_t {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
    Clamp,
    MirrorClampToEdge,
    MirrorClampToBorder,
    MirrorClamp,
};
enum class SamplerFilter : std::uint8_t { Nearest = 1, Linear = 2 };
enum class SamplerMipFilter : std::uint8_t { None = 1, Nearest = 2, Linear = 3 };
enum class SamplerReduction : std::uint8_t {
    WeightedAverage,
    Minimum,
    Maximum,
};
// The TSC comparison-operation field in TSC word-0 bits 10-12; enabled
// comparison sampling uses the function, disabled sampling ignores it.
enum class SamplerCompareFunction : std::uint8_t {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
};

// A decoded TIC texture descriptor. Except where noted, the extent fields
// describe the texture view the guest asked for: when the descriptor names a
// non-zero base mipmap level, width, height, depth, the GOB block fields,
// address, backing_extent, row_bytes and storage_height already describe that
// view, by retargeting a sub-range of a mip chain.
struct TextureDescriptor {
    std::array<std::uint32_t, 8> raw_words{};
    std::uint64_t address = 0;
    // Storage bytes of the view: every layer's mip chain, covered in full by
    // reads through resolve_texture_binding.
    std::size_t backing_extent = 0;
    // Level-0 (of the view) storage row stride: the GOB-aligned block row for
    // block-linear storage, or the encoded pitch stride for pitch storage.
    std::size_t row_bytes = 0;
    // Level-0 (of the view) storage rows; rows count texel blocks for
    // block-compressed formats.
    std::size_t storage_height = 0;
    // Distance between consecutive layers' mip chains in guest storage. For
    // views that start above level 0 the guest stores full mip chains per
    // layer, so this stays the full chain's aligned stride.
    std::size_t layer_stride = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // Texel depth: the encoded depth field for 3D targets, 1 otherwise.
    std::uint32_t depth = 0;
    // Mipmap levels of the view, after clamping to the widest axis.
    std::uint32_t levels = 0;
    // Array layers: the encoded depth field for array targets, 6 per
    // cubemap, encoded depth times 6 for cubemap arrays, 1 otherwise.
    std::uint32_t layers = 0;
    // Mip chain level this view starts at; 0 for a whole-chain texture.
    std::uint32_t base_level = 0;
    std::uint32_t samples = 0;
    std::uint8_t block_height_log2 = 0;
    // log2 GOBs per block-of-GOBs along Z; only 3D targets admit a non-zero
    // value, as the hardware requires.
    std::uint8_t block_depth_log2 = 0;
    TextureFormat format = TextureFormat::R10G10B10A2Unorm;
    TextureLayout layout = TextureLayout::BlockLinear;
    TextureType type = TextureType::Texture2D;
    std::array<TextureSwizzle, 4> swizzle{
        TextureSwizzle::Red, TextureSwizzle::Green,
        TextureSwizzle::Blue, TextureSwizzle::Alpha};
    bool srgb = false;
    bool normalized_coordinates = true;
    bool operator==(const TextureDescriptor&) const = default;
};

struct SamplerDescriptor {
    std::array<std::uint32_t, 8> raw_words{};
    std::array<SamplerWrap, 3> wrap{
        SamplerWrap::ClampToEdge, SamplerWrap::ClampToEdge,
        SamplerWrap::ClampToEdge};
    SamplerFilter magnification = SamplerFilter::Linear;
    SamplerFilter minification = SamplerFilter::Linear;
    SamplerMipFilter mip_filter = SamplerMipFilter::Linear;
    SamplerReduction reduction = SamplerReduction::WeightedAverage;
    bool depth_compare = false;
    SamplerCompareFunction compare_function = SamplerCompareFunction::Never;
    // This controls sampler/border behavior independently of texture format.
    bool srgb_border_conversion = false;
    // Packed TSC sRGB-border RGB bytes. These replace the float RGB border
    // components only when conversion is enabled for an sRGB texture.
    std::array<std::uint8_t, 3> srgb_border_color{};
    float lod_bias = 0.0f;
    float min_lod = 0.0f;
    float max_lod = 0.0f;
    float max_anisotropy = 1.0f;
    // TSC word-1 bits 26-30 use the guest's five-bit LodSnap conversion table;
    // host sampler creation does not consume the value.
    float trilinear_optimization = 0.0f;
    // TSC word-1 bit 25 is decoded but never consumed;
    // its polarity is not established, so the Metal mapping only accepts the
    // zero state the reached profile validated.
    bool float_coordinate_normalization = false;
    // TSC words 4-7 as IEEE-754 RGBA. The packed sRGB-border fields overlap
    // otherwise independent bits in words 2-3 rather than these float words.
    std::array<float, 4> border_color{0.0f, 0.0f, 0.0f, 0.0f};
};

using TextureDescriptorResult = std::variant<TextureDescriptor, TextureError>;
using SamplerDescriptorResult = std::variant<SamplerDescriptor, TextureError>;

// Coordinator-owned decoded pool entries. Reuse requires a tracked CPU-clean
// range, unchanged mapping identity and unchanged GPU-write authority.
class TextureDescriptorCache {
public:
    TextureDescriptorCache();
    ~TextureDescriptorCache();
    TextureDescriptorCache(const TextureDescriptorCache&) = delete;
    TextureDescriptorCache& operator=(const TextureDescriptorCache&) = delete;
    [[nodiscard]] TextureDescriptorResult texture(
        ResourceMemory& memory, std::uint64_t pool, std::uint32_t index);
    [[nodiscard]] SamplerDescriptorResult sampler(
        ResourceMemory& memory, std::uint64_t pool, std::uint32_t index);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One level of one layer of a sampled texture deswizzled into tightly packed
// rows, addressed inside LinearTextureData::bytes. Bytes hold block rows in
// row-major order with depth planes outermost for 3D levels: row_bytes equals
// block columns times bytes per block, and size covers depth times block
// rows. offset is relative to the start of bytes.
struct LinearTextureSlice {
    std::uint32_t level = 0;
    std::uint32_t layer = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::size_t offset = 0;
    std::size_t row_bytes = 0;
    std::size_t size = 0;
};

// A sampled texture deswizzled into tightly packed rows of uploaded bytes.
// width, height and depth are the base slice's texel extent. For uncompressed
// four-bytes-per-pixel formats row_bytes equals width * 4 and the first slice
// holds height rows. For block compressed formats the rows are block rows:
// row_bytes equals ceil(width / block_width) * bytes_per_block and the first
// slice holds ceil(height / block_height) raw compressed block rows in
// row-major block order. Multi-level or layered views append the remaining
// slices layer-major, then by level, each exactly slices[i].size bytes, so a
// Metal upload can address every level and layer through slices.
struct LinearTextureData {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 1;
    std::size_t row_bytes = 0;
    std::vector<std::byte> bytes;
    std::vector<LinearTextureSlice> slices;
};

using LinearTextureResult = std::variant<LinearTextureData, TextureError>;

struct ResolvedTextureBinding {
    std::uint32_t handle = 0;
    std::uint32_t texture_id = 0;
    std::uint32_t sampler_id = 0;
    std::uint64_t texture_descriptor_address = 0;
    std::uint64_t sampler_descriptor_address = 0;
    TextureDescriptor texture;
    SamplerDescriptor sampler;
    // Bytes actually read from the backing during resolution. The legacy
    // resolver reads the complete non-null backing; descriptor-only resolution
    // leaves this zero and supplies separate retained-range evidence.
    std::size_t verified_backing_bytes = 0;
    // A valid descriptor with address zero denotes an unbound sampled texture.
    // Its backend resource has no guest backing and must not retain/read VA zero.
    bool is_null = false;
};

using TextureBindingResult = std::variant<ResolvedTextureBinding, TextureError>;

[[nodiscard]] TextureDescriptorResult decode_texture_descriptor(
    std::span<const std::byte> bytes);
[[nodiscard]] SamplerDescriptorResult decode_sampler_descriptor(
    std::span<const std::byte> bytes) noexcept;

// Resolves an explicit reflected cbuf, or the handle selected by register 0x982
// when handle_cbuf is absent. Only
// independent TIC/TSC indexing (register 0x48d == 0) is supported when a
// sampler is consumed. Fetch/query-only bindings skip all TSC state and reads;
// their returned sampler descriptor and address remain default-initialized.
[[nodiscard]] TextureBindingResult resolve_texture_binding(
    const RenderStop& stop, std::size_t stage,
    std::uint32_t shader_handle_byte_offset,
    const GpuAddressSpace& memory, bool resolve_sampler = true,
    std::optional<std::uint8_t> handle_cbuf = std::nullopt) noexcept;

// Descriptor-only admission retains the complete non-null backing with its live
// mapping/backing generations and lease. This proves range validity, not payload
// readability or CPU authority. The cache owns payload reads and must either
// reuse compatible GPU authority or request an explicit incompatible-alias resolve.
// Null bindings carry an empty backing. Calls use ResourceMemory's serialized
// coordinator contract; consumers must revalidate the lease before GPU access.
struct RetainedTextureBinding {
    ResolvedTextureBinding binding;
    RetainedRange backing;
};
using RetainedTextureBindingResult =
    std::variant<RetainedTextureBinding, TextureError, RendererError>;

[[nodiscard]] RetainedTextureBindingResult resolve_texture_binding_descriptors(
    const RenderStop& stop, std::size_t stage,
    std::uint32_t shader_handle_byte_offset,
    ResourceMemory& memory, bool resolve_sampler = true,
    std::optional<std::uint8_t> handle_cbuf = std::nullopt,
    TextureDescriptorCache* descriptors = nullptr);

// Independent v3 sampler provenance consumes only the handle's TSC half.
[[nodiscard]] SamplerDescriptorResult resolve_sampler_binding(
    const RenderStop& stop, std::size_t stage, std::uint8_t handle_cbuf,
    std::uint32_t shader_handle_byte_offset, const GpuAddressSpace& memory) noexcept;
[[nodiscard]] SamplerDescriptorResult resolve_sampler_binding(
    const RenderStop& stop, std::size_t stage, std::uint8_t handle_cbuf,
    std::uint32_t shader_handle_byte_offset, ResourceMemory& memory,
    TextureDescriptorCache& descriptors);

// Rebuilds the guest storage layout of a decoded descriptor's view: the
// description calculate_texture_size validated when the descriptor was
// decoded, with the layer stride of sub-chain views restored to the full mip
// chain's stride that the guest data preserves. Forged or inconsistent
// descriptors return a structured error.
[[nodiscard]] TextureLayoutResult<TextureLayoutDescription>
texture_view_layout(const TextureDescriptor& texture);

// Reads texture.backing_extent bytes through the checked GPU address space
// and deswizzles every admitted level and layer into tightly packed uploaded
// bytes, one LinearTextureSlice per level and layer. BC and ASTC blocks are
// preserved raw; no CPU decompression runs. The descriptor must come from the
// admitted decode profile.
[[nodiscard]] LinearTextureResult read_linear_texture(
    const TextureDescriptor& texture, const GpuAddressSpace& memory);

}  // namespace armigo::graphics
