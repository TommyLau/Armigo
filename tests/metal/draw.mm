// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/platform/metal_draw.h>
#include <armigo/platform/metal_engine_renderer.h>
#include <armigo/platform/metal_color_clear.h>
#include "engine_renderer.h"
#include "engine_pipeline.h"
#include "native_pipeline_state.h"
#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include <armigo/graphics/color_clear.h>
#include <armigo/graphics/image_readback.h>
#include <armigo/graphics/texture_layout.h>
#include <armigo/memory/write_tracker.h>
#include <armigo/shader/maxwell.h>
#include <armigo/shader/graphics.h>
#include <armigo/shader_input/shader_input.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <variant>
#include <sys/mman.h>

namespace {
namespace gm = armigo::shader::graphics_msl;
namespace gx = armigo::shader::graphics;
namespace mx = armigo::shader::maxwell;
using Stage = armigo::shader_input::Stage;
using Family = mx::OpcodeFamily;
using namespace armigo::graphics;
using namespace armigo::platform;
unsigned failures=0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c); ++failures; } } while(false)
void put(mx::DecodedProgram& program, unsigned at, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) program.sph[at + i] = std::byte(value >> (i * 8));
}

mx::DecodedProgram base(Stage stage) {
    mx::DecodedProgram result;
    result.stage = stage;
    put(result, 0, armigo::shader_input::sph_stage_type_for(stage) << 10);
    return result;
}

void attribute(mx::DecodedProgram& program, unsigned component,
               bool output = false, unsigned interpolation = 2) {
    unsigned at = 0, bit = 0, value = 1;
    if (component >= 28 && component < 32) {
        at = output ? 0x35 : 0x17;
        bit = component - 24;
    } else if (program.stage == Stage::Vertex) {
        at = (output ? 0x36 : 0x18) + (component - 32) / 8;
        bit = (component - 32) % 8;
    } else {
        at = 0x18 + (component - 32) / 4;
        bit = ((component - 32) % 4) * 2;
        value = interpolation;
    }
    program.sph[at] |= std::byte(value << bit);
}

void append(mx::DecodedProgram& program, Family family, mx::InstructionPayload payload) {
    const auto i = program.instructions.size();
    program.instructions.push_back({
        static_cast<std::uint32_t>((i / 3) * 32 + (i % 3 + 1) * 8),
        0, family, {7, false}, std::move(payload)});
}

void mov(mx::DecodedProgram& program, std::uint8_t destination, float value) {
    append(program, Family::Mov32Immediate,
           mx::Mov32Immediate{{destination}, 15, std::bit_cast<std::uint32_t>(value)});
}

void load(mx::DecodedProgram& program, std::uint8_t destination, unsigned component) {
    append(program, Family::AttributeLoad,
           mx::AttributeAccess{{destination}, {255}, {255},
                               static_cast<std::uint16_t>(component * 4), 0, false});
}

void store(mx::DecodedProgram& program, std::uint8_t source, unsigned component) {
    append(program, Family::AttributeStore,
           mx::AttributeAccess{{source}, {255}, {255},
                               static_cast<std::uint16_t>(component * 4), 0, false});
}

void exit(mx::DecodedProgram& program) {
    append(program, Family::Exit, mx::Exit{15});
}

bool lower(mx::DecodedProgram decoded, std::optional<gx::Program>& output) {
    auto result = gx::lower(decoded);
    if (const auto* error = std::get_if<gx::Error>(&result)) {
        std::fprintf(stderr, "Graphics lowering failed: %s at %u.\n",
                     error->message(), error->pc);
        return false;
    }
    output = std::get<gx::Program>(std::move(result));
    return true;
}

bool make_programs(std::optional<gx::Program>& vertex,
                   std::optional<gx::Program>& fragment) {
    auto vs = base(Stage::Vertex);
    for (unsigned component = 32; component < 35; ++component) attribute(vs, component);
    for (unsigned component = 36; component < 39; ++component) attribute(vs, component);
    for (unsigned component = 28; component < 32; ++component) attribute(vs, component, true);
    attribute(vs, 32, true);
    attribute(vs, 33, true);
    load(vs, 1, 32); load(vs, 2, 33); load(vs, 3, 34);
    // The authored coordinates have an even representation. A nonidentity
    // word mask exercises raw integer emission without changing the geometry.
    append(vs, Family::WordAndImmediate, mx::WordAndImmediate{{1},{1},0xfffffffeu});
    load(vs, 4, 36); load(vs, 5, 37);
    mov(vs, 6, 2.0f);
    append(vs, Family::FloatMultiplyConstant,
           mx::FloatMultiply{{1}, {1}, mx::ConstantBufferOperand{3, 0},
                             mx::RoundingMode::NearestEven, mx::FlushMode::FlushToZero});
    store(vs, 1, 28); store(vs, 2, 29); store(vs, 3, 30); store(vs, 6, 31);
    store(vs, 4, 32); store(vs, 5, 33);
    exit(vs);

    auto fs = base(Stage::Fragment);
    attribute(fs, 31);
    attribute(fs, 32, false, 2);
    attribute(fs, 33, false, 2);
    put(fs, 0x48, 0xf);
    append(fs, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{4}, {255}, {255}, 32, 0x7ff});
    append(fs, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{5}, {255}, {255}, 33, 0x7ff});
    append(fs, Family::InterpolateAttribute,
           mx::InterpolateAttribute{{6}, {255}, {255}, 31, 0x7ff});
    append(fs, Family::MultiFunction, mx::Reciprocal{{7}, {6}});
    append(fs, Family::FloatMultiplyRegister,
           mx::FloatMultiply{{8}, {4}, mx::RegisterOperand{7},
                             mx::RoundingMode::Zero, mx::FlushMode::FlushToZero});
    append(fs, Family::FloatMultiplyRegister,
           mx::FloatMultiply{{9}, {5}, mx::RegisterOperand{7},
                             mx::RoundingMode::Zero, mx::FlushMode::FlushToZero});
    append(fs, Family::TextureSample,
           mx::TextureSample2D{{0}, {2}, {8}, {9}, 32, true});
    exit(fs);
    return lower(std::move(vs), vertex) && lower(std::move(fs), fragment);
}


// Independently authored fragment fixture exercises three comparison outcomes
// through the runtime's real descriptor decode, upload and sampler binding.
constexpr const char* depth_fragment_source = R"metal(
#include <metal_stdlib>
using namespace metal;
fragment float4 armigo_fragment_main(depth2d<float> tex [[texture(0)]],
                                    sampler state [[sampler(0)]]) {
    return float4(tex.sample_compare(state, float2(0.25f, 0.25f), 0.5f),
                  tex.sample_compare(state, float2(0.75f, 0.25f), 0.5f),
                  tex.sample_compare(state, float2(0.25f, 0.25f), 0.0f), 1.0f);
}
)metal";

struct Fixtures {
    std::shared_ptr<CompiledGraphicsStage> vertex=std::make_shared<CompiledGraphicsStage>();
    std::shared_ptr<CompiledGraphicsStage> fragment=std::make_shared<CompiledGraphicsStage>();
    std::string vertex_source, fragment_source;
    Fixtures() {
        std::optional<gx::Program> v,f;
        if (!make_programs(v,f)) throw std::runtime_error("fixture lowering failed");
        gm::EmitOptions vo,fo;
        vo.vertex_depth=gm::VertexDepthConvention::NegativeOneToOne;
        vo.vertex_attributes={{0,3,gm::VertexFormat::Float32x3},{1,7,gm::VertexFormat::Float32x3}};
        fo.fragment_position=gm::FragmentPositionConvention::MetalPositionWMatchesGuestPositionW;
        const auto vs=std::get<gm::EmittedStage>(gm::emit(*v,vo));
        const auto fs=std::get<gm::EmittedStage>(gm::emit(*f,fo));
        const auto copy=[](CompiledGraphicsStage& dest,const gm::EmittedStage& source) {
            dest.stage=source.stage; dest.entry_point=source.entry_point;
            dest.vertex_depth=source.vertex_depth; dest.constant_buffers=source.constant_buffers;
            dest.textures=source.textures; dest.interface=source.interface;
            // Legacy emission has no typed colour records: synthesize its known
            // target-0 float32x4 export like the package loader does for v1.
            if (source.stage==Stage::Fragment)
                dest.color_outputs={{0,15,ColorOutputScalar::f32,4}};
        };
        copy(*vertex,vs); copy(*fragment,fs);
        vertex->vertex_attributes=vo.vertex_attributes; fragment->fragment_position=fo.fragment_position;
        vertex_source=vs.source; fragment_source=fs.source;
        // The vertex library can also hold an independently authored fragment
        // fixture, so the existing offline compiler harness builds both paths.
        vertex_source+=R"MSL(
fragment float4 armigo_fragment_main(
    constant uint* short_cb [[buffer(29)]],
    constant uint* disabled_cb [[buffer(18)]],
    constant uint* empty_cb [[buffer(19)]],
    constant uint* sizes [[buffer(30)]]) {
    const float first = sizes[17] >= 4u ? as_type<float>(short_cb[0]) : 0.0f;
    const float second = sizes[17] >= 8u ? as_type<float>(short_cb[1]) : 0.0f;
    const float disabled = sizes[2] >= 4u ? as_type<float>(disabled_cb[0]) : 0.0f;
    const float empty = sizes[5] >= 4u ? as_type<float>(empty_cb[0]) : 0.0f;
    bool table_ok = sizes[17] == 7u && sizes[2] == 0u && sizes[5] == 0u &&
                    sizes[8] == 70000u;
    for (uint slot = 0; slot < 18; ++slot)
        if (slot != 17u && slot != 8u) table_ok = table_ok && sizes[slot] == 0u;
    // These direct reads verify that disabled/empty allocations are backed by
    // zeros even when the compiler speculates a candidate's first word load.
    const uint padding = (sizes[2] == 0u ? disabled_cb[0] : 0u) |
                         (sizes[5] == 0u ? empty_cb[0] : 0u) | short_cb[7];
    return float4(first, second, disabled + empty,
                  table_ok && padding == 0u ? 1.0f : 0.0f);
}
)MSL";
    }
    void load(const std::filesystem::path& path) {
        for (const auto& [stage,name]:{std::pair{vertex,"vertex.metallib"},std::pair{fragment,"fragment.metallib"}}) {
            std::ifstream input(path/name,std::ios::binary);
            std::vector<char> data{std::istreambuf_iterator<char>(input),{}};
            if (data.empty()) throw std::runtime_error("compiled fixture library missing");
            stage->metallib.resize(data.size()); std::memcpy(stage->metallib.data(),data.data(),data.size());
        }
    }
    static CompiledGraphicsLookupResult lookup(void* raw,const RenderStop&,const GpuAddressSpace&) {
        auto& f=*static_cast<Fixtures*>(raw); return CompiledGraphicsPair{f.vertex,f.fragment};
    }
};

// Class-typed vertex inputs: an independently authored Metal source whose
// vertex entry reads two float inputs, one signed-short input and one raw
// integer input through [[attribute(location)]]. Both class stages compile
// from the fragment library, which the fixed fixture harness already builds.
struct Backing {
    std::vector<std::byte> bytes=std::vector<std::byte>(0x10000,std::byte{0x19});
    bool fail_read=false, fail_write=false; unsigned writes=0; unsigned fail_after=~0u;
    static bool read(void* raw,std::uint64_t address,std::span<std::byte> out) noexcept {
        auto& b=*static_cast<Backing*>(raw);
        if (b.fail_read || address<0x800000 || address-0x800000>b.bytes.size() ||
            out.size()>b.bytes.size()-(address-0x800000)) return false;
        std::memcpy(out.data(),b.bytes.data()+address-0x800000,out.size()); return true;
    }
    static bool write(void* raw,std::uint64_t address,std::span<const std::byte> in) noexcept {
        auto& b=*static_cast<Backing*>(raw); ++b.writes;
        if (b.fail_write || b.writes>b.fail_after || address<0x800000 || address-0x800000>b.bytes.size() ||
            in.size()>b.bytes.size()-(address-0x800000)) return false;
        std::memcpy(b.bytes.data()+address-0x800000,in.data(),in.size()); return true;
    }
    GpuAddressSpace memory(std::shared_ptr<void> owner = {}) {
        Mapping m{0x100000,bytes.size(),0,0x800000,bytes.size(),0x1000, std::move(owner)};
        return std::get<GpuAddressSpace>(make_address_space({&m,1},{read,write,this}));
    }
    static bool direct_read(void*,std::uint64_t address,std::span<std::byte> out) noexcept {
        std::memcpy(out.data(),reinterpret_cast<const void*>(address),out.size()); return true;
    }
    static bool direct_write(void*,std::uint64_t address,std::span<const std::byte> in) noexcept {
        std::memcpy(reinterpret_cast<void*>(address),in.data(),in.size()); return true;
    }
    GpuAddressSpace host_memory(std::shared_ptr<void> owner) {
        const auto guest=reinterpret_cast<std::uint64_t>(bytes.data());
        Mapping m{0x100000,bytes.size(),0,guest,bytes.size(),0x1000,std::move(owner)};
        return std::get<GpuAddressSpace>(make_address_space({&m,1},{direct_read,direct_write,nullptr}));
    }
    void word(std::size_t offset,std::uint32_t word) {std::memcpy(bytes.data()+offset,&word,4);}
};
ColorClearPlan image(unsigned width,unsigned height) {
    ColorClearPlan p; p.width=width;p.height=height;p.layout=ColorTargetLayout::BlockLinear;
    p.row_bytes=64;p.storage_size=512;return p;
}
RenderStop snapshot(Backing& b) {
    RenderStop s; s.engine=engine_class::threed;s.kind=RenderKind::Draw;
    s.draw=DrawDescription{true,4,2,3,1,0,1};
    auto& r=s.registers;
    r[0xdf]=1;r[0x487]=1;r[0x36b]=r[0x36c]=0x1b02;r[0x4b9]=1;r[0x680]=0x1111;
    r[0x3ef]=0xffff;r[0x646]=1;r[0x647]=0x901;r[0x648]=0x405;
    r[0x64b]=1;r[0x286]=0x6420;r[0x64f]=0x181d;
    for (auto [reg,f]:{std::pair{0x280,4.f},{0x281,-4.f},{0x282,.5f},{0x283,4.f},{0x284,4.f},{0x285,.5f},{0x303,1.f}})
        r[reg]=std::bit_cast<std::uint32_t>(f);
    r[0x300]=r[0x301]=8u<<16;r[0x380]=1;r[0x381]=r[0x382]=8u<<16;
    r[0x3fd]=r[0x3fe]=8u<<16;
    s.programs[1].enabled=s.programs[5].enabled=true;
    s.color_targets[0]={0x100000,8,8,0xd6,0,1,128,0};
    r[0x5f3]=0x102000;r[0x5f5]=0x10200f;r[0x5f6]=1;
    constexpr std::array<std::uint16_t,5> indices{77,88,0,1,2};
    std::memcpy(b.bytes.data()+0x2000,indices.data(),sizeof(indices));
    constexpr std::array<float,24> vertices{
        99,99,99,99,99,99,
        -2,-2,-2,.25f,.75f,0,
         2,-2,-2,.75f,.75f,0,
        -2, 2,-2,.25f,.25f,0};
    std::memcpy(b.bytes.data()+0x1000,vertices.data(),sizeof(vertices));
    r[0x458]=(2u<<21)|(7u<<27);r[0x459]=(12u<<7)|(2u<<21)|(7u<<27);
    r[0x700]=0x1000|24;r[0x702]=0x101000;r[0x7c1]=0x101000+sizeof(vertices)-1;
    s.constant_buffers[0][3]={0x105000,4,true};b.word(0x5000,std::bit_cast<std::uint32_t>(1.f));
    s.constant_buffers[4][4]={0x105100,36,true};b.word(0x5120,0);r[0x982]=4;
    r[0x55e]=0x104000;r[0x558]=0x104100;
    std::uint32_t word0=9;
    for (unsigned i=0;i<4;++i) {word0|=2u<<(7+i*3);word0|=(i+2u)<<(19+i*3);}
    for(unsigned i=0;i<8;++i){b.word(0x4000+i*4,0);b.word(0x4100+i*4,0);}
    b.word(0x4000,word0);b.word(0x4004,0x103000);b.word(0x4008,3u<<21);
    b.word(0x4010,1u|(1u<<23));b.word(0x4014,1u|(1u<<31));
    b.word(0x4100,2u|(2u<<3)|(2u<<6));b.word(0x4104,2u|(2u<<4)|(3u<<6));
    const auto p=image(2,2);
    constexpr std::array<std::uint32_t,4> colors{0xc00003ff,0xc00ffc00,0xfff00000,0xffffffff};
    for(unsigned y=0;y<2;++y)for(unsigned x=0;x<2;++x)
        b.word(0x3000+std::get<std::size_t>(pixel_byte_offset(p,x,y)),colors[y*2+x]);
    return s;
}
unsigned srgb(float c) {
    const float encoded=c<=.0031308f?12.92f*c:1.055f*std::pow(c,1.f/2.4f)-.055f;
    return unsigned(std::lround(encoded*255));
}
// Replace the RGB10A2 descriptor and texels with a uniform two-by-two RGBA8
struct ClassFixtures {
    std::shared_ptr<CompiledGraphicsStage> vertex=std::make_shared<CompiledGraphicsStage>();
    std::shared_ptr<CompiledGraphicsStage> fragment=std::make_shared<CompiledGraphicsStage>();
    // Each class stage keeps the standard entry name and is appended to the
    // opposite fixture library, whose same-named function lives in the other
    // file, so no source defines a name twice.
    std::string class_vertex_source,class_fragment_source;
    ClassFixtures() {
        class_vertex_source=R"metal(
struct armigo_class_in {
    float4 position_input [[attribute(1)]];
};
struct armigo_class_out {
    float4 position [[position]];
    float4 color [[user(locn0)]];
};
vertex armigo_class_out armigo_class_vertex_main(armigo_class_in in [[stage_in]]) {
    const float4 position_input=in.position_input;
    armigo_class_out out;
    // The colour channels carry the fetched position so the pixel check
    // validates the Float2 fetch itself; Metal fills z and w of the float4
    // member from the two-component guest format.
    out.position=float4(position_input.xy,0.0,1.0);
    out.color=float4(position_input.x*0.25f+0.5f,position_input.y*0.25f+0.5f,0.0,1.0);
    return out;
}
)metal";
        class_fragment_source=R"metal(
struct armigo_class_fout {
    float4 c0 [[color(0)]];
};
struct armigo_class_fin {
    float4 color [[user(locn0)]];
};
fragment armigo_class_fout armigo_class_fragment_main(armigo_class_fin in [[stage_in]]) {
    return {in.color};
}
)metal";
        vertex->stage=Stage::Vertex; vertex->entry_point="armigo_class_vertex_main";
        vertex->vertex_depth=gm::VertexDepthConvention::NegativeOneToOne;
        vertex->vertex_class_inputs={{1,VertexFormatClass::float_class}};
        vertex->interface.position_output_mask=15;
        vertex->interface.generic_inputs={{1,15,gx::Interpolation::ScreenLinear}};
        vertex->interface.generic_outputs={{0,15,gx::Interpolation::ScreenLinear}};
        fragment->stage=Stage::Fragment; fragment->entry_point="armigo_class_fragment_main";
        fragment->interface.color_output_mask=15;
        fragment->color_outputs={{0,15,ColorOutputScalar::f32,4}};
        fragment->interface.generic_inputs={{0,15,gx::Interpolation::ScreenLinear}};
    }
    void load(const std::filesystem::path& path) {
        std::ifstream input(path/"fragment.metallib",std::ios::binary);
        std::vector<char> data{std::istreambuf_iterator<char>(input),{}};
        if (data.empty()) throw std::runtime_error("compiled class fixture library missing");
        for (auto* stage:{vertex.get(),fragment.get()}) {
            stage->metallib.resize(data.size());
            std::memcpy(stage->metallib.data(),data.data(),data.size());
        }
    }
    static CompiledGraphicsLookupResult lookup(void* raw,const RenderStop&,const GpuAddressSpace&) {
        auto& f=*static_cast<ClassFixtures*>(raw); return CompiledGraphicsPair{f.vertex,f.fragment};
    }
};

// Every class input sits on its own stream with a distinct stride, byte
// offset and guest format; all vertex rows carry identical values so the
// interpolated pixel color is exact.
RenderStop class_snapshot(Backing& b) {
    auto s=snapshot(b);
    auto& r=s.registers;
    // One class-typed input: the position reads R32G32Float from stream zero
    // at eight-byte rows; the shared draw starts at base vertex one over the
    // three indices, so rows one through three are the accessed range.
    r[0x458+1]=(0u<<7)|0u|(4u<<21)|(7u<<27);
    r[0x700]=0x1000u|8;
    r[0x701]=0;
    r[0x702]=0x100000u+0x1000;
    r[0x7c0]=0;
    r[0x7c1]=0x100000u+0x1000+3*8+8-1;
    r[0x620]=0;
    constexpr std::array<float,8> rows{99.f,99.f, -2.f,-2.f, 2.f,-2.f, -2.f,2.f};
    std::memcpy(b.bytes.data()+0x1000,rows.data(),sizeof(rows));
    return s;
}
// Replace the RGB10A2 descriptor and texels with a uniform two-by-two RGBA8
// image so every interpolated sample returns exactly one texel value. Alpha is
// an intermediate value that must survive both sampled views unchanged.
RenderStop rgba8_snapshot(Backing& b,bool srgb_decode) {
    auto s=snapshot(b);
    std::uint32_t word0=8;
    for(unsigned i=0;i<4;++i){word0|=2u<<(7+i*3);word0|=(i+2u)<<(19+i*3);}
    b.word(0x4000,word0);
    b.word(0x4010,1u|(1u<<23)|(srgb_decode?1u<<22:0u));
    constexpr std::uint32_t texel=64u|(128u<<8)|(192u<<16)|(128u<<24);
    const auto p=image(2,2);
    for(unsigned y=0;y<2;++y)for(unsigned x=0;x<2;++x)
        b.word(0x3000+std::get<std::size_t>(pixel_byte_offset(p,x,y)),texel);
    return s;
}
void verify_rgba8_sampling(Fixtures& fixtures) {
    const auto p=image(8,8);
    const auto sample=[&](bool srgb_decode) {
        Backing backing;
        const auto stop=rgba8_snapshot(backing,srgb_decode);
        auto memory=backing.memory();
        MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
        const auto result=execute_metal_draw(&context,stop,memory);
        if(!result.ok())std::fprintf(stderr,"Backend: %.*s\n",int(result.detail.size()),result.detail.data());
        CHECK(result.ok()&&stats.completed_draws==1);
        const auto texel=[&](unsigned x,unsigned y) {
            const auto offset=std::get<std::size_t>(pixel_byte_offset(p,x,y));
            std::array<unsigned,4> value{};
            for(unsigned lane=0;lane<4;++lane)
                value[lane]=std::to_integer<unsigned>(backing.bytes[offset+lane]);
            return value;
        };
        // (2,5) is well inside the triangle; (6,1) is outside it.
        return std::pair{texel(2,5),texel(6,1)};
    };
    const auto [linear,linear_outside]=sample(false);
    const auto [decoded,decoded_outside]=sample(true);
    constexpr std::array<unsigned,4> bytes{64,128,192,128};
    for(unsigned lane=0;lane<3;++lane) {
        CHECK(std::abs(int(linear[lane])-int(srgb(bytes[lane]/255.f)))<=2);
        CHECK(std::abs(int(decoded[lane])-int(bytes[lane]))<=2);
    }
    // Alpha is intermediate: neither sampled view nor the sRGB target may change it.
    CHECK(linear[3]==bytes[3]&&decoded[3]==bytes[3]);
    for(unsigned lane=0;lane<4;++lane)
        CHECK(linear_outside[lane]==0x19&&decoded_outside[lane]==0x19);
    // Linear sampling is sRGB-encoded once for the target; sRGB sampling
    // round-trips. Channel order must be R,G,B,A in both cases.
    CHECK(decoded[0]+16<linear[0]&&decoded[2]+16<linear[2]);
    CHECK(decoded[0]<decoded[1]&&decoded[1]<decoded[2]);
    CHECK(linear[0]<linear[1]&&linear[1]<linear[2]);
    // The packed RGB10A2 descriptor has no sRGB form and must not complete.
    Backing backing;
    const auto stop=snapshot(backing);
    backing.word(0x4010,(1u|(1u<<23))|(1u<<22));
    auto memory=backing.memory();
    MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
    const auto rejected=execute_metal_draw(&context,stop,memory);
    CHECK(!rejected.ok()&&rejected.status==CommandStatus::Unsupported);
    CHECK(stats.completed_draws==0&&backing.writes==0);
    const auto inside=std::get<std::size_t>(pixel_byte_offset(p,2,5));
    for(unsigned lane=0;lane<4;++lane)CHECK(backing.bytes[inside+lane]==std::byte{0x19});
}
void alpha_blend_state(RenderStop& stop) {
    auto& r=stop.registers;
    r[0x4d8]=1; r[0x780]=0;
    r[0x781]=r[0x784]=1;
    r[0x782]=r[0x785]=5; r[0x783]=r[0x786]=6;
}
float decode_srgb(float value) {
    return value<=0.04045f ? value/12.92f : std::pow((value+0.055f)/1.055f,2.4f);
}
void verify_alpha_blending(Fixtures& fixtures) {
    const auto p=image(8,8);
    for (const unsigned format : {0xd1u,0xd6u}) {
        const bool rgb10=format==0xd1;
        const std::uint32_t background=rgb10 ? 205u|(410u<<10)|(614u<<20)|(3u<<30)
            : 40u|(80u<<8)|(120u<<16)|(170u<<24);
        for (const bool enabled : {false,true}) for (const unsigned alpha : {0u,128u,255u}) {
            Backing backing;
            auto stop=rgba8_snapshot(backing,false);
            stop.color_targets[0].format=format;
            if (enabled) alpha_blend_state(stop);
            else stop.registers[0x781]=0xdead;  // Dormant blend factors have no effect.
            const auto sampled=image(2,2);
            for(unsigned y=0;y<2;++y)for(unsigned x=0;x<2;++x)
                backing.word(0x3000+std::get<std::size_t>(pixel_byte_offset(sampled,x,y)),
                    64u|(128u<<8)|(192u<<16)|(alpha<<24));
            std::array<bool,512> touched{};
            for(unsigned y=0;y<8;++y)for(unsigned x=0;x<8;++x) {
                const auto offset=std::get<std::size_t>(pixel_byte_offset(p,x,y));
                backing.word(offset,background);
                for(unsigned lane=0;lane<4;++lane)touched[offset+lane]=true;
            }
            const auto before=backing.bytes;
            auto memory=backing.memory();
            MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
            const auto result=execute_metal_draw(&context,stop,memory);
            if(!result.ok())std::fprintf(stderr,"Blend backend: %.*s\n",int(result.detail.size()),result.detail.data());
            CHECK(result.ok()&&stats.completed_draws==1&&stats.written_pixels==64);
            const auto read=[&](unsigned x,unsigned y) {
                std::uint32_t word=0;
                const auto offset=std::get<std::size_t>(pixel_byte_offset(p,x,y));
                std::memcpy(&word,backing.bytes.data()+offset,4);return word;
            };
            const auto actual=read(2,5);
            CHECK(read(6,1)==background);
            const float a=alpha/255.f;
            const std::array<unsigned,4> source{64,128,192,alpha};
            const std::array<unsigned,4> destination=rgb10
                ? std::array<unsigned,4>{205,410,614,3}
                : std::array<unsigned,4>{40,80,120,170};
            for(unsigned lane=0;lane<4;++lane) {
                const unsigned maximum=rgb10 ? (lane==3?3:1023) : 255;
                float old=destination[lane]/float(maximum);
                if(!rgb10&&lane<3)old=decode_srgb(old);
                const float value=source[lane]/255.f;
                const float blended=enabled ? value*a+old*(1-a) : value;
                const unsigned expected=!rgb10&&lane<3 ? srgb(blended)
                    : unsigned(std::lround(blended*maximum));
                const unsigned shift=rgb10 ? (lane==3?30:lane*10) : lane*8;
                const unsigned got=(actual>>shift)&maximum;
                if(lane==3)CHECK(got==expected);
                else CHECK(std::abs(int(got)-int(expected))<=(rgb10?1:2));
            }
            for(std::size_t i=0;i<backing.bytes.size();++i)
                if(i>=touched.size()||!touched[i])CHECK(backing.bytes[i]==before[i]);
        }
    }
    for(const unsigned reg : {0x374u,0x4d5u,0x4d7u,0x780u,0x781u,0x782u,0x783u,0x784u,0x785u,0x786u}) {
        Backing backing;auto stop=rgba8_snapshot(backing,false);alpha_blend_state(stop);
        ++stop.registers[reg];
        const auto before=backing.bytes;auto memory=backing.memory();
        MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
        const auto result=execute_metal_draw(&context,stop,memory);
        CHECK(result.status==CommandStatus::Unsupported&&stats.completed_draws==0&&
              stats.written_pixels==0&&backing.writes==0);
        CHECK(backing.bytes==before);
    }
    for(const auto control : {std::pair{0x374u,1u},std::pair{0x374u,2u},
                             std::pair{0x4d5u,1u},std::pair{0x4d7u,1u}}) {
        Backing backing;auto stop=rgba8_snapshot(backing,false);
        stop.registers[control.first]=control.second;
        const auto before=backing.bytes;auto memory=backing.memory();
        MetalDrawStatistics stats;stats.completed_draws=7;stats.written_pixels=99;
        MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
        const auto result=execute_metal_draw(&context,stop,memory);
        CHECK(result.status==CommandStatus::Unsupported&&stats.completed_draws==7&&
              stats.written_pixels==99&&backing.writes==0);
        CHECK(backing.bytes==before);
    }
}
// An 8x8 BC1 image whose four 4x4 blocks separate the observable channels:
// the covered texel of block (0,0) selects BC1 transparent black, block (0,1)
// samples uniformly as c0 and block (1,1) uniformly as c1. Identity UVs put
// every rendered pixel center on a texel center so samples are pure texels.
RenderStop bc1_snapshot(Backing& b,bool srgb_decode,std::uint16_t c0,std::uint16_t c1) {
    auto s=snapshot(b);
    std::uint32_t word0=0x24;
    for(unsigned i=0;i<4;++i){word0|=2u<<(7+i*3);word0|=(i+2u)<<(19+i*3);}
    b.word(0x4000,word0);
    b.word(0x4010,7u|(1u<<23)|(srgb_decode?1u<<22:0u));
    b.word(0x4014,7u|(1u<<31));
    // The position output carries w=2, so the triangle spans the 8-pixel
    // target exactly; natural UVs put pixel centers on texel centers.
    constexpr std::array<float,24> vertices{
        99,99,99,99,99,99,
        -2,-2,-2,0,1,0,
         2,-2,-2,1,1,0,
        -2, 2,-2,0,0,0};
    std::memcpy(b.bytes.data()+0x1000,vertices.data(),sizeof(vertices));
    // Two blocks per row are two 4-byte units per GOB coordinate: block
    // (column,row) sits at unit coordinate (2*column,row).
    const ColorClearPlan plan{.width=4,.height=2,.row_bytes=64,.storage_size=512,
                              .layout=ColorTargetLayout::BlockLinear};
    const auto store=[&](unsigned column,unsigned row,std::uint32_t indices) {
        std::array<std::byte,8> block{};
        for(unsigned i=0;i<2;++i){block[i]=std::byte(c0>>(i*8));block[2+i]=std::byte(c1>>(i*8));}
        for(unsigned i=0;i<4;++i)block[4+i]=std::byte(indices>>(i*8));
        const auto offset=std::get<std::size_t>(pixel_byte_offset(plan,column*2,row));
        std::memcpy(b.bytes.data()+0x3000+offset,block.data(),8);};
    store(0,0,3u<<16);       // texel (0,2): index 3, transparent black in 3-color mode
    store(1,0,0);            // outside the triangle
    store(0,1,0);            // uniform c0
    store(1,1,0x55555555u);  // uniform c1
    return s;
}
void verify_bc1_sampling(Fixtures& fixtures) {
    const auto p=image(8,8);
    const auto run=[&](bool srgb_decode,std::uint16_t c1) {
        Backing backing;
        const auto stop=bc1_snapshot(backing,srgb_decode,0x001f,c1);
        auto memory=backing.memory();
        MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
        const auto result=execute_metal_draw(&context,stop,memory);
        if(!result.ok())std::fprintf(stderr,"Backend: %.*s\n",int(result.detail.size()),result.detail.data());
        CHECK(result.ok()&&stats.completed_draws==1);
        const auto texel=[&](unsigned x,unsigned y) {
            const auto offset=std::get<std::size_t>(pixel_byte_offset(p,x,y));
            std::array<unsigned,4> value{};
            for(unsigned lane=0;lane<4;++lane)
                value[lane]=std::to_integer<unsigned>(backing.bytes[offset+lane]);
            return value;
        };
        return std::array{texel(0,2),texel(2,5),texel(4,6),texel(6,1)};
    };
    const auto close=[&](unsigned got,unsigned expected) {
        return std::abs(int(got)-int(expected))<=2;
    };
    // c0 is 5-bit blue; c1 is either 5-bit mid red (16/31) or full green.
    // The c0<c1 ordering selects BC1 3-color mode with transparent black.
    constexpr std::uint16_t mid_red=0x8000,full_green=0x07e0;
    const auto unorm=run(false,mid_red);
    const auto srgb_view=run(true,mid_red);
    const unsigned mid_encoded=srgb(16.f/31.f);
    const unsigned mid_roundtrip=unsigned(std::lround(255.0*16.0/31.0));
    for(unsigned lane=0;lane<4;++lane) {
        // (0,2) samples the transparent texel in both sampled views.
        CHECK(unorm[0][lane]==0);CHECK(srgb_view[0][lane]==0);
    }
    for(const auto* view:{&unorm,&srgb_view}) {
        // (2,5) samples c0: blue fills lanes 2 and 3 only.
        CHECK(close((*view)[1][0],0)&&close((*view)[1][1],0));
        CHECK(close((*view)[1][2],255)&&close((*view)[1][3],255));
    }
    // (4,6) samples c1: mid red reaches the sRGB target differently per view.
    CHECK(close(unorm[2][0],mid_encoded)&&close(unorm[2][1],0)&&close(unorm[2][2],0));
    CHECK(close(srgb_view[2][0],mid_roundtrip)&&close(srgb_view[2][1],0)&&close(srgb_view[2][2],0));
    CHECK(unorm[2][0]>=srgb_view[2][0]+32);
    for(const auto* view:{&unorm,&srgb_view}) CHECK((*view)[2][3]==255);
    for(unsigned lane=0;lane<4;++lane) {
        CHECK(unorm[3][lane]==0x19);CHECK(srgb_view[3][lane]==0x19);
    }
    // Channel order must not confuse red with blue: the c1 lane ordering in
    // the mid-red run is opposite to the blue quadrant.
    CHECK(unorm[1][2]>unorm[1][0]+64&&unorm[2][0]>unorm[2][2]+64);
    const auto green_unorm=run(false,full_green);
    const auto green_srgb=run(true,full_green);
    for(const auto* view:{&green_unorm,&green_srgb}) {
        // (4,6) samples full green in lane 1 only; blue stays in lane 2.
        CHECK(close((*view)[2][0],0)&&close((*view)[2][1],255)&&close((*view)[2][2],0));
        CHECK((*view)[2][3]==255);
        for(unsigned lane=0;lane<4;++lane)CHECK((*view)[3][lane]==0x19);
    }
    CHECK(green_unorm[1][2]>green_unorm[1][0]+64);
    for(unsigned lane=0;lane<4;++lane) {
        CHECK(green_unorm[0][lane]==0);CHECK(green_srgb[0][lane]==0);
    }
}
// The portable readback tests use coordinate-coded CPU storage. This case feeds
// the helper the bytes an actual completed Metal draw wrote and checks that the
// selected crop carries the colored pixels rather than loaded background.
void verify_readback(Fixtures& fixtures) {
    Backing backing;
    const auto stop=snapshot(backing);
    auto memory=backing.memory();
    MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
    // execute_metal_draw returns only after command completion and guest
    // writeback, so the backing below holds the finished target.
    const auto result=execute_metal_draw(&context,stop,memory);
    if(!result.ok())std::fprintf(stderr,"Backend: %.*s\n",int(result.detail.size()),result.detail.data());
    CHECK(result.ok()&&stats.completed_draws==1);
    const auto plan=image(8,8);
    const auto before=backing.bytes;
    // Nonzero-origin, non-square crop: columns 1..4 and rows 2..6.
    constexpr std::array<std::int32_t,4> crop{1,2,5,7};
    const auto readback=readback_rgba8(plan,std::span<const std::byte>(backing.bytes),crop);
    const auto* readback_image=std::get_if<Rgba8Image>(&readback);
    if(readback_image==nullptr) {
        if(const auto* error=std::get_if<ImageReadbackError>(&readback))
            std::fprintf(stderr,"Readback: %.*s\n",int(error->detail.size()),error->detail.data());
        CHECK(readback_image!=nullptr);
        return;
    }
    constexpr unsigned width=4,height=5;
    CHECK(readback_image->width==width&&readback_image->height==height);
    CHECK(readback_image->pixels.size()==std::size_t{width}*height*4);
    if(readback_image->pixels.size()!=std::size_t{width}*height*4) return;
    unsigned covered=0,outside=0;
    for(unsigned row=0;row<height;++row)for(unsigned col=0;col<width;++col) {
        // Expectations are indexed by output position, but evaluated at the
        // source target coordinate this crop selects.
        const unsigned x=1+col,y=2+row;
        const auto* pixel=readback_image->pixels.data()+(std::size_t{row}*width+col)*4;
        if(y>x+1) {
            const float tx=(float(x)+.5f)/8,ty=(float(y)+.5f)/8;
            const std::array expected{srgb((1-tx)*(1-ty)+tx*ty),srgb(tx),srgb(ty),255u};
            for(unsigned lane=0;lane<3;++lane)
                CHECK(std::abs(int(std::to_integer<unsigned>(pixel[lane]))-int(expected[lane]))<=2);
            CHECK(pixel[3]==std::byte{255});
            ++covered;
        } else if(x>y+1) {
            for(unsigned lane=0;lane<4;++lane)CHECK(pixel[lane]==std::byte{0x19});
            ++outside;
        }
    }
    // The crop must include both covered and confidently uncovered pixels, and
    // reading it must leave the completed target untouched.
    CHECK(covered>=8&&outside>=1);
    CHECK(backing.bytes==before);
}
// A consumed texture_handle_cbuf fact is a draw-time contract: bound handles
// resolve through the live selector register, so a stage whose recorded slot
// register 0x982 does not select must stop before any GPU work or guest
// writeback, with a structured diagnostic naming the fact.
void verify_texture_handle_fact(Fixtures& fixtures) {
    const auto run=[&](bool vertex_recorded,bool fragment_recorded,unsigned recorded) {
        Backing backing;
        auto stop=snapshot(backing);  // The snapshot selects handle buffer 4 via r[0x982].
        auto memory=backing.memory();
        MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
        if (vertex_recorded)fixtures.vertex->texture_handle_cbuf=recorded;
        if (fragment_recorded)fixtures.fragment->texture_handle_cbuf=recorded;
        const auto result=execute_metal_draw(&context,stop,memory);
        const auto completed=stats.completed_draws;
        if (vertex_recorded)fixtures.vertex->texture_handle_cbuf.reset();
        if (fragment_recorded)fixtures.fragment->texture_handle_cbuf.reset();
        return std::pair{result,completed};
    };
    // A recorded slot matching the selector register draws and completes.
    const auto [matched,matched_completed]=run(false,true,4);
    if(!matched.ok())std::fprintf(stderr,"Backend: %.*s\n",int(matched.detail.size()),matched.detail.data());
    CHECK(matched.ok()&&matched_completed==1);
    // A recorded slot the register does not select is a structured stop naming
    // the fact, on either stage.
    for (const auto [vertex_recorded,fragment_recorded]:
            {std::pair{false,true},std::pair{true,false}}) {
        const auto [mismatched,completed]=run(vertex_recorded,fragment_recorded,2);
        CHECK(mismatched.status==CommandStatus::Unsupported&&completed==0&&
              mismatched.detail.find("texture_handle_cbuf")!=std::string_view::npos);
    }
}
// Guest layer stride of the snapshot's two-by-two, single-level block-linear
// image, computed from the same layout layer the descriptor decoder uses.
std::size_t array_layer_stride() {
    TextureLayoutDescription description{};
    description.width=2;description.height=2;description.depth=1;
    description.levels=1;description.layers=2;
    description.block_width=1;description.block_height=1;description.bytes_per_block=4;
    description.gob_block_height_log2=0;description.gob_block_depth_log2=0;
    description.storage=TextureStorageLayout::BlockLinear;
    return std::get<TextureLayoutDescription>(calculate_texture_size(description)).layer_stride;
}
// Consumed texture_descriptor facts are draw-time contracts verified against
// the live TIC/TSC before any upload or sampling: matching descriptors draw,
// while a target or format the derived shape does not describe stops the draw
// with a structured error naming the fact and handle.
void verify_texture_descriptor_facts(Fixtures& fixtures) {
    const DerivedTextureFact fact{4,32,true,false};
    const auto run=[&](bool record,const auto& mutate) {
        Backing backing;auto stop=snapshot(backing);mutate(backing);
        auto memory=backing.memory();
        if(record)fixtures.fragment->texture_facts.push_back(fact);
        MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
        const auto result=execute_metal_draw(&context,stop,memory);
        if(record)fixtures.fragment->texture_facts.clear();
        return std::tuple{result,stats.completed_draws,backing.writes};
    };
    // The live TIC is a plain 2D unorm texture and matches the fact: draw.
    {
        const auto [result,completed,writes]=run(true,[](Backing&){});
        if(!result.ok())std::fprintf(stderr,"Backend: %.*s\n",int(result.detail.size()),result.detail.data());
        CHECK(result.ok()&&completed==1&&writes>=1);
    }
    // A live 2D-array target now admits the derived 2D shape at the fact
    // level; the legacy encode still binds plain 2D textures only, so the
    // draw stops on that capability rule without a fact mismatch.
    {
        const auto [result,completed,writes]=run(true,[](Backing& b) {
            b.word(0x4010,1u|(5u<<23));});
        CHECK(result.status==CommandStatus::Unsupported&&completed==0&&writes==0&&
              result.detail.find("texture_descriptor")==std::string_view::npos&&
              result.detail.find("plain 2D")!=std::string_view::npos);
    }
    // A null 2D-array TIC still rejects normalized-coordinate and comparison
    // mismatches: null storage does not relax shape, coordinate or
    // sampler-comparison facts.
    for (unsigned mutation = 0; mutation < 2; ++mutation) {
        const auto [result, completed, writes] = run(true, [mutation](Backing& b) {
            b.word(0x4004, 0);
            if (mutation == 0) {
                b.word(0x4010, 1u | (7u << 23));
                b.word(0x4014, 1u);
            }
            else b.word(0x4100, 2u | (2u << 3) | (2u << 6) | (1u << 9) | (3u << 10));
        });
        const std::array<std::string_view, 2> diagnostics{
            "normalized-coordinate", "depth-compare"};
        CHECK(result.status == CommandStatus::Unsupported && completed == 0 && writes == 0);
        CHECK(result.detail.find("texture_descriptor") != std::string_view::npos);
        if (result.detail.find(diagnostics[mutation]) == std::string_view::npos)
            std::fprintf(stderr, "Null texture rejection %u: %.*s\n", mutation,
                int(result.detail.size()), result.detail.data());
        CHECK(result.detail.find(diagnostics[mutation]) != std::string_view::npos);
    }
    // The same snapshot without a recorded fact stops on the backend's plain
    // 2D-only binding rule instead.
    {
        const auto [result,completed,writes]=run(false,[](Backing& b) {
            b.word(0x4010,1u|(5u<<23));});
        CHECK(result.status==CommandStatus::Unsupported&&completed==0&&writes==0&&
              result.detail.find("texture_descriptor")==std::string_view::npos);
    }
    // The live decode admits an integer RGBA8 TIC, but the derived fact's
    // float-sampled component class stops the draw before sampling.
    {
        const auto [result,completed,writes]=run(true,[](Backing& b) {
            const std::uint32_t word0=8u|(4u<<7)|(4u<<10)|(4u<<13)|(4u<<16)|
                                      (2u<<19)|(3u<<22)|(4u<<25)|(5u<<28);
            b.word(0x4000,word0);});
        CHECK(result.status==CommandStatus::Unsupported&&completed==0&&writes==0&&
              result.detail.find("texture_descriptor")!=std::string_view::npos&&
              result.detail.find("sampled scalar")!=std::string_view::npos);
    }
    // Legacy reflection cannot safely bind integer shader types.
    {
        const auto [result,completed,writes]=run(false,[](Backing& b) {
            b.word(0x4000,8u|(4u<<7)|(4u<<10)|(4u<<13)|(4u<<16)|
                (2u<<19)|(3u<<22)|(4u<<25)|(5u<<28));});
        CHECK(result.status==CommandStatus::Unsupported&&completed==0&&writes==0&&
              result.detail.find("integer texture types")!=std::string_view::npos);
    }
    // Samplerless fetch/query resources belong to the persistent backend.
    {
        const auto saved=fixtures.fragment->textures.front().sampler_index;
        fixtures.fragment->textures.front().sampler_index=armigo::shader::graphics_msl::TextureBinding::no_sampler;
        const auto [result,completed,writes]=run(false,[](Backing&){});
        fixtures.fragment->textures.front().sampler_index=saved;
        CHECK(result.status==CommandStatus::Unsupported&&completed==0&&writes==0&&
              result.detail.find("samplerless texture types")!=std::string_view::npos);
    }
    // A recorded depth-compare expectation never matches the admitted
    // non-compare sampler state.
    {
        Backing backing;auto stop=snapshot(backing);auto memory=backing.memory();
        fixtures.fragment->texture_facts.push_back({4,32,true,true});
        MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
        const auto result=execute_metal_draw(&context,stop,memory);
        fixtures.fragment->texture_facts.clear();
        CHECK(result.status==CommandStatus::Unsupported&&stats.completed_draws==0&&
              backing.writes==0&&result.detail.find("depth-compare")!=std::string_view::npos);
    }
}
void verify_depth_sampling(Fixtures& fixtures, const std::filesystem::path& path) {
    const auto ordinary_library = fixtures.fragment->metallib;
    std::ifstream input(path / "depth_fragment.metallib", std::ios::binary);
    const std::vector<char> library{std::istreambuf_iterator<char>(input), {}};
    CHECK(!library.empty());
    if (library.empty()) return;
    fixtures.fragment->metallib.resize(library.size());
    std::memcpy(fixtures.fragment->metallib.data(), library.data(), library.size());
    fixtures.fragment->texture_facts = {{4,32,true,true}};
    fixtures.fragment->texture_facts.front().component_format = TextureComponentFormat::depth;
    // Never, Less, Equal, LessEqual, Greater, NotEqual, GreaterEqual, Always.
    const std::array<std::array<unsigned,3>,8> expected{{
        {0,0,0}, {0,255,0}, {0,0,255}, {0,255,255},
        {255,0,0}, {255,255,0}, {255,0,255}, {255,255,255}}};
    const std::array<unsigned,6> formats{0x3a,0x29,0x2f,0x2a,0x2b,0x30};
    const auto prepare = [](Backing& backing, unsigned format, unsigned op) {
        auto stop = snapshot(backing);
        const unsigned type = format == 0x29 ? 4 : (format == 0x2f || format == 0x30 ? 7 : 2);
        const unsigned channel = format == 0x29 ? 3 : 2;
        const unsigned green = format == 0x29 ? 2 : 4;
        const unsigned blue_alpha = format == 0x30 ? 2 : 4;
        backing.word(0x4000, format | (type<<7) | (green<<10) | (blue_alpha<<13) | (blue_alpha<<16) |
            (channel<<19) | (channel<<22) | (channel<<25) | (7u<<28));
        backing.word(0x4100, 2u|(2u<<3)|(2u<<6)|(1u<<9)|(op<<10));
        const unsigned bpp = format == 0x3a ? 2 : (format == 0x30 ? 8 : 4);
        for (unsigned y=0;y<2;++y) for (unsigned x=0;x<2;++x) {
            const auto offset=0x3000+y*16+x*bpp;
            if (format == 0x3a) {
                const std::uint16_t depth=x?0xffff:0;
                std::memcpy(backing.bytes.data()+offset,&depth,sizeof(depth));
            } else if (format == 0x2f || format == 0x30) {
                backing.word(offset,std::bit_cast<std::uint32_t>(x?1.0f:0.0f));
                if (format == 0x30) backing.word(offset+4,0x5a);
            } else backing.word(offset,(x?0xffffff00u:0u)|0x5au);
        }
        return stop;
    };
    for (const unsigned format : formats) for (unsigned op=0;op<8;++op) {
        Backing backing; const auto stop=prepare(backing,format,op); auto memory=backing.memory();
        MetalDrawStatistics stats; MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
        const auto result=execute_metal_draw(&context,stop,memory);
        if (!result.ok()) std::fprintf(stderr,"Depth format %x op %u: %.*s\n",format,op,
            int(result.detail.size()),result.detail.data());
        CHECK(result.ok() && stats.completed_draws==1);
        if (!result.ok()) continue;
        const auto p=image(8,8);
        for (unsigned y=0;y<8;++y) for (unsigned x=0;x<8;++x) if (y>x+1) {
            const auto offset=std::get<std::size_t>(pixel_byte_offset(p,x,y));
            for (unsigned lane=0;lane<3;++lane)
                CHECK(std::to_integer<unsigned>(backing.bytes[offset+lane])==expected[op][lane]);
            CHECK(backing.bytes[offset+3]==std::byte{255});
        }
    }
    // Null depth storage is zero: LessEqual rejects reference 0.5 and accepts 0.
    {
        Backing backing;
        const auto stop = prepare(backing, 0x2f, 3);
        backing.word(0x4004, 0);
        auto memory = backing.memory();
        MetalDrawStatistics stats;
        MetalDrawContext context{Fixtures::lookup, &fixtures, &stats};
        const auto result = execute_metal_draw(&context, stop, memory);
        if (!result.ok()) std::fprintf(stderr, "Null depth sample: %.*s\n",
            int(result.detail.size()), result.detail.data());
        CHECK(result.ok() && stats.completed_draws == 1);
        const auto offset = std::get<std::size_t>(pixel_byte_offset(image(8, 8), 2, 5));
        const std::array expected_null{std::byte{0}, std::byte{0}, std::byte{255}, std::byte{255}};
        CHECK(std::equal(expected_null.begin(), expected_null.end(), backing.bytes.begin() + offset));
    }
    // Every rejected descriptor must stop before target writeback; the last
    // case is admitted.
    for (unsigned mutation=0;mutation<4;++mutation) {
        Backing backing; const auto stop=prepare(backing,0x29,1); auto memory=backing.memory();
        if (mutation==0) { // Comparison against a live color format.
            backing.word(0x4000,8u|(2u<<7)|(2u<<10)|(2u<<13)|(2u<<16)|
                (2u<<19)|(3u<<22)|(4u<<25)|(5u<<28));
        } else if (mutation==1) { // Depth sampler comparison disabled.
            backing.word(0x4100,2u|(2u<<3)|(2u<<6)|(1u<<10));
        } else if (mutation==2) { // D24S8 R selects stencil, even on depth storage.
            backing.word(0x4000,0x29u|(4u<<7)|(2u<<10)|(4u<<13)|(4u<<16)|
                (2u<<19)|(2u<<22)|(2u<<25)|(7u<<28));
        } else { // A non-comparison float fact admits depth storage (depth in R).
            fixtures.fragment->texture_facts.front().comparison=false;
            fixtures.fragment->texture_facts.front().component_format=TextureComponentFormat::floating;
            backing.word(0x4100,2u|(2u<<3)|(2u<<6));
        }
        MetalDrawStatistics stats; MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
        const auto result=execute_metal_draw(&context,stop,memory);
        if (mutation==3) {
            CHECK(result.detail.find("float-sampled")==std::string_view::npos);
        } else {
            CHECK(result.status==CommandStatus::Unsupported && stats.completed_draws==0 && backing.writes==0);
            CHECK(result.detail.find("texture_descriptor")!=std::string_view::npos);
        }
        fixtures.fragment->texture_facts.front().comparison=true;
        fixtures.fragment->texture_facts.front().component_format=TextureComponentFormat::depth;
    }
    fixtures.fragment->texture_facts.clear();
    fixtures.fragment->metallib=ordinary_library;
}

void verify(Fixtures& fixtures) {
    Backing backing;auto stop=snapshot(backing);auto memory=backing.memory();
    const auto before=backing.bytes;
    MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
    auto result=execute_metal_draw(&context,stop,memory);
    if (!result.ok())std::fprintf(stderr,"Backend: %.*s\n",int(result.detail.size()),result.detail.data());
    CHECK(result.ok());CHECK(stats.completed_draws==1);CHECK(stats.written_pixels==64);

    // Inline indices are owned by the stopped draw and must never fall back to
    // an unrelated legacy index-buffer range.
    Backing inline_backing;
    auto inline_stop=snapshot(inline_backing);
    inline_stop.inline_indices={0,1,2};
    std::fill_n(inline_backing.bytes.begin()+0x2004,6,std::byte{0});
    auto inline_memory=inline_backing.memory();
    MetalDrawStatistics inline_stats;
    MetalDrawContext inline_context{Fixtures::lookup,&fixtures,&inline_stats};
    const auto inline_result=execute_metal_draw(&inline_context,inline_stop,inline_memory);
    CHECK(inline_result.status==CommandStatus::Unsupported);
    CHECK(inline_result.detail=="Legacy Metal draw does not support inline indices");
    CHECK(inline_stats.completed_draws==0&&inline_backing.writes==0);

    const auto p=image(8,8);std::vector<bool> texels(512);
    for(unsigned y=0;y<8;++y)for(unsigned x=0;x<8;++x) {
        const auto offset=std::get<std::size_t>(pixel_byte_offset(p,x,y));
        for(unsigned i=0;i<4;++i)texels[offset+i]=true;
        // Stay off the diagonal coverage boundary; a partial triangle must load old pixels.
        if (y>x+1) {
            const float tx=(float(x)+.5f)/8,ty=(float(y)+.5f)/8;
            const std::array expected{srgb((1-tx)*(1-ty)+tx*ty),srgb(tx),srgb(ty),255u};
            for(unsigned lane=0;lane<4;++lane)
                CHECK(std::abs(int(std::to_integer<unsigned>(backing.bytes[offset+lane]))-int(expected[lane]))<=2);
        } else if (x>y+1) for(unsigned i=0;i<4;++i)CHECK(backing.bytes[offset+i]==std::byte{0x19});
    }
    for(std::size_t i=0;i<512;++i)if(!texels[i])CHECK(backing.bytes[i]==before[i]);
    CHECK(std::equal(backing.bytes.begin()+512,backing.bytes.end(),before.begin()+512));
    // Scissoring must preserve pixels that are geometrically covered but excluded.
    backing.bytes=before;
    auto scissored=stop;scissored.registers[0x381]=(6u<<16)|2;scissored.registers[0x382]=(6u<<16)|2;
    CHECK(execute_metal_draw(&context,scissored,memory).ok());
    for(unsigned y=0;y<8;++y)for(unsigned x=0;x<8;++x)
        if(x<2 || x>=6 || y<2 || y>=6) {
            const auto offset=std::get<std::size_t>(pixel_byte_offset(p,x,y));
            for(unsigned lane=0;lane<4;++lane)CHECK(backing.bytes[offset+lane]==before[offset+lane]);
        }
    const auto inside=std::get<std::size_t>(pixel_byte_offset(p,2,5));
    CHECK(backing.bytes[inside]!=before[inside]);
    // A smaller viewport clip and matching transform use only that part of
    // the larger target; the remaining pixels retain their loaded values.
    backing.bytes=before;
    auto subregion=stop;
    subregion.registers[0x300]=subregion.registers[0x301]=4u<<16;
    subregion.registers[0x280]=subregion.registers[0x283]=std::bit_cast<std::uint32_t>(2.f);
    subregion.registers[0x281]=std::bit_cast<std::uint32_t>(-2.f);
    subregion.registers[0x284]=std::bit_cast<std::uint32_t>(2.f);
    CHECK(execute_metal_draw(&context,subregion,memory).ok());
    for(unsigned y=0;y<8;++y)for(unsigned x=0;x<8;++x) {
        const auto offset=std::get<std::size_t>(pixel_byte_offset(p,x,y));
        if(x>=4 || y>=4)
            for(unsigned lane=0;lane<4;++lane)CHECK(backing.bytes[offset+lane]==before[offset+lane]);
        else if(y>x+1) {
            const float tx=(float(x)+.5f)/4,ty=(float(y)+.5f)/4;
            const std::array expected{srgb((1-tx)*(1-ty)+tx*ty),srgb(tx),srgb(ty),255u};
            for(unsigned lane=0;lane<4;++lane)
                CHECK(std::abs(int(std::to_integer<unsigned>(backing.bytes[offset+lane]))-int(expected[lane]))<=2);
        }
    }
    // Reversing the winding must cull the oriented triangle.
    backing.bytes=before;std::swap_ranges(backing.bytes.begin()+0x2006,backing.bytes.begin()+0x2008,backing.bytes.begin()+0x2008);
    CHECK(execute_metal_draw(&context,stop,memory).ok());
    CHECK(std::equal(backing.bytes.begin(),backing.bytes.begin()+512,before.begin()));
    // A live constant-buffer change collapses the triangle, independently of attributes.
    backing.bytes=before;backing.word(0x5000,0);
    CHECK(execute_metal_draw(&context,stop,memory).ok());
    CHECK(std::equal(backing.bytes.begin(),backing.bytes.begin()+512,before.begin()));
    backing.bytes=before;const auto completed=stats.completed_draws;backing.writes=0;
    CHECK(!execute_metal_draw(nullptr,stop,memory).ok());
    MetalDrawContext missing;CHECK(!execute_metal_draw(&missing,stop,memory).ok());
    fixtures.vertex->vertex_depth=gm::VertexDepthConvention::ZeroToOne;
    CHECK(!execute_metal_draw(&context,stop,memory).ok());fixtures.vertex->vertex_depth=gm::VertexDepthConvention::NegativeOneToOne;
    const auto vertex_name=fixtures.vertex->entry_point;
    fixtures.vertex->entry_point="unexpected_vertex";CHECK(!execute_metal_draw(&context,stop,memory).ok());
    fixtures.vertex->entry_point=vertex_name;
    const auto fragment_name=fixtures.fragment->entry_point;
    fixtures.fragment->entry_point="unexpected_fragment";CHECK(!execute_metal_draw(&context,stop,memory).ok());
    fixtures.fragment->entry_point=fragment_name;
    const auto cb_size=fixtures.vertex->constant_buffers[0].minimum_size;
    fixtures.vertex->constant_buffers[0].minimum_size=1;CHECK(!execute_metal_draw(&context,stop,memory).ok());
    fixtures.vertex->constant_buffers[0].minimum_size=cb_size;
    auto saved_library=std::move(fixtures.vertex->metallib);
    CHECK(!execute_metal_draw(&context,stop,memory).ok());fixtures.vertex->metallib=std::move(saved_library);
    const auto old_mask=fixtures.fragment->interface.generic_inputs[0].component_mask;
    fixtures.fragment->interface.generic_inputs[0].component_mask=15;
    CHECK(!execute_metal_draw(&context,stop,memory).ok());fixtures.fragment->interface.generic_inputs[0].component_mask=old_mask;
    auto changed=stop;changed.registers[0x286]=0;CHECK(!execute_metal_draw(&context,changed,memory).ok());
    changed=stop;changed.registers[0x300]|=1;CHECK(!execute_metal_draw(&context,changed,memory).ok());
    changed=stop;changed.registers[0x301]=7u<<16;CHECK(!execute_metal_draw(&context,changed,memory).ok());
    changed=stop;changed.registers[0x4d8]=1;CHECK(!execute_metal_draw(&context,changed,memory).ok());
    changed=stop;changed.draw->first=100;CHECK(!execute_metal_draw(&context,changed,memory).ok());
    changed=stop;changed.draw->base_vertex=-1;CHECK(!execute_metal_draw(&context,changed,memory).ok());
    changed=stop;changed.constant_buffers[0][3].size=3;CHECK(!execute_metal_draw(&context,changed,memory).ok());
    backing.fail_read=true;CHECK(execute_metal_draw(&context,stop,memory).status==CommandStatus::MemoryFault);backing.fail_read=false;
    CHECK(backing.writes==0 && stats.completed_draws==completed && backing.bytes==before);
    backing.fail_write=true;CHECK(execute_metal_draw(&context,stop,memory).status==CommandStatus::MemoryFault);
    CHECK(stats.completed_draws==completed && backing.bytes==before);
    // A late write fault exposes an already-written prefix but never completion.
    Backing late;auto late_stop=snapshot(late);auto late_memory=late.memory();
    const auto late_before=late.bytes;late.fail_after=4;
    MetalDrawStatistics late_stats;MetalDrawContext late_context{Fixtures::lookup,&fixtures,&late_stats};
    const auto late_result=execute_metal_draw(&late_context,late_stop,late_memory);
    CHECK(late_result.status==CommandStatus::MemoryFault && late.writes==5);
    CHECK(late_stats.completed_draws==0 && late_stats.written_pixels==0);
    const auto visible=std::get<std::size_t>(pixel_byte_offset(p,0,1));
    CHECK(late.bytes[visible]!=late_before[visible]);
    for(std::size_t i=0;i<512;++i)if(!texels[i])CHECK(late.bytes[i]==late_before[i]);
}
// A draw whose vertex inputs are typed by reflection class: the descriptor is
// built from the guest attribute registers, converting Float2, RGBA8 unorm,
// R16G16 snorm and RGBA8 uint on four distinct streams; Metal's missing
// component fill must supply the position w.
void verify_class_inputs(ClassFixtures& fixtures) {
    const auto p=image(8,8);
    Backing backing;
    const auto stop=class_snapshot(backing);
    auto memory=backing.memory();
    MetalDrawStatistics stats;MetalDrawContext context{ClassFixtures::lookup,&fixtures,&stats};
    const auto result=execute_metal_draw(&context,stop,memory);
    if(!result.ok())std::fprintf(stderr,"Class backend: %.*s\n",int(result.detail.size()),result.detail.data());
    CHECK(result.ok()&&stats.completed_draws==1&&stats.written_pixels==64);
    const auto texel=[&](unsigned x,unsigned y) {
        const auto offset=std::get<std::size_t>(pixel_byte_offset(p,x,y));
        std::array<unsigned,4> value{};
        for(unsigned lane=0;lane<4;++lane)
            value[lane]=std::to_integer<unsigned>(backing.bytes[offset+lane]);
        return value;
    };
    const auto inside=texel(2,5), outside=texel(6,1);
    // Pixel (2,5) centres on NDC (-0.375,-0.375) inside the clipped triangle
    // (-1,-1),(1,-1),(-1,1); with unit clip w both colour channels of the
    // fetched position interpolate affinely to -0.375.
    const std::array<unsigned,4> expected{
        srgb(-0.375f*0.25f+0.5f), srgb(-0.375f*0.25f+0.5f), 0u, 255u};
    for(unsigned lane=0;lane<4;++lane) {
        CHECK(std::abs(int(inside[lane])-int(expected[lane]))<=2);
        CHECK(outside[lane]==0x19);
    }
    // A cached pipeline must reproduce the same frame.
    Backing again;
    const auto repeat=class_snapshot(again);
    auto repeat_memory=again.memory();
    MetalDrawStatistics repeat_stats;MetalDrawContext repeat_context{ClassFixtures::lookup,&fixtures,&repeat_stats};
    CHECK(execute_metal_draw(&repeat_context,repeat,repeat_memory).ok());
    CHECK(repeat_stats.completed_draws==1);
    const auto inside_again=texel(2,5);
    for(unsigned lane=0;lane<4;++lane)
        CHECK(std::abs(int(inside_again[lane])-int(expected[lane]))<=2);
    // A float guest format feeding a uint-typed input is a structured stop.
    Backing mismatch;
    auto mismatch_memory=mismatch.memory();
    auto wrong=class_snapshot(mismatch);
    const auto saved_class=fixtures.vertex->vertex_class_inputs[0].input_class;
    fixtures.vertex->vertex_class_inputs[0].input_class=VertexFormatClass::uint_class;
    const auto refused=execute_metal_draw(&context,wrong,mismatch_memory);
    fixtures.vertex->vertex_class_inputs[0].input_class=saved_class;
    CHECK(!refused.ok()&&refused.status==CommandStatus::Unsupported&&mismatch.writes==0);
    // A cleared attribute register leaves the shader input unsupplied.
    Backing missing;
    auto missing_memory=missing.memory();
    auto absent=class_snapshot(missing);
    absent.registers[0x459]=0;
    const auto unbound=execute_metal_draw(&context,absent,missing_memory);
    CHECK(!unbound.ok()&&unbound.status==CommandStatus::Unsupported&&missing.writes==0);
    // A disabled stream behind a supplied attribute stops as well.
    Backing disabled;
    auto disabled_memory=disabled.memory();
    auto off=class_snapshot(disabled);
    off.registers[0x700]=8;
    CHECK(!execute_metal_draw(&context,off,disabled_memory).ok());
}

void verify_runtime_constant_buffers(Fixtures& fixtures) {
    Fixtures runtime=fixtures;
    runtime.fragment=std::make_shared<CompiledGraphicsStage>();
    auto& stage=*runtime.fragment;
    stage.stage=Stage::Fragment;
    stage.entry_point="armigo_fragment_main";
    stage.metallib=fixtures.vertex->metallib;
    stage.interface.color_output_mask=15;
    stage.color_outputs={{0,15,ColorOutputScalar::f32,4}};
    stage.constant_buffer_windows={{17,32,29},{2,4,18},{5,4,19}};
    stage.constant_buffer_support_index=30;
    Backing backing;
    auto stop=snapshot(backing);
    stop.constant_buffers[4]={};
    // Only seven bytes remain mapped. Copying window bytes or rounded-up
    // semantic bytes instead of the exact intersection would fault here.
    stop.constant_buffers[4][17]={0x10fff9,7,true};
    backing.word(0xfff9,std::bit_cast<std::uint32_t>(1.f));
    backing.bytes[0xfffd]=backing.bytes[0xfffe]=backing.bytes[0xffff]=std::byte{0xff};
    stop.constant_buffers[4][2]={UINT64_MAX,64,false};
    stop.constant_buffers[4][5]={UINT64_MAX,0,true};
    stop.constant_buffers[4][8]={UINT64_MAX,70000,true};
    auto memory=backing.memory();
    const auto before=backing.bytes;
    MetalDrawStatistics stats;
    MetalDrawContext context{Fixtures::lookup,&runtime,&stats};
    const auto result=execute_metal_draw(&context,stop,memory);
    if (!result.ok())std::fprintf(stderr,"Runtime CB backend: %.*s\n",int(result.detail.size()),result.detail.data());
    CHECK(result.ok() && stats.completed_draws==1);
    const auto p=image(8,8);
    const auto offset=std::get<std::size_t>(pixel_byte_offset(p,2,5));
    CHECK(backing.bytes[offset]==std::byte{255});
    CHECK(backing.bytes[offset+1]==std::byte{0});
    CHECK(backing.bytes[offset+2]==std::byte{0});
    CHECK(backing.bytes[offset+3]==std::byte{255});
    CHECK(std::equal(backing.bytes.begin()+512,backing.bytes.end(),before.begin()+512));

    // Reuse the same compiled stages across changes in bound size and enabled
    // state. Enlarging the binding beyond its window must not enlarge uploads.
    backing.bytes=before;
    stop.constant_buffers[4][17]={0x10ffe0,64,true};
    backing.word(0xffe0,std::bit_cast<std::uint32_t>(.25f));
    backing.word(0xffe4,std::bit_cast<std::uint32_t>(.5f));
    backing.word(0xfffc,0);
    stop.constant_buffers[4][2]={0x105300,4,true};
    backing.word(0x5300,std::bit_cast<std::uint32_t>(.75f));
    CHECK(execute_metal_draw(&context,stop,memory).ok());
    CHECK(std::abs(int(std::to_integer<unsigned>(backing.bytes[offset]))-int(srgb(.25f)))<=1);
    CHECK(std::abs(int(std::to_integer<unsigned>(backing.bytes[offset+1]))-int(srgb(.5f)))<=1);
    CHECK(std::abs(int(std::to_integer<unsigned>(backing.bytes[offset+2]))-int(srgb(.75f)))<=1);
    stop.constant_buffers[4][2].enabled=false;
    CHECK(execute_metal_draw(&context,stop,memory).ok());
    CHECK(backing.bytes[offset+2]==std::byte{0});

    // A failing enabled upload remains a guest-memory fault, not a zero CB.
    stop.constant_buffers[4][17]={0x200000,4,true};
    const auto completed=stats.completed_draws;
    const auto writes=backing.writes;
    CHECK(execute_metal_draw(&context,stop,memory).status==CommandStatus::MemoryFault);
    CHECK(stats.completed_draws==completed && backing.writes==writes);
    stop.constant_buffers[4][17]={0x10fff9,7,true};

    // Custom lookup providers receive the same policy/index checks as packages.
    const auto reject=[&] {
        CHECK(!execute_metal_draw(&context,stop,memory).ok());
        CHECK(stats.completed_draws==completed && backing.writes==writes);
    };
    stage.constant_buffer_support_index.reset(); reject();
    stage.constant_buffer_support_index=29; reject();
    stage.constant_buffer_support_index=31; reject();
    stage.constant_buffer_support_index=30;
    stage.constant_buffers={{0,4,0}}; reject(); stage.constant_buffers.clear();
    stage.constant_buffer_windows[0].window_bytes=3; reject();
    stage.constant_buffer_windows[0].window_bytes=32;
    stage.constant_buffer_windows.push_back({17,4,0}); reject();
    stage.constant_buffer_windows.pop_back();
    // A vertex support index may not alias its stream, while fragment windows
    // above already proved that another stage may use that same index range.
    runtime.vertex=std::make_shared<CompiledGraphicsStage>(*fixtures.vertex);
    runtime.vertex->constant_buffers.clear();
    runtime.vertex->constant_buffer_windows={{3,4,0}};
    runtime.vertex->constant_buffer_support_index=18; reject();
}
// Multi-target reflection loads, but the single-target renderer must stop the
// draw with a structured unsupported diagnostic and zero guest writes: for
// extra targets, typed integer exports, sparse masks, depth/sample-mask
// writes and mask/record disagreements alike.

namespace metal_backend = armigo::platform::metal;
RenderCallbacks engine_backend(std::shared_ptr<Fixtures> fixtures,
    std::shared_ptr<MetalEngineStatistics> statistics,
    std::shared_ptr<metal_backend::SubmissionTestControl> control = {},
    std::uint64_t timeout = 30'000'000'000ull, bool defer = false,
    std::uint32_t batch = 64, bool host_addressable = false) {
    MetalEngineOptions options;
    options.mode = MetalEngineMode::Persistent;
    options.lookup = Fixtures::lookup;
    options.lookup_context = fixtures.get();
    options.lookup_owner = fixtures;
    options.statistics = std::move(statistics);
    options.completion_timeout_ns = timeout;
    options.defer_gpu_writeback = defer;
    options.max_batch_draws = batch;
    options.guest_memory_host_addressable = host_addressable;
    options.cpu_write_tracking = host_addressable;
    auto created = metal_backend::create_engine_backend(std::move(options),
        metal_backend::create_pipeline_state_cache, std::move(control));
    if (const auto* error = std::get_if<RendererError>(&created)) {
        const auto result = error->command();
        std::fprintf(stderr, "Engine factory: %.*s\n", int(result.detail.size()), result.detail.data());
        CHECK(false);
        return {};
    }
    return std::get<RenderCallbacks>(std::move(created));
}
// The persistent binding path serves same-dimensions arrayness differences
// from a layer-zero view in the shader's target:
// a live 2D-array texture under a 2D fact samples layer 0 while layer 1
// holds a different uniform color, and a null 2D-array TIC types its dummy
// resource with the shader's 2D target and samples its zeros.
void verify_persistent_target_views(Fixtures& fixtures) {
    const DerivedTextureFact fact{4,32,true,false};
    const std::size_t stride=array_layer_stride();
    const auto array_texture=[&](Backing& b) {
        b.word(0x4010,1u|(5u<<23));
        b.word(0x4014,1u|(1u<<16)|(1u<<31));
        b.word(0x4000,8u|(2u<<7)|(2u<<10)|(2u<<13)|(2u<<16)|
            (2u<<19)|(3u<<22)|(4u<<25)|(5u<<28));
        for(unsigned y=0;y<2;++y)for(unsigned x=0;x<2;++x)
            b.word(0x3000+std::get<std::size_t>(pixel_byte_offset(image(2,2),x,y)),
                   0xff204081u);
        for(unsigned y=0;y<2;++y)for(unsigned x=0;x<2;++x)
            b.word(0x3000+stride+std::get<std::size_t>(pixel_byte_offset(image(2,2),x,y)),
                   0xff80c0e1u);
    };
    const auto read_pixel=[](const Backing& backing,unsigned x,unsigned y) {
        const auto offset=std::get<std::size_t>(pixel_byte_offset(image(8,8),x,y));
        std::array<unsigned,4> value{};
        for(unsigned lane=0;lane<4;++lane)
            value[lane]=std::to_integer<unsigned>(backing.bytes[offset+lane]);
        return value;
    };
    using Pixels=std::pair<std::array<unsigned,4>,std::array<unsigned,4>>;
    const auto run=[&](const auto& configure) -> Pixels {
        auto package=std::make_shared<Fixtures>(fixtures);
        package->fragment=std::make_shared<CompiledGraphicsStage>(*fixtures.fragment);
        package->fragment->texture_handle_cbuf=4;
        package->fragment->texture_facts={fact};
        auto stats=std::make_shared<MetalEngineStatistics>();
        auto backend=engine_backend(package,stats);
        if (!backend.draw) return {};
        auto backing=std::make_shared<Backing>();
        const auto stop=snapshot(*backing);
        configure(*backing);
        auto memory=backing->memory(backing);
        const auto result=backend.draw(backend.context,stop,memory);
        if(!result.ok())std::fprintf(stderr,"Persistent target view: %.*s\n",
            int(result.detail.size()),result.detail.data());
        CHECK(result.ok()&&stats->completed_draws==1&&backing->writes>=1);
        // (2,5) is inside the triangle; (6,1) is untouched fill. The target
        // write sRGB-encodes the sampled unorm value in this profile.
        const auto value=std::pair{read_pixel(*backing,2,5),read_pixel(*backing,6,1)};
        return value;
    };
    // The 2D fact samples the 2D-array texture's first layer only; layer 1's
    // uniform color would encode to (243,225,187) in every channel instead.
    {
        const auto [inside,outside]=run(array_texture);
        CHECK(std::abs(int(inside[0])-int(srgb(0x81/255.f)))<=2&&
              std::abs(int(inside[1])-int(srgb(0x40/255.f)))<=2&&
              std::abs(int(inside[2])-int(srgb(0x20/255.f)))<=2&&inside[3]==255);
        for(unsigned lane=0;lane<4;++lane)CHECK(outside[lane]==0x19);
    }
    // A null 2D-array TIC binds the shader-typed 2D null and samples zeros.
    {
        const auto [inside,outside]=run([](Backing& b) {
            b.word(0x4004,0);
            b.word(0x4010,1u|(5u<<23));
            b.word(0x4014,1u|(1u<<16)|(1u<<31));
        });
        for(unsigned lane=0;lane<4;++lane)CHECK(inside[lane]<=2);
        for(unsigned lane=0;lane<4;++lane)CHECK(outside[lane]==0x19);
    }
}
void verify_engine_parity(Fixtures& fixtures) {
    auto package = std::make_shared<Fixtures>(fixtures);
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = engine_backend(package, stats);
    if (!backend.draw) return;
    auto actual = std::make_shared<Backing>();
    auto stop = snapshot(*actual);
    auto memory = actual->memory(actual);
    const auto initial = actual->bytes;
    for (unsigned variant = 0; variant != 8; ++variant) {
        actual->bytes = initial;
        auto draw = stop;
        if (variant == 1) {
            draw.registers[0x381] = (6u << 16) | 2;
            draw.registers[0x382] = (6u << 16) | 2;
        } else if (variant == 2) {
            draw.registers[0x300] = draw.registers[0x301] = 4u << 16;
            draw.registers[0x280] = draw.registers[0x283] = std::bit_cast<std::uint32_t>(2.f);
            draw.registers[0x281] = std::bit_cast<std::uint32_t>(-2.f);
            draw.registers[0x284] = std::bit_cast<std::uint32_t>(2.f);
        } else if (variant == 3) {
            std::swap_ranges(actual->bytes.begin()+0x2006,actual->bytes.begin()+0x2008,actual->bytes.begin()+0x2008);
        } else if (variant == 4) {
            actual->word(0x5000, 0);
        } else if (variant >= 5) {
            alpha_blend_state(draw);
            if (variant == 6) draw.color_targets[0].format = 0xd1;
            if (variant == 7) draw.registers[0x646] = 0;
        }
        Backing expected = *actual;
        auto expected_memory = expected.memory();
        MetalDrawContext old{Fixtures::lookup, package.get(), nullptr};
        const auto legacy = execute_metal_draw(&old, draw, expected_memory);
        CHECK(legacy.ok());
        const auto result = backend.draw(backend.context, draw, memory);
        if (!result.ok()) std::fprintf(stderr, "Engine parity %u: %.*s\n", variant,
            int(result.detail.size()), result.detail.data());
        CHECK(result.ok());
        CHECK(actual->bytes == expected.bytes);
        if (!result.ok()) break;
    }
    CHECK(stats->completed_draws == 8);
    CHECK(stats->device_creations == 1);
    CHECK(stats->memory_domains == 1);
    CHECK(stats->command_buffers == 8);
    // A new allocation at the same addresses must not inherit the prior cache.
    auto replacement = std::make_shared<Backing>();
    auto replacement_stop = snapshot(*replacement);
    auto replacement_memory = replacement->memory(replacement);
    CHECK(backend.draw(backend.context, replacement_stop, replacement_memory).ok());
    CHECK(stats->device_creations == 1);
    CHECK(stats->memory_domains == 1);
    for (bool partial : {false, true}) {
        auto clear = replacement_stop;
        clear.kind = RenderKind::Clear;
        clear.draw.reset();
        clear.registers[0x206] = 1;
        clear.registers[0x674] = 0xfu << 2;
        clear.registers[0x360] = std::bit_cast<std::uint32_t>(0.25f);
        clear.registers[0x361] = std::bit_cast<std::uint32_t>(0.5f);
        clear.registers[0x363] = std::bit_cast<std::uint32_t>(1.f);
        if (partial) {
            clear.registers[0x43e] = 1u << 8;
            clear.registers[0x381] = (6u << 16) | 2;
            clear.registers[0x382] = (6u << 16) | 2;
        }
        Backing expected = *replacement;
        auto expected_memory = expected.memory();
        CHECK(execute_metal_color_clear(nullptr, clear, expected_memory).ok());
        const auto result = backend.clear(backend.context, clear, replacement_memory);
        if (!result.ok()) std::fprintf(stderr, "Engine clear: %.*s\n", int(result.detail.size()), result.detail.data());
        CHECK(result.ok());
        CHECK(replacement->bytes == expected.bytes);
    }
    CHECK(stats->completed_clears == 2);
    // Distinct GPU VAs of one retained allocation share a backing identity.
    replacement->bytes = initial;
    auto alias_stop = stop;
    alias_stop.registers[0x5f3] += 0x100000;
    alias_stop.registers[0x5f5] += 0x100000;
    alias_stop.registers[0x702] += 0x100000;
    alias_stop.registers[0x7c1] += 0x100000;
    const std::array mappings{
        Mapping{0x100000, 0x10000, 0, 0x800000, 0x10000, 0x1000, replacement},
        Mapping{0x200000, 0x10000, 0, 0x800000, 0x10000, 0x1000, replacement}};
    auto aliases = std::get<GpuAddressSpace>(make_address_space(mappings,
        {Backing::read, Backing::write, replacement.get()}));
    Backing expected = *replacement;
    auto expected_aliases = std::get<GpuAddressSpace>(make_address_space(mappings,
        {Backing::read, Backing::write, &expected}));
    MetalDrawContext old{Fixtures::lookup, package.get(), nullptr};
    CHECK(execute_metal_draw(&old, alias_stop, expected_aliases).ok());
    CHECK(backend.draw(backend.context, alias_stop, aliases).ok());
    CHECK(replacement->bytes == expected.bytes);
    CHECK(stats->memory_domains == 1 && stats->device_creations == 1);
}

void verify_engine_mapping_notifications(Fixtures& fixtures) {
    auto package = std::make_shared<Fixtures>(fixtures);
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = engine_backend(package, stats);
    CHECK(backend.draw && backend.memory_changed);
    if (!backend.draw || !backend.memory_changed) return;
    auto original = std::make_shared<Backing>();
    std::weak_ptr<Backing> lifetime = original;
    auto stop = snapshot(*original);
    auto memory = original->memory(original);
    CHECK(backend.draw(backend.context, stop, memory).ok());
    const auto first_pixels = original->bytes;
    const auto writes = original->writes;
    // Notify while the old allocation is accessible. A clean resident resource
    // is not a reason to retain its removed mapping or issue another writeback.
    CHECK(backend.memory_changed(backend.context, memory.mappings(), {},
        memory.backing_callbacks()).ok());
    CHECK(original->writes == writes);
    stop = snapshot(*original);
    original->word(0x5000, std::bit_cast<std::uint32_t>(0.5f));
    Backing expected = *original;
    MetalDrawContext legacy{Fixtures::lookup, package.get(), nullptr};
    CHECK(execute_metal_draw(&legacy, stop, expected.memory()).ok());
    CHECK(backend.draw(backend.context, stop, memory).ok());
    CHECK(original->bytes == expected.bytes);
    CHECK(!std::equal(first_pixels.begin(), first_pixels.begin() + 512, original->bytes.begin()));
    {
        const std::array<std::shared_ptr<void>, 1> released{original};
        CHECK(backend.memory_changed(backend.context, memory.mappings(), released,
            memory.backing_callbacks()).ok());
    }
    memory = std::get<GpuAddressSpace>(make_address_space({}, {}));
    original.reset();
    CHECK(lifetime.expired());

    // Reuse every resource VA for a different allocation. The next real draw
    // must rebuild its attachment, texture, index, vertex and constant bindings.
    auto replacement = std::make_shared<Backing>();
    stop = snapshot(*replacement);
    replacement->word(0x5000, std::bit_cast<std::uint32_t>(0.75f));
    const auto layout = image(2, 2);
    for (unsigned y = 0; y != 2; ++y)
        for (unsigned x = 0; x != 2; ++x)
            replacement->word(0x3000 + std::get<std::size_t>(pixel_byte_offset(layout, x, y)), 0xc00ffc00);
    expected = *replacement;
    CHECK(execute_metal_draw(&legacy, stop, expected.memory()).ok());
    memory = replacement->memory(replacement);
    const auto result = backend.draw(backend.context, stop, memory);
    if (!result.ok()) std::fprintf(stderr, "Remapped draw: %.*s\n", int(result.detail.size()), result.detail.data());
    CHECK(result.ok());
    CHECK(replacement->bytes == expected.bytes);
    CHECK(stats->completed_draws == 3);
    CHECK(stats->device_creations == 1 && stats->memory_domains == 1);
}

void verify_null_texture_remap_sequence(Fixtures& fixtures) {
    enum class NullMetadata { Color, Depth, Integer };
    for (const auto mode : {MetalEngineMode::Legacy, MetalEngineMode::Persistent})
    for (const auto null_metadata : {NullMetadata::Color, NullMetadata::Depth, NullMetadata::Integer}) {
        auto package = std::make_shared<Fixtures>(fixtures);
        if (null_metadata != NullMetadata::Color) {
            package->fragment = std::make_shared<CompiledGraphicsStage>(*fixtures.fragment);
            package->fragment->texture_handle_cbuf = 4;
            package->fragment->texture_facts = {{4, 32, true, false}};
            auto& fact = package->fragment->texture_facts.front();
            fact.component_format = TextureComponentFormat::floating;
            fact.scalar = TextureSampledScalar::f32;
        }
        auto stats = std::make_shared<MetalEngineStatistics>();
        MetalEngineOptions options;
        options.mode = mode;
        options.lookup = Fixtures::lookup;
        options.lookup_context = package.get();
        options.lookup_owner = package;
        options.statistics = stats;
        auto created = create_metal_engine_backend(options);
        CHECK(std::holds_alternative<RenderCallbacks>(created));
        if (!std::holds_alternative<RenderCallbacks>(created)) continue;
        auto backend = std::get<RenderCallbacks>(std::move(created));
        CHECK(backend.draw);
        CHECK(bool(backend.memory_changed) == (mode == MetalEngineMode::Persistent));
        if (!backend.draw) continue;

        auto backing = std::make_shared<Backing>();
        const auto stop = rgba8_snapshot(*backing, false);
        std::uint32_t color_format_word = 0;
        std::memcpy(&color_format_word, backing->bytes.data() + 0x4000, sizeof(color_format_word));
        const auto fill_texture = [&](std::size_t offset, std::uint32_t color) {
            for (unsigned y = 0; y < 2; ++y) for (unsigned x = 0; x < 2; ++x)
                backing->word(offset + std::get<std::size_t>(pixel_byte_offset(image(2, 2), x, y)), color);
        };
        fill_texture(0x3000, 0xff0000ff);
        fill_texture(0x6000, 0x8000ff00);
        const Mapping lower{0x100000, 0x3000, 0, 0x800000, 0x10000, 0x1000, backing};
        const Mapping pools{0x104000, 0xc000, 0x4000, 0x800000, 0x10000, 0x1000, backing};
        Mapping texture{0x103000, 0x1000, 0x3000, 0x800000, 0x10000, 0x1000, backing};
        const auto memory_with = [&](bool mapped) {
            std::vector<Mapping> mappings{lower, pools};
            if (mapped) mappings.push_back(texture);
            return std::get<GpuAddressSpace>(make_address_space(mappings,
                {Backing::read, Backing::write, backing.get()}));
        };
        auto memory = memory_with(true);
        const std::array<std::array<unsigned, 4>, 3> expected{{
            {255, 0, 0, 255}, {0, 0, 0, 0}, {0, 255, 0, 128}}};
        for (unsigned phase = 0; phase < expected.size(); ++phase) {
            if (phase == 1) {
                // Only sampled storage disappears; pools, handles and targets remain live.
                if (backend.memory_changed) {
                    const auto notified = backend.memory_changed(backend.context,
                        std::span{&texture, 1}, {}, memory.backing_callbacks());
                    CHECK(notified.ok());
                    if (!notified.ok()) break;
                }
                memory = memory_with(false);
                backing->word(0x4004, 0);
                if (null_metadata == NullMetadata::Depth) {
                    // An absent D32F backing still binds a zero color resource for
                    // the reflected ordinary float sampler, without comparison.
                    backing->word(0x4000, 0x2fu | (7u << 7) | (4u << 10) |
                        (4u << 13) | (4u << 16) | (2u << 19) | (2u << 22) |
                        (2u << 25) | (7u << 28));
                } else if (null_metadata == NullMetadata::Integer) {
                    // An absent unsigned-integer backing does not change the
                    // reflected float resource class or its all-zero result.
                    std::uint32_t integer_format_word = color_format_word;
                    for (unsigned component = 0; component < 4; ++component) {
                        const unsigned shift = 7 + component * 3;
                        integer_format_word = (integer_format_word & ~(7u << shift)) | (4u << shift);
                    }
                    backing->word(0x4000, integer_format_word);
                }
            } else if (phase == 2) {
                // Reuse the GPU VA with a different source offset in the same owner.
                texture.buffer_offset = 0x6000;
                memory = memory_with(true);
                backing->word(0x4000, color_format_word);
                backing->word(0x4004, 0x103000);
            }
            const auto result = backend.draw(backend.context, stop, memory);
            if (!result.ok()) std::fprintf(stderr, "Null texture transition mode %u metadata %u phase %u: %.*s\n",
                unsigned(mode), unsigned(null_metadata), phase,
                int(result.detail.size()), result.detail.data());
            CHECK(result.ok());
            if (!result.ok()) break;
            CHECK(stats->completed_draws == phase + 1);
            const auto target = image(8, 8);
            for (unsigned y = 0; y < 8; ++y) for (unsigned x = 0; x < 8; ++x) {
                const auto offset = std::get<std::size_t>(pixel_byte_offset(target, x, y));
                if (y > x + 1) {
                    for (unsigned lane = 0; lane < 4; ++lane)
                        CHECK(std::to_integer<unsigned>(backing->bytes[offset + lane]) == expected[phase][lane]);
                } else if (x > y + 1) {
                    for (unsigned lane = 0; lane < 4; ++lane)
                        CHECK(backing->bytes[offset + lane] == std::byte{0x19});
                }
            }
        }
        CHECK(stats->completed_draws == 3);
        if (mode == MetalEngineMode::Persistent) {
            CHECK(stats->device_creations == 1 && stats->memory_domains == 1);
            CHECK(stats->command_buffers == 3);
        }
    }
}

RenderStop pitch_clear_snapshot(std::uint32_t width) {
    RenderStop clear;
    clear.engine = engine_class::threed;
    clear.kind = RenderKind::Clear;
    clear.registers[0xdf] = 1;
    clear.registers[0x206] = 1;
    clear.registers[0x674] = 0xfu << 2;
    clear.registers[0x3fd] = width << 16;
    clear.registers[0x3fe] = 139u << 16;
    clear.registers[0x360] = std::bit_cast<std::uint32_t>(1.f);
    clear.registers[0x362] = std::bit_cast<std::uint32_t>(1.f);
    clear.registers[0x363] = std::bit_cast<std::uint32_t>(1.f);
    clear.color_targets[0] = {0x100000, 192, 139, 0xd5, 0x1000, 1, 0, 0};
    return clear;
}

void verify_engine_pitch_clear_padding(Fixtures& fixtures) {
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = engine_backend(std::make_shared<Fixtures>(fixtures), stats);
    if (!backend.clear) return;
    auto actual = std::make_shared<Backing>();
    // Distinct row padding catches both overwrites and incorrect row strides.
    for (std::size_t offset = 0; offset < actual->bytes.size(); ++offset)
        actual->bytes[offset] = std::byte((offset * 29 + offset / 192) & 0xff);
    const auto initial = actual->bytes;
    auto memory = actual->memory(actual);
    const auto clear = pitch_clear_snapshot(37);
    Backing expected = *actual;
    auto expected_memory = expected.memory();
    CHECK(execute_metal_color_clear(nullptr, clear, expected_memory).ok());
    const auto result = backend.clear(backend.context, clear, memory);
    if (!result.ok()) std::fprintf(stderr, "Pitch clear: %.*s\n",
        int(result.detail.size()), result.detail.data());
    CHECK(result.ok());
    CHECK(actual->bytes == expected.bytes);
    for (std::size_t row = 0; row < 139; ++row) {
        const auto padding = row * 192 + 37 * 4;
        CHECK(std::equal(actual->bytes.begin() + padding,
                         actual->bytes.begin() + (row + 1) * 192,
                         initial.begin() + padding));
    }
    CHECK(stats->completed_clears == 1);
}

void verify_engine_clear_scissor_parity(Fixtures& fixtures) {
    auto package = std::make_shared<Fixtures>(fixtures);
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = engine_backend(package, stats);
    if (!backend.clear) return;
    auto actual = std::make_shared<Backing>();
    auto memory = actual->memory(actual);
    for (const auto control : {0x100u, 0x101u, 0u, 1u}) {
        for (const auto enable : {0u, 1u}) {
            std::fill(actual->bytes.begin(), actual->bytes.end(), std::byte{0x19});
            auto clear = pitch_clear_snapshot(48);
            clear.registers[0x43e] = control;
            clear.registers[0x380] = enable;
            // Disabled scissoring must ignore even an inverted rectangle.
            clear.registers[0x381] = enable ? (20u << 16) | 3 : (2u << 16) | 9;
            clear.registers[0x382] = enable ? (30u << 16) | 4 : 0;
            Backing expected = *actual;
            auto expected_memory = expected.memory();
            CHECK(execute_metal_color_clear(nullptr, clear, expected_memory).ok());
            const auto result = backend.clear(backend.context, clear, memory);
            if (!result.ok()) std::fprintf(stderr, "Clear control %x enable %u: %.*s\n",
                control, enable, int(result.detail.size()), result.detail.data());
            CHECK(result.ok());
            CHECK(actual->bytes == expected.bytes);
        }
    }
    CHECK(stats->completed_clears == 8);
    // A subsequent draw on the same backend proves no sticky failure latched.
    const auto draw = snapshot(*actual);
    Backing expected = *actual;
    auto expected_memory = expected.memory();
    MetalDrawContext legacy{Fixtures::lookup, package.get(), nullptr};
    CHECK(execute_metal_draw(&legacy, draw, expected_memory).ok());
    CHECK(backend.draw(backend.context, draw, memory).ok());
    CHECK(actual->bytes == expected.bytes);
    CHECK(stats->completed_draws == 1);
}

void verify_engine_selection(Fixtures& fixtures) {
    const auto* prior_value = std::getenv("ARMIGO_METAL_RENDERER");
    const std::optional<std::string> prior = prior_value ? std::optional<std::string>(prior_value) : std::nullopt;
    auto package = std::make_shared<Fixtures>(fixtures);
    MetalEngineOptions options;
    options.lookup = Fixtures::lookup;
    options.lookup_context = package.get();
    options.lookup_owner = package;
    auto statistics = std::make_shared<MetalEngineStatistics>();
    options.statistics = statistics;
    ::setenv("ARMIGO_METAL_RENDERER", "invalid", 1);
    CHECK(std::holds_alternative<RendererError>(create_metal_engine_backend(options)));
    options.mode = MetalEngineMode::Legacy;
    auto explicit_legacy = create_metal_engine_backend(options);
    CHECK(std::holds_alternative<RenderCallbacks>(explicit_legacy));
    options.mode = MetalEngineMode::Environment;
    ::unsetenv("ARMIGO_METAL_RENDERER");
    CHECK(std::holds_alternative<RenderCallbacks>(create_metal_engine_backend(options)));
    CHECK(statistics->device_creations == 0);
    ::setenv("ARMIGO_METAL_RENDERER", "persistent", 1);
    auto selected = create_metal_engine_backend(options);
    CHECK(std::holds_alternative<RenderCallbacks>(selected));
    CHECK(statistics->device_creations == 1);
    if (const auto* callback = std::get_if<RenderCallbacks>(&selected)) {
        auto backing = std::make_shared<Backing>();
        const auto stop = snapshot(*backing);
        const auto borrowed = backing->memory();
        const auto result = callback->draw(callback->context, stop, borrowed);
        CHECK(result.status == CommandStatus::Unsupported);
        CHECK(result.detail == "Persistent engine renderer requires retained GPU mapping owners");
        CHECK(backing->writes == 0);
        const auto pinned = backing->memory(backing);
        CHECK(callback->draw(callback->context, stop, pinned).detail == result.detail);
    }
    options.lookup_owner.reset();
    CHECK(std::holds_alternative<RendererError>(create_metal_engine_backend(options)));
    if (prior) ::setenv("ARMIGO_METAL_RENDERER", prior->c_str(), 1);
    else ::unsetenv("ARMIGO_METAL_RENDERER");
}

struct EngineFeedback final : metal_backend::SubmissionTestControl {
    std::mutex mutex;
    std::condition_variable ready;
    bool entered = false, released = false, fail = false;
    CommandResult after_gpu_quiescence(CommandResult actual) override {
        std::unique_lock lock(mutex);
        entered = true;
        ready.notify_all();
        ready.wait(lock, [&] { return released; });
        return fail ? CommandResult{CommandStatus::MemoryFault, "injected engine GPU failure"} : actual;
    }
    void await_entry() {
        std::unique_lock lock(mutex);
        CHECK(ready.wait_for(lock, std::chrono::seconds(10), [&] { return entered; }));
    }
    void release() {
        std::lock_guard lock(mutex);
        released = true;
        ready.notify_all();
    }
};
void verify_engine_completion(Fixtures& fixtures) {
    auto package = std::make_shared<Fixtures>(fixtures);
    for (unsigned mode = 0; mode != 4; ++mode) {
        auto stats = std::make_shared<MetalEngineStatistics>();
        auto control = std::make_shared<EngineFeedback>();
        control->fail = mode == 1;
        auto backend = engine_backend(package, stats, control, mode == 3 ? 1'000'000 : 30'000'000'000ull);
        if (!backend.draw) continue;
        auto backing = std::make_shared<Backing>();
        auto stop = snapshot(*backing);
        auto memory = std::make_optional(backing->memory(backing));
        const auto before = backing->bytes;
        if (mode == 2) backing->fail_write = true;
        std::atomic<bool> returned = false;
        CommandResult result;
        std::thread worker([&] {
            result = backend.draw(backend.context, stop, *memory);
            returned.store(true, std::memory_order_release);
        });
        control->await_entry();
        if (mode == 3) {
            worker.join();
            CHECK(returned.load(std::memory_order_acquire));
            CHECK(result.status == CommandStatus::Stopped);
            // Late feedback cannot dereference this destroyed borrowed view.
            memory.reset();
        } else {
            CHECK(!returned.load(std::memory_order_acquire));
        }
        CHECK(backing->bytes == before);
        CHECK(backing->writes == 0);
        CHECK(stats->completed_draws == 0);
        control->release();
        if (mode != 3) worker.join();
        if (mode == 0) {
            CHECK(result.ok());
            CHECK(stats->completed_draws == 1);
            CHECK(backing->bytes != before);
        } else {
            CHECK(!result.ok());
            CHECK(stats->completed_draws == 0);
            const auto failed_writes = backing->writes;
            auto retry_memory = backing->memory(backing);
            const auto retry = backend.draw(backend.context, stop, retry_memory);
            CHECK(retry.status == result.status && retry.detail == result.detail);
            CHECK(backing->writes == failed_writes);
            backend = {};
            CHECK(backing->bytes == before);
            CHECK(backing->writes == failed_writes);
        }
    }
}

// Deferred write-back encodes draws into one batch and commits it where the
// guest can observe results: draws return
// without a commit or wait, coherence (full or ranged) commits and waits,
// budget pressure splits batches, and the batch commits at destruction. The
// byte oracle is the legacy synchronous draw result.
void verify_engine_deferred_batches(Fixtures& fixtures) {
    auto package = std::make_shared<Fixtures>(fixtures);
    const auto target_bytes = 8u * 8u * 4u;
    // N draws make one command buffer, committed only by a coherence call.
    {
        auto stats = std::make_shared<MetalEngineStatistics>();
        auto backend = engine_backend(package, stats, {}, 30'000'000'000ull, true, 8);
        if (!backend.draw || !backend.coherence) {CHECK(!"deferred backend missing callbacks");return;}
        CHECK(!backend.ordered_effect);
        auto backing = std::make_shared<Backing>();
        auto stop = snapshot(*backing);
        Backing initial = *backing;
        Backing expected = *backing;
        auto memory = backing->memory(backing);
        auto expected_memory = expected.memory();
        MetalDrawContext old{Fixtures::lookup, package.get(), nullptr};
        for (unsigned index = 0; index != 4; ++index) {
            CHECK(execute_metal_draw(&old, stop, expected_memory).ok());
            const auto result = backend.draw(backend.context, stop, memory);
            if (!result.ok()) std::fprintf(stderr, "Deferred batch draw: %.*s\n",
                int(result.detail.size()), result.detail.data());
            CHECK(result.ok());
        }
        // Encoded work stays uncommitted: no command buffer, no completion,
        // no guest writeback, and the bytes still match the untouched oracle.
        CHECK(stats->command_buffers == 0);
        CHECK(stats->completed_draws == 0);
        CHECK(backing->writes == 0);
        CHECK(std::equal(backing->bytes.begin(), backing->bytes.begin() + target_bytes,
            initial.bytes.begin()));
        const auto result = backend.coherence(backend.context, memory, 0, 0);
        if (!result.ok()) std::fprintf(stderr, "Deferred full coherence: %.*s\n",
            int(result.detail.size()), result.detail.data());
        CHECK(result.ok());
        CHECK(stats->command_buffers == 1);
        CHECK(stats->render_passes == 1);
        CHECK(stats->completed_draws == 4);
        CHECK(stats->draws_per_command_buffer == 4.0);
        CHECK(stats->cpu_wait_ns > 0);
        CHECK(backing->writes > 0);
        CHECK(std::equal(backing->bytes.begin(), backing->bytes.end(), expected.bytes.begin()));
    }
    // A ranged coherence publishes exactly the bytes the front end is about
    // to write, so an upload can observe prior GPU work mid-batch.
    {
        auto stats = std::make_shared<MetalEngineStatistics>();
        auto backend = engine_backend(package, stats, {}, 30'000'000'000ull, true, 8);
        if (!backend.draw || !backend.coherence) {CHECK(!"deferred backend missing callbacks");return;}
        auto backing = std::make_shared<Backing>();
        auto stop = snapshot(*backing);
        Backing expected = *backing;
        auto memory = backing->memory(backing);
        auto expected_memory = expected.memory();
        MetalDrawContext old{Fixtures::lookup, package.get(), nullptr};
        for (unsigned index = 0; index != 2; ++index) {
            CHECK(execute_metal_draw(&old, stop, expected_memory).ok());
            CHECK(backend.draw(backend.context, stop, memory).ok());
        }
        // Bytes no pending GPU work writes (a constant-buffer update between
        // draws) leave the batch open: no commit, wait or writeback.
        const auto untouched = backend.coherence(backend.context, memory, 0x10f000, 4);
        CHECK(untouched.ok());
        CHECK(stats->command_buffers == 0);
        CHECK(stats->completed_draws == 0);
        CHECK(backing->writes == 0);
        const auto result = backend.coherence(backend.context, memory, 0x100000, target_bytes);
        if (!result.ok()) std::fprintf(stderr, "Deferred ranged coherence: %.*s\n",
            int(result.detail.size()), result.detail.data());
        CHECK(result.ok());
        CHECK(stats->command_buffers == 1);
        CHECK(stats->completed_draws == 2);
        CHECK(backing->writes > 0);
        CHECK(std::equal(backing->bytes.begin(), backing->bytes.begin() + target_bytes,
            expected.bytes.begin()));
    }
    // Budget pressure splits the batch at the configured size without waiting
    // on the GPU between the splits.
    {
        auto stats = std::make_shared<MetalEngineStatistics>();
        auto backend = engine_backend(package, stats, {}, 30'000'000'000ull, true, 2);
        if (!backend.draw || !backend.coherence) {CHECK(!"deferred backend missing callbacks");return;}
        auto backing = std::make_shared<Backing>();
        auto stop = snapshot(*backing);
        Backing expected = *backing;
        auto memory = backing->memory(backing);
        auto expected_memory = expected.memory();
        MetalDrawContext old{Fixtures::lookup, package.get(), nullptr};
        for (unsigned index = 0; index != 4; ++index) {
            CHECK(execute_metal_draw(&old, stop, expected_memory).ok());
            const auto result = backend.draw(backend.context, stop, memory);
            if (!result.ok()) std::fprintf(stderr, "Deferred split draw: %.*s\n",
                int(result.detail.size()), result.detail.data());
            CHECK(result.ok());
        }
        CHECK(stats->command_buffers == 2);
        CHECK(stats->completed_draws == 0 || stats->completed_draws == 2 ||
              stats->completed_draws == 4);
        const auto result = backend.coherence(backend.context, memory, 0, 0);
        if (!result.ok()) std::fprintf(stderr, "Deferred split coherence: %.*s\n",
            int(result.detail.size()), result.detail.data());
        CHECK(result.ok());
        CHECK(stats->command_buffers == 2);
        CHECK(stats->render_passes == 2);
        CHECK(stats->completed_draws == 4);
        CHECK(stats->draws_per_command_buffer == 2.0);
        CHECK(backing->writes > 0);
        CHECK(std::equal(backing->bytes.begin(), backing->bytes.end(), expected.bytes.begin()));
    }
    // A GPU failure injected into a committed batch surfaces at the next
    // draw with its original diagnostic and stays sticky; failed work never
    // writes guest memory.
    {
        auto stats = std::make_shared<MetalEngineStatistics>();
        auto control = std::make_shared<EngineFeedback>();
        control->fail = true;
        auto backend = engine_backend(package, stats, control, 30'000'000'000ull, true, 2);
        if (!backend.draw) {CHECK(!"deferred backend missing draw");return;}
        auto backing = std::make_shared<Backing>();
        auto stop = snapshot(*backing);
        auto memory = backing->memory(backing);
        CHECK(backend.draw(backend.context, stop, memory).ok());
        const auto committed = backend.draw(backend.context, stop, memory);
        if (!committed.ok()) std::fprintf(stderr, "Deferred failure batch: %.*s\n",
            int(committed.detail.size()), committed.detail.data());
        CHECK(committed.ok());
        // The budget flush committed the two draws; their feedback blocks in
        // the test control until the injection is released.
        control->await_entry();
        control->release();
        CommandResult surfaced, sticky;
        try {
            auto fail_memory = backing->memory(backing);
            surfaced = backend.draw(backend.context, stop, fail_memory);
        } catch (...) {CHECK(!"deferred failure surface threw");return;}
        CHECK(surfaced.status == CommandStatus::MemoryFault);
        CHECK(surfaced.detail == "injected engine GPU failure");
        auto sticky_memory = backing->memory(backing);
        sticky = backend.draw(backend.context, stop, sticky_memory);
        CHECK(sticky.status == surfaced.status && sticky.detail == surfaced.detail);
        CHECK(stats->completed_draws == 0);
        CHECK(backing->writes == 0);
        backend = {};
    }
    // Destruction commits and drains the still-open batch.
    {
        auto stats = std::make_shared<MetalEngineStatistics>();
        auto backend = engine_backend(package, stats, {}, 30'000'000'000ull, true, 8);
        if (!backend.draw) {CHECK(!"deferred backend missing draw");return;}
        auto backing = std::make_shared<Backing>();
        auto stop = snapshot(*backing);
        Backing initial = *backing;
        Backing expected = *backing;
        auto memory = backing->memory(backing);
        auto expected_memory = expected.memory();
        MetalDrawContext old{Fixtures::lookup, package.get(), nullptr};
        CHECK(execute_metal_draw(&old, stop, expected_memory).ok());
        CHECK(backend.draw(backend.context, stop, memory).ok());
        CHECK(backing->writes == 0);
        // Destruction commits and waits for the open batch, but after the
        // callbacks return no CPU capability remains: the results stay
        // GPU-authoritative and only the completion statistics convert.
        backend = {};
        CHECK(stats->command_buffers == 1);
        CHECK(stats->completed_draws == 1);
        CHECK(backing->writes == 0);
        CHECK(std::equal(backing->bytes.begin(), backing->bytes.end(), initial.bytes.begin()));
    }
    // Without deferred write-back every draw still commits and completes on
    // its own, with its guest writeback visible immediately.
    {
        auto stats = std::make_shared<MetalEngineStatistics>();
        auto backend = engine_backend(package, stats);
        if (!backend.draw) {CHECK(!"deferred backend missing draw");return;}
        auto backing = std::make_shared<Backing>();
        auto stop = snapshot(*backing);
        auto memory = backing->memory(backing);
        for (unsigned index = 0; index != 2; ++index) {
            const auto before = backing->writes;
            const auto result = backend.draw(backend.context, stop, memory);
            if (!result.ok()) std::fprintf(stderr, "Synchronous draw: %.*s\n",
                int(result.detail.size()), result.detail.data());
            CHECK(result.ok());
            CHECK(stats->completed_draws == index + 1);
            CHECK(stats->command_buffers == index + 1);
            CHECK(backing->writes > before);
        }
        CHECK(stats->draws_per_command_buffer == 1.0);
    }
}

// Two draws in one open deferred batch must each observe guest CPU bytes as of
// their own callback entry: the second draw revalidates CPU-backed uploads
// against bytes the guest changed between the entries and encodes replacement
// uploads, while the already enqueued first draw keeps the immutable copies it
// captured. The byte oracle is the legacy synchronous draw result.
void verify_engine_batch_freshness(Fixtures& fixtures) {
    auto package = std::make_shared<Fixtures>(fixtures);
    auto stats = std::make_shared<MetalEngineStatistics>();
    // The cap stays above both draws, so no budget commit splits the batch.
    auto backend = engine_backend(package, stats, {}, 30'000'000'000ull, true, 8);
    if (!backend.draw || !backend.coherence) {CHECK(!"deferred backend missing callbacks");return;}
    auto backing = std::make_shared<Backing>();
    auto stop = snapshot(*backing);
    Backing expected = *backing;
    auto memory = backing->memory(backing);
    auto expected_memory = expected.memory();
    MetalDrawContext old{Fixtures::lookup, package.get(), nullptr};
    // Draw one encodes against the original bytes and stays uncommitted.
    CHECK(execute_metal_draw(&old, stop, expected_memory).ok());
    CHECK(backend.draw(backend.context, stop, memory).ok());
    CHECK(stats->command_buffers == 0);
    CHECK(stats->completed_draws == 0);
    // The guest CPU rewrites vertex, constant and texture bytes while the
    // batch is still open, identically for the oracle and the batch.
    const auto layout = image(2, 2);
    const float moved_u = 0.75f;
    for (auto* target : {backing.get(), &expected}) {
        target->word(0x5000, std::bit_cast<std::uint32_t>(0.5f));
        std::memcpy(target->bytes.data() + 0x1000 + 9 * sizeof(float), &moved_u,
                    sizeof(moved_u));
        for (unsigned y = 0; y < 2; ++y)for (unsigned x = 0; x < 2; ++x)
            target->word(0x3000 + std::get<std::size_t>(pixel_byte_offset(layout, x, y)),
                         0xff204081u);
    }
    // Draw two must encode against the new bytes; the batch is still open.
    CHECK(execute_metal_draw(&old, stop, expected_memory).ok());
    const auto encoded = backend.draw(backend.context, stop, memory);
    if (!encoded.ok()) std::fprintf(stderr, "Batch freshness draw: %.*s\n",
        int(encoded.detail.size()), encoded.detail.data());
    CHECK(encoded.ok());
    CHECK(stats->command_buffers == 0);
    // Coherence commits the open batch; the readback must show draw one's
    // original inputs where only it drew, and draw two's updated inputs.
    const auto coherent = backend.coherence(backend.context, memory, 0, 0);
    if (!coherent.ok()) std::fprintf(stderr, "Batch freshness coherence: %.*s\n",
        int(coherent.detail.size()), coherent.detail.data());
    CHECK(coherent.ok());
    CHECK(stats->command_buffers == 1);
    CHECK(stats->completed_draws == 2);
    CHECK(stats->draws_per_command_buffer == 2.0);
    CHECK(backing->writes > 0);
    CHECK(std::equal(backing->bytes.begin(), backing->bytes.end(), expected.bytes.begin()));
}

// Completion feedback staged per outstanding submission: each entry blocks
// until the test releases its index, so the first batch's completion can be
// held back past the second batch's failure. Release waits for the recorded
// feedback, so later engine calls observe the settled completion states.
struct StagedFeedback final : metal_backend::SubmissionTestControl {
    std::mutex mutex;
    std::condition_variable ready;
    unsigned entered = 0, returned = 0, encoded = 0, committed = 0;
    RendererStatus before_encode(SubmissionId) override {
        std::lock_guard lock(mutex);
        ++encoded;
        ready.notify_all();
        return renderer_success();
    }
    RendererStatus before_commit(SubmissionId) override {
        std::lock_guard lock(mutex);
        ++committed;
        ready.notify_all();
        return renderer_success();
    }
    bool await_encoded(unsigned count) {
        std::unique_lock lock(mutex);
        return ready.wait_for(lock, std::chrono::seconds(10), [&] { return encoded >= count; });
    }
    bool released[3] {};
    CommandResult results[3];
    CommandResult after_gpu_quiescence(CommandResult actual) override {
        std::unique_lock lock(mutex);
        const unsigned index = ++entered;
        CHECK(index <= 2);
        if (index > 2) return actual;
        ready.notify_all();
        ready.wait(lock, [&] { return released[index]; });
        const CommandResult result = results[index].ok() ? actual : results[index];
        ++returned;
        ready.notify_all();
        return result;
    }
    void await(unsigned count) {
        std::unique_lock lock(mutex);
        CHECK(ready.wait_for(lock, std::chrono::seconds(10),
                             [&] { return entered >= count; }));
    }
    void release(unsigned index, CommandResult result) {
        std::unique_lock lock(mutex);
        released[index] = true;
        results[index] = std::move(result);
        ready.notify_all();
        CHECK(ready.wait_for(lock, std::chrono::seconds(10),
                             [&] { return returned >= index; }));
    }
};

bool await_atomic(std::atomic<unsigned>& value, unsigned expected) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(value.load(std::memory_order_acquire)<expected &&
          std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool reached=value.load(std::memory_order_acquire)>=expected;
    CHECK(reached);return reached;
}

struct ThreadLookup {
    std::shared_ptr<Fixtures> fixtures;
    std::mutex mutex;
    std::condition_variable ready;
    std::thread::id thread;
    unsigned calls = 0;
    bool block = false, entered = false, released = false;
    std::uint64_t observed_address = 0;
    bool observations_valid = true;
    unsigned blocked_reads = 0;
    std::vector<std::array<std::byte, 4>> observations;
    std::vector<std::array<std::uint32_t, 2>> register_observations;
    static CompiledGraphicsLookupResult lookup(void* raw, const RenderStop& stop,
                                               const GpuAddressSpace& memory) {
        auto& probe = *static_cast<ThreadLookup*>(raw);
        {
            std::unique_lock lock(probe.mutex);
            probe.thread = std::this_thread::get_id();
            ++probe.calls;
            probe.register_observations.push_back({stop.registers[0xe00], stop.registers[0xe01]});
            probe.entered = true;
            probe.ready.notify_all();
            if(probe.block) probe.ready.wait(lock,[&] { return probe.released; });
            if (probe.observed_address) {
                std::array<std::byte, 4> bytes{};
                if (!memory.read(probe.observed_address, bytes).ok()) {
                    ++probe.blocked_reads;
                    probe.ready.notify_all();
                    return CommandResult{CommandStatus::Stopped, "Synthetic shader input read is pending"};
                }
                probe.observations.push_back(bytes);
            }
        }
        return CompiledGraphicsPair{probe.fixtures->vertex, probe.fixtures->fragment};
    }
    void await_entry() {
        std::unique_lock lock(mutex);
        CHECK(ready.wait_for(lock,std::chrono::seconds(10),[&] { return entered; }));
    }
    void release() {
        std::lock_guard lock(mutex);
        released=true;
        ready.notify_all();
    }
};

struct RetryPublicationGate {
    std::mutex mutex;
    std::condition_variable ready;
    unsigned attempts = 0;
    unsigned publications = 0;
    bool permitted = false;
    std::optional<CommandResult> publish() {
        std::lock_guard lock(mutex);
        ++attempts;
        ready.notify_all();
        if (!permitted) return std::nullopt;
        ++publications;
        return CommandResult{};
    }
    void await_attempts(unsigned expected) {
        std::unique_lock lock(mutex);
        CHECK(ready.wait_for(lock, std::chrono::seconds(10),
                             [&] { return attempts >= expected; }));
    }
    void release() {
        std::lock_guard lock(mutex);
        permitted = true;
        ready.notify_all();
    }
};

struct AsyncCommand {
    std::mutex mutex;
    std::condition_variable ready;
    bool started = false;
    bool done = false;
    CommandResult result;
    void begin() {
        std::lock_guard lock(mutex);
        started = true;
        ready.notify_all();
    }
    void complete(CommandResult value) {
        std::lock_guard lock(mutex);
        result = value;
        done = true;
        ready.notify_all();
    }
    void await_started() {
        std::unique_lock lock(mutex);
        CHECK(ready.wait_for(lock, std::chrono::seconds(10), [&] { return started; }));
    }
    bool await_done() {
        std::unique_lock lock(mutex);
        return ready.wait_for(lock, std::chrono::seconds(10), [&] { return done; });
    }
    bool completed() {
        std::lock_guard lock(mutex);
        return done;
    }
};

RenderCallbacks threaded_backend(std::shared_ptr<ThreadLookup> lookup,
                                 std::shared_ptr<MetalEngineStatistics> statistics,
                                 std::shared_ptr<metal_backend::SubmissionTestControl> control = {}) {
    MetalEngineOptions options;
    options.mode = MetalEngineMode::Persistent;
    options.lookup = ThreadLookup::lookup;
    options.lookup_context = lookup.get();
    options.lookup_owner = lookup;
    options.statistics = std::move(statistics);
    options.defer_gpu_writeback = true;
    options.max_batch_draws = 8;
    options.guest_memory_host_addressable = true;
    options.cpu_write_tracking = true;
    auto created = metal_backend::create_engine_backend(std::move(options),
        metal_backend::create_pipeline_state_cache, std::move(control));
    CHECK(std::holds_alternative<RenderCallbacks>(created));
    return std::holds_alternative<RenderCallbacks>(created)
        ? std::get<RenderCallbacks>(std::move(created)) : RenderCallbacks{};
}

void verify_engine_threaded_owned_fifo(Fixtures& fixtures) {
    auto probe=std::make_shared<ThreadLookup>();
    probe->fixtures=std::make_shared<Fixtures>(fixtures);
    probe->block=true;
    auto stats=std::make_shared<MetalEngineStatistics>();
    auto backend=threaded_backend(probe,stats);
    if(!backend.draw||!backend.coherence||!backend.ordered_effect||!backend.status) {
        CHECK(!"threaded backend missing callbacks");return;
    }
    auto backing=std::make_shared<Backing>();
    auto stop=snapshot(*backing);
    auto memory=backing->host_memory(backing);
    const auto producer=std::this_thread::get_id();
    CHECK(backend.draw(backend.context,stop,memory).ok());
    probe->await_entry();
    auto second=stop;
    std::atomic<unsigned> queued=0;
    CommandResult second_result;
    std::thread enqueue([&] {
        for (unsigned i = 0; i != 64; ++i) {
            second_result=backend.draw(backend.context,second,memory);
            if (!second_result.ok()) break;
            queued.fetch_add(1,std::memory_order_release);
        }
    });
    // Backend preparation is blocked in lookup, but another producer must be
    // able to append its owned snapshot without waiting for that preparation.
    const auto queued_before_release=await_atomic(queued,64);
    // Both original snapshots may be discarded as soon as enqueue returns.
    stop={};
    second={};
    auto clear=pitch_clear_snapshot(37);
    CHECK(backend.clear(backend.context,clear,memory).ok());
    clear={};
    RenderStop dispatch;
    dispatch.kind=RenderKind::Compute;
    dispatch.compute=ComputeLaunch{};
    CHECK(backend.dispatch(backend.context,dispatch,memory).ok());
    dispatch={};
    probe->release();
    enqueue.join();
    CHECK(queued_before_release);
    CHECK(second_result.ok());
    CHECK(backend.coherence(backend.context,memory,0,0).ok());
    CHECK(backend.status(backend.context).ok());
    CHECK(stats->completed_draws==65);
    CHECK(stats->completed_clears==1);
    CHECK(stats->completed_dispatches==1);
    {
        std::lock_guard lock(probe->mutex);
        CHECK(probe->calls>0);
        CHECK(probe->thread!=producer);
    }
}

void verify_engine_queued_render_snapshots(Fixtures& fixtures) {
    auto probe = std::make_shared<ThreadLookup>();
    probe->fixtures = std::make_shared<Fixtures>(fixtures);
    probe->block = true;
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = threaded_backend(probe, stats);
    auto backing = std::make_shared<Backing>();
    auto memory = backing->host_memory(backing);
    auto stop = snapshot(*backing);
    const auto enqueue = [&](std::uint64_t stream, std::uint64_t serial,
                             std::uint32_t first, std::uint32_t second, bool changed) {
        stop.register_stream = stream;
        stop.register_serial = serial;
        stop.registers[0xe00] = first;
        stop.registers[0xe01] = second;
        stop.changed_registers = {};
        if (changed) stop.changed_registers[0xe00 / 64] = 1ull << (0xe00 % 64);
        CHECK(backend.draw(backend.context, stop, memory).ok());
    };
    enqueue(9001, 1, 1, 17, false);
    probe->await_entry();
    enqueue(9001, 2, 2, 17, true);
    // Serial gaps and stream changes still carry the source's full register
    // array, independent of the sparse changed-register hint.
    enqueue(9001, 4, 4, 29, true);
    enqueue(9002, 1, 5, 40, false);
    enqueue(9002, 2, 5, 40, false);
    enqueue(0, 0, 6, 50, false);
    enqueue(9002, 3, 7, 60, true);
    probe->release();
    CHECK(backend.coherence(backend.context, memory, 0, 0).ok());
    // Recycled queue storage must receive the next source snapshot in full,
    // rather than retaining registers from its previous request.
    enqueue(9002, 4, 8, 60, true);
    stop = {};
    CHECK(backend.coherence(backend.context, memory, 0, 0).ok());

    // A render executed while a publication is held retains the complete
    // queued snapshot after the producer releases its source.
    auto gate = std::make_shared<RetryPublicationGate>();
    GuestEffect blocking;
    blocking.publish = [gate] { return gate->publish(); };
    CHECK(backend.ordered_effect(backend.context, memory, std::move(blocking)).ok());
    gate->await_attempts(1);
    stop = snapshot(*backing);
    enqueue(9003, 1, 9, 70, false);
    stop = {};
    gate->release();
    CHECK(backend.coherence(backend.context, memory, 0, 0).ok());
    const std::vector<std::array<std::uint32_t, 2>> expected{
        {1, 17}, {2, 17}, {4, 29}, {5, 40}, {5, 40}, {6, 50}, {7, 60}, {8, 60}, {9, 70}};
    std::lock_guard lock(probe->mutex);
    CHECK(probe->register_observations == expected);
    CHECK(stats->completed_draws == expected.size());
}

void verify_engine_ordered_write_fifo(Fixtures& fixtures) {
    auto probe = std::make_shared<ThreadLookup>();
    probe->fixtures = std::make_shared<Fixtures>(fixtures);
    probe->block = true;
    constexpr std::uint64_t address = 0x10f000;
    probe->observed_address = address;
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = threaded_backend(probe, stats);
    if (!backend.draw || !backend.coherence || !backend.ordered_write) {
        CHECK(!"threaded backend missing ordered write callbacks");return;
    }
    auto backing = std::make_shared<Backing>();
    auto stop = snapshot(*backing);
    auto memory = backing->host_memory(backing);
    std::array<std::byte, 4> before{};
    std::copy_n(backing->bytes.begin() + 0xf000, before.size(), before.begin());
    std::array<std::byte, 4> payload{
        std::byte{0x21}, std::byte{0x43}, std::byte{0x65}, std::byte{0x87}};
    const auto accepted = payload;

    CHECK(backend.draw(backend.context, stop, memory).ok());
    probe->await_entry();
    CHECK(backend.ordered_write(backend.context, memory, address, payload).ok());
    payload.fill(std::byte{0xee});
    CHECK(backend.draw(backend.context, stop, memory).ok());
    probe->release();
    CHECK(backend.coherence(backend.context, memory, 0, 0).ok());

    {
        std::lock_guard lock(probe->mutex);
        CHECK(probe->observations_valid);
        CHECK(probe->observations.size() == 2);
        if (probe->observations.size() == 2) {
            CHECK(probe->observations[0] == before);
            CHECK(probe->observations[1] == accepted);
        }
    }
    CHECK(std::equal(accepted.begin(), accepted.end(), backing->bytes.begin() + 0xf000));
    CHECK(stats->completed_draws == 2);
}

void verify_engine_ordered_write_read_coherence(Fixtures& fixtures) {
    auto package = std::make_shared<Fixtures>(fixtures);
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = engine_backend(package, stats, {}, 30'000'000'000ull, true, 8, true);
    if (!backend.ordered_effect || !backend.ordered_write || !backend.read_coherence ||
        !backend.coherence) {
        CHECK(!"threaded backend missing ordered write coherence callbacks");return;
    }
    auto backing = std::make_shared<Backing>();
    const auto guest = reinterpret_cast<std::uint64_t>(backing->bytes.data());
    const std::array<Mapping, 2> mappings{{
        {0x100000, backing->bytes.size(), 0, guest, backing->bytes.size(), 0x1000, backing},
        {0x200000, backing->bytes.size(), 0, guest, backing->bytes.size(), 0x1000, backing},
    }};
    auto memory = std::get<GpuAddressSpace>(make_address_space(
        mappings, {Backing::direct_read, Backing::direct_write, nullptr}));
    constexpr std::uint64_t write_address = 0x106000;
    constexpr std::uint64_t write_alias = 0x206000;
    constexpr std::uint64_t disjoint_alias = 0x208000;
    std::array<std::byte, 4> payload{
        std::byte{0x10}, std::byte{0x32}, std::byte{0x54}, std::byte{0x76}};
    std::array<std::byte, 4> disjoint_before{};
    CHECK(memory.read(disjoint_alias, disjoint_before).ok());

    auto gate = std::make_shared<RetryPublicationGate>();
    GuestEffect blocking;
    blocking.publish = [gate] { return gate->publish(); };
    CHECK(backend.ordered_effect(backend.context, memory, std::move(blocking)).ok());
    gate->await_attempts(1);
    CHECK(backend.ordered_write(backend.context, memory, write_address, payload).ok());

    AsyncCommand disjoint;
    std::array<std::byte, 4> disjoint_after{};
    std::thread disjoint_reader([&] {
        disjoint.begin();
        auto result = backend.read_coherence(
            backend.context, memory, disjoint_alias, disjoint_after.size());
        if (result.ok()) {
            const auto read = memory.read(disjoint_alias, disjoint_after);
            if (!read.ok()) result = {CommandStatus::MemoryFault, read.detail};
        }
        disjoint.complete(result);
    });
    disjoint.await_started();
    const bool bypassed = disjoint.await_done();
    CHECK(bypassed);
    if (!bypassed) {
        gate->release();
        disjoint_reader.join();
        return;
    }
    disjoint_reader.join();
    CHECK(disjoint.result.ok());
    CHECK(disjoint_after == disjoint_before);

    AsyncCommand overlapping;
    std::array<std::byte, 4> observed{};
    std::thread overlapping_reader([&] {
        overlapping.begin();
        auto result = backend.read_coherence(
            backend.context, memory, write_alias, observed.size());
        if (result.ok()) {
            const auto read = memory.read(write_alias, observed);
            if (!read.ok()) result = {CommandStatus::MemoryFault, read.detail};
        }
        overlapping.complete(result);
    });
    overlapping.await_started();
    unsigned checkpoint = 0;
    {
        std::lock_guard lock(gate->mutex);
        checkpoint = gate->attempts;
    }
    gate->await_attempts(checkpoint + 1);
    CHECK(!overlapping.completed());
    gate->release();
    overlapping_reader.join();
    CHECK(overlapping.result.ok());
    CHECK(observed == payload);
    CHECK(backend.coherence(backend.context, memory, 0, 0).ok());
}

void verify_engine_ordered_write_byte_backpressure(Fixtures& fixtures) {
    constexpr std::size_t budget = 64u * 1024u * 1024u;
    auto package = std::make_shared<Fixtures>(fixtures);
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = engine_backend(package, stats, {}, 30'000'000'000ull, true, 8, true);
    if (!backend.ordered_effect || !backend.ordered_write || !backend.coherence) {
        CHECK(!"threaded backend missing ordered write backpressure callbacks");return;
    }
    auto backing = std::make_shared<Backing>();
    backing->bytes.resize(budget + 0x1000, std::byte{0});
    auto memory = backing->host_memory(backing);
    auto gate = std::make_shared<RetryPublicationGate>();
    GuestEffect blocking;
    blocking.publish = [gate] { return gate->publish(); };
    CHECK(backend.ordered_effect(backend.context, memory, std::move(blocking)).ok());
    gate->await_attempts(1);

    std::vector<std::byte> payload(budget, std::byte{0x5a});
    CHECK(backend.ordered_write(backend.context, memory, 0x100000, payload).ok());
    const std::array<std::byte, 1> tail{std::byte{0xa5}};
    AsyncCommand extra;
    std::thread producer([&] {
        extra.begin();
        extra.complete(backend.ordered_write(
            backend.context, memory, 0x100000 + budget, tail));
    });
    extra.await_started();
    unsigned checkpoint = 0;
    {
        std::lock_guard lock(gate->mutex);
        checkpoint = gate->attempts;
    }
    gate->await_attempts(checkpoint + 1);
    CHECK(!extra.completed());
    gate->release();
    CHECK(extra.await_done());
    producer.join();
    CHECK(extra.result.ok());
    CHECK(backend.coherence(backend.context, memory, 0, 0).ok());
    CHECK(backing->bytes.front() == payload.front());
    CHECK(backing->bytes[budget - 1] == payload.back());
    CHECK(backing->bytes[budget] == tail.front());
}

void verify_engine_ordered_write_teardown(Fixtures& fixtures) {
    struct AllocationPin {
        std::shared_ptr<std::vector<std::byte>> storage;
    };
    auto package = std::make_shared<Fixtures>(fixtures);
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = engine_backend(package, stats, {}, 30'000'000'000ull, true, 8, true);
    if (!backend.ordered_effect || !backend.ordered_write) {
        CHECK(!"threaded backend missing ordered write teardown callbacks");return;
    }
    auto storage = std::make_shared<std::vector<std::byte>>(0x1000, std::byte{0x19});
    auto pin = std::make_shared<AllocationPin>();
    pin->storage = storage;
    std::weak_ptr<AllocationPin> pin_lifetime = pin;
    std::optional<GpuAddressSpace> memory;
    {
        const Mapping mapping{0x100000, storage->size(), 0,
            reinterpret_cast<std::uint64_t>(storage->data()), storage->size(), 0x1000, pin};
        memory.emplace(std::get<GpuAddressSpace>(make_address_space(
            {&mapping, 1}, {Backing::direct_read, Backing::direct_write, nullptr})));
    }
    auto gate = std::make_shared<RetryPublicationGate>();
    GuestEffect blocking;
    blocking.publish = [gate] { return gate->publish(); };
    CHECK(backend.ordered_effect(backend.context, *memory, std::move(blocking)).ok());
    gate->await_attempts(1);
    const std::array<std::byte, 4> payload{
        std::byte{0xde}, std::byte{0xad}, std::byte{0xbe}, std::byte{0xef}};
    CHECK(backend.ordered_write(backend.context, *memory, 0x100040, payload).ok());
    const auto before = *storage;
    std::weak_ptr<void> backend_lifetime = backend.owner;
    pin.reset();
    memory.reset();
    CHECK(!pin_lifetime.expired());
    backend.owner.reset();
    CHECK(backend_lifetime.expired());
    CHECK(pin_lifetime.expired());
    CHECK(*storage == before);
    {
        std::lock_guard lock(gate->mutex);
        CHECK(gate->publications == 0);
    }
}

void verify_engine_ordered_write_failure(Fixtures& fixtures) {
    auto package = std::make_shared<Fixtures>(fixtures);
    {
        auto stats = std::make_shared<MetalEngineStatistics>();
        auto control = std::make_shared<EngineFeedback>();
        control->fail = true;
        auto backend = engine_backend(package, stats, control,
                                      30'000'000'000ull, true, 8, true);
        if (!backend.draw || !backend.coherence || !backend.status ||
            !backend.ordered_write) {
            CHECK(!"threaded backend missing ordered write failure callbacks");return;
        }
        auto backing = std::make_shared<Backing>();
        auto stop = snapshot(*backing);
        auto memory = backing->host_memory(backing);
        constexpr std::uint64_t address = 0x100000;
        std::array<std::byte, 4> before{};
        std::copy_n(backing->bytes.begin(), before.size(), before.begin());
        const std::array<std::byte, 4> payload{
            std::byte{0xaa}, std::byte{0xbb}, std::byte{0xcc}, std::byte{0xdd}};
        CHECK(backend.draw(backend.context, stop, memory).ok());
        CHECK(backend.ordered_write(backend.context, memory, address, payload).ok());
        control->await_entry();
        CHECK(std::equal(before.begin(), before.end(), backing->bytes.begin()));
        control->release();
        const auto failed = backend.coherence(backend.context, memory, 0, 0);
        CHECK(failed.status == CommandStatus::MemoryFault);
        CHECK(failed.detail == "injected engine GPU failure");
        CHECK(std::equal(before.begin(), before.end(), backing->bytes.begin()));
        const auto sticky = backend.ordered_write(backend.context, memory, address, payload);
        CHECK(sticky.status == failed.status && sticky.detail == failed.detail);
        const auto status = backend.status(backend.context);
        CHECK(status.status == failed.status && status.detail == failed.detail);
    }
    {
        auto stats = std::make_shared<MetalEngineStatistics>();
        auto backend = engine_backend(package, stats, {},
                                      30'000'000'000ull, true, 8, true);
        if (!backend.ordered_write || !backend.status) return;
        auto backing = std::make_shared<Backing>();
        auto memory = backing->host_memory(backing);
        const std::array<std::byte, 4> payload{
            std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}};
        const auto invalid = backend.ordered_write(
            backend.context, memory, 0x200000, payload);
        CHECK(invalid.status == CommandStatus::MemoryFault);
        CHECK(invalid.detail == "Engine queued write range is not backed");
        const auto sticky = backend.status(backend.context);
        CHECK(sticky.status == invalid.status && sticky.detail == invalid.detail);
        const auto cancelled = backend.ordered_write(
            backend.context, memory, 0x10f000, payload);
        CHECK(cancelled.status == invalid.status && cancelled.detail == invalid.detail);
    }
}

void verify_engine_threaded_status_and_backpressure(Fixtures& fixtures) {
    auto package=std::make_shared<Fixtures>(fixtures);
    {
        auto stats=std::make_shared<MetalEngineStatistics>();
        auto control=std::make_shared<EngineFeedback>();
        control->fail=true;
        auto backend=engine_backend(package,stats,control,30'000'000'000ull,true,1,true);
        if(!backend.draw||!backend.status) return;
        auto backing=std::make_shared<Backing>();
        auto stop=snapshot(*backing);
        auto memory=backing->host_memory(backing);
        CHECK(backend.draw(backend.context,stop,memory).ok());
        control->await_entry();
        control->release();
        CommandResult failed;
        const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        do {
            failed=backend.status(backend.context);
            if(!failed.ok()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while(std::chrono::steady_clock::now()<end);
        CHECK(failed.status==CommandStatus::MemoryFault);
        CHECK(failed.detail=="injected engine GPU failure");
    }
    {
        auto stats=std::make_shared<MetalEngineStatistics>();
        auto backend=engine_backend(package,stats,{},30'000'000'000ull,true,8,true);
        if(!backend.blit||!backend.coherence||!backend.status) return;
        auto backing=std::make_shared<Backing>();
        auto memory=backing->host_memory(backing);
        RenderStop invalid;
        invalid.kind=RenderKind::Blit;
        // Acceptance is asynchronous; the worker's exact failure is then sticky.
        CHECK(backend.blit(backend.context,invalid,memory).ok());
        CommandResult failed;
        const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(10);
        do {
            failed=backend.status(backend.context);
            if(!failed.ok()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } while(std::chrono::steady_clock::now()<end);
        CHECK(!failed.ok());
        const auto sticky=backend.coherence(backend.context,memory,0,0);
        CHECK(sticky.status==failed.status&&sticky.detail==failed.detail);
    }
    {
        auto stats=std::make_shared<MetalEngineStatistics>();
        auto backend=engine_backend(package,stats,{},30'000'000'000ull,true,8,true);
        if(!backend.ordered_effect||!backend.coherence) return;
        auto backing=std::make_shared<Backing>();
        auto memory=backing->host_memory(backing);
        auto attempts=std::make_shared<std::atomic<unsigned>>(0);
        auto permit=std::make_shared<std::atomic<bool>>(false);
        GuestEffect gate;
        gate.publish=[attempts,permit]() -> std::optional<CommandResult> {
            attempts->fetch_add(1,std::memory_order_release);
            if(!permit->load(std::memory_order_acquire)) return std::nullopt;
            return CommandResult{};
        };
        CHECK(backend.ordered_effect(backend.context,memory,std::move(gate)).ok());
        if(!await_atomic(*attempts,1)) return;
        std::atomic<unsigned> accepted=0;
        std::atomic<bool> returned=false;
        std::thread producer([&] {
            for(unsigned index=0;index!=4096;++index) {
                if(!backend.ordered_effect(backend.context,memory,{}).ok()) break;
                accepted.fetch_add(1,std::memory_order_release);
            }
            returned.store(true,std::memory_order_release);
        });
        CHECK(await_atomic(accepted,4095));
        CHECK(!returned.load(std::memory_order_acquire));
        permit->store(true,std::memory_order_release);
        producer.join();
        CHECK(returned.load(std::memory_order_acquire));
        CHECK(accepted.load(std::memory_order_acquire)==4096);
        CHECK(backend.coherence(backend.context,memory,0,0).ok());
    }
}

void verify_engine_ordered_effect_idle_pump(Fixtures& fixtures) {
    auto package=std::make_shared<Fixtures>(fixtures);
    auto stats=std::make_shared<MetalEngineStatistics>();
    auto control=std::make_shared<EngineFeedback>();
    auto backend=engine_backend(package,stats,control,30'000'000'000ull,true,8,true);
    if(!backend.draw||!backend.coherence||!backend.ordered_effect) {
        CHECK(!"host-addressable deferred backend missing ordered effects");return;
    }
    auto backing=std::make_shared<Backing>();
    auto stop=snapshot(*backing);
    auto memory=backing->host_memory(backing);
    const auto before=backing->bytes;
    auto published=std::make_shared<std::atomic<unsigned>>(0);
    auto boundary_content=std::make_shared<std::atomic<unsigned>>(0);
    auto finished=std::make_shared<std::atomic<unsigned>>(0);
    GuestEffect increment;
    increment.publish=[published,boundary_content,backing,before]() -> std::optional<CommandResult> {
        boundary_content->store(backing->bytes!=before,std::memory_order_relaxed);
        published->fetch_add(1,std::memory_order_release);return CommandResult{};
    };
    GuestEffect semaphore;
    semaphore.address=0x10fff0;semaphore.size=4;
    const std::uint32_t payload=0x78563412;
    std::memcpy(semaphore.payload.data(),&payload,sizeof(payload));
    GuestEffect completion;
    completion.publish=[finished]() -> std::optional<CommandResult> {
        finished->store(1,std::memory_order_release);return CommandResult{};
    };

    CHECK(backend.draw(backend.context,stop,memory).ok());
    CHECK(backend.ordered_effect(backend.context,memory,std::move(increment)).ok());
    const auto queued_store=backend.ordered_effect(backend.context,memory,std::move(semaphore));
    if(!queued_store.ok()) std::fprintf(stderr,"Ordered store: %.*s\n",
        int(queued_store.detail.size()),queued_store.detail.data());
    CHECK(queued_store.ok());
    CHECK(backend.ordered_effect(backend.context,memory,std::move(completion)).ok());
    control->await_entry();
    CHECK(published->load(std::memory_order_acquire)==0);
    CHECK(backing->bytes==before);
    control->release();
    if(!await_atomic(*finished,1)) return;
    CHECK(published->load(std::memory_order_acquire)==1);
    CHECK(boundary_content->load(std::memory_order_relaxed)==1);
    std::uint32_t stored=0;
    std::memcpy(&stored,backing->bytes.data()+0xfff0,sizeof(stored));
    CHECK(stored==payload);
    CHECK(backend.coherence(backend.context,memory,0,0).ok());
    CHECK(published->load(std::memory_order_acquire)==1);
    CHECK(stats->completed_draws==1);
}

void verify_engine_ordered_semaphore_acquire(Fixtures& fixtures) {
    // An acquire queued behind a release is checked once the release is
    // published; an unmet one stops the channel instead of passing.
    auto package=std::make_shared<Fixtures>(fixtures);
    auto stats=std::make_shared<MetalEngineStatistics>();
    auto control=std::make_shared<EngineFeedback>();
    auto backend=engine_backend(package,stats,control,30'000'000'000ull,true,8,true);
    if(!backend.draw||!backend.coherence||!backend.ordered_effect) {
        CHECK(!"host-addressable deferred backend missing ordered effects");return;
    }
    auto backing=std::make_shared<Backing>();
    auto stop=snapshot(*backing);
    auto memory=backing->host_memory(backing);
    const std::uint32_t payload=0x2468ace0;
    GuestEffect release;
    release.address=0x10fff0;release.size=4;
    std::memcpy(release.payload.data(),&payload,sizeof(payload));
    GuestEffect met;
    met.address=0x10fff0;met.acquire=GuestEffect::Acquire{payload,1};
    auto finished=std::make_shared<std::atomic<unsigned>>(0);
    GuestEffect completion;
    completion.publish=[finished]() -> std::optional<CommandResult> {
        finished->store(1,std::memory_order_release);return CommandResult{};
    };
    CHECK(backend.draw(backend.context,stop,memory).ok());
    CHECK(backend.ordered_effect(backend.context,memory,std::move(release)).ok());
    CHECK(backend.ordered_effect(backend.context,memory,std::move(met)).ok());
    CHECK(backend.ordered_effect(backend.context,memory,std::move(completion)).ok());
    control->await_entry();
    control->release();
    if(!await_atomic(*finished,1)) return;
    CHECK(backend.coherence(backend.context,memory,0,0).ok());

    GuestEffect unmet;
    unmet.address=0x10fff0;unmet.acquire=GuestEffect::Acquire{payload+1,1};
    CHECK(backend.ordered_effect(backend.context,memory,std::move(unmet)).ok());
    const auto stopped=backend.coherence(backend.context,memory,0,0);
    CHECK(!stopped.ok());
    CHECK(stopped.detail.find("acquire dependency is unmet")!=std::string_view::npos);
}

void verify_engine_ordered_effect_fifo(Fixtures& fixtures) {
    auto package=std::make_shared<Fixtures>(fixtures);
    auto stats=std::make_shared<MetalEngineStatistics>();
    auto control=std::make_shared<StagedFeedback>();
    auto backend=engine_backend(package,stats,control,30'000'000'000ull,true,8,true);
    if(!backend.draw||!backend.coherence||!backend.ordered_effect) {
        CHECK(!"host-addressable deferred backend missing ordered effects");return;
    }
    auto backing=std::make_shared<Backing>();
    auto stop=snapshot(*backing);
    auto memory=backing->host_memory(backing);
    const auto before=backing->bytes;
    auto order=std::make_shared<std::atomic<unsigned>>(0);
    const auto effect=[order](unsigned expected,unsigned next) {
        GuestEffect result;
        result.publish=[order,expected,next]() -> std::optional<CommandResult> {
            unsigned current=expected;
            if(!order->compare_exchange_strong(current,next,std::memory_order_release,
                                                std::memory_order_relaxed))
                return CommandResult{CommandStatus::Stopped,"Synthetic guest effect reordered"};
            return CommandResult{};
        };
        return result;
    };

    CHECK(backend.draw(backend.context,stop,memory).ok());
    CHECK(backend.ordered_effect(backend.context,memory,effect(0,1)).ok());
    CHECK(backend.draw(backend.context,stop,memory).ok());
    CHECK(backend.ordered_effect(backend.context,memory,effect(1,2)).ok());
    control->await(1);
    CHECK(order->load(std::memory_order_acquire)==0);
    CHECK(backing->bytes==before);
    control->release(1,{});
    if(!await_atomic(*order,1)) return;
    CHECK(backing->bytes!=before);
    control->await(2);
    CHECK(order->load(std::memory_order_acquire)==1);
    control->release(2,{});
    if(!await_atomic(*order,2)) return;
    CHECK(backend.coherence(backend.context,memory,0,0).ok());
    CHECK(stats->completed_draws==2);
    CHECK(stats->command_buffers==2);
}

// Later clears alias the prefix target. Native encoding must proceed while
// feedback is held, but neither later GPU execution nor guest writeback may
// overwrite the prefix before its publication.
void verify_engine_effect_pipeline(Fixtures& fixtures) {
    for (const unsigned batch : {1u, 8u}) {
        for (const bool fail_prefix : {false, true}) {
            auto stats = std::make_shared<MetalEngineStatistics>();
            auto control = std::make_shared<StagedFeedback>();
            auto backend = engine_backend(std::make_shared<Fixtures>(fixtures), stats,
                control, 30'000'000'000ull, true, batch, true);
            auto backing = std::make_shared<Backing>();
            const auto host = reinterpret_cast<std::uint64_t>(backing->bytes.data());
            const std::array<Mapping, 2> mappings{{
                {0x100000, backing->bytes.size(), 0, host, backing->bytes.size(), 0x1000, backing},
                {0x200000, backing->bytes.size(), 0, host, backing->bytes.size(), 0x1000, backing},
            }};
            auto memory = std::get<GpuAddressSpace>(make_address_space(
                mappings, {Backing::direct_read, Backing::direct_write, nullptr}));
            auto first = pitch_clear_snapshot(37);
            auto second = first;
            second.color_targets[0].address = 0x200000;
            second.registers[0x360] = 0;
            second.registers[0x361] = std::bit_cast<std::uint32_t>(1.f);
            second.registers[0x362] = 0;
            Backing expected = *backing;
            CHECK(execute_metal_color_clear(nullptr, first, expected.memory()).ok());
            const auto initial = backing->bytes;
            auto published = std::make_shared<std::atomic<unsigned>>(0);
            const auto effect = [published](unsigned expected_value) {
                GuestEffect result;
                result.publish = [published, expected_value]() -> std::optional<CommandResult> {
                    if (published->load(std::memory_order_acquire) != expected_value)
                        return CommandResult{CommandStatus::Stopped, "Synthetic pipeline effect reordered"};
                    published->store(expected_value + 1, std::memory_order_release);
                    return CommandResult{};
                };
                return result;
            };
            CHECK(backend.clear(backend.context, first, memory).ok());
            CHECK(backend.ordered_effect(backend.context, memory, effect(0)).ok());
            control->await(1);
            CHECK(backend.clear(backend.context, second, memory).ok());
            CHECK(backend.ordered_effect(backend.context, memory, effect(1)).ok());
            CHECK(control->await_encoded(2));
            CHECK(published->load(std::memory_order_acquire) == 0);
            CHECK(backing->bytes == initial);
            {
                std::lock_guard lock(control->mutex);
                CHECK(control->committed == 1);
            }
            if (fail_prefix) {
                control->release(1, {CommandStatus::MemoryFault, "Synthetic pipeline prefix failed"});
                const auto result = backend.coherence(backend.context, memory, 0, 0);
                CHECK(result.status == CommandStatus::MemoryFault);
                CHECK(result.detail == "Synthetic pipeline prefix failed");
                backend = {};
                CHECK(published->load(std::memory_order_acquire) == 0);
                CHECK(backing->bytes == initial);
                std::lock_guard lock(control->mutex);
                CHECK(control->committed == 1);
                continue;
            }
            control->release(1, {});
            CHECK(await_atomic(*published, 1));
            control->await(2);
            CHECK(published->load(std::memory_order_acquire) == 1);
            CHECK(backing->bytes == expected.bytes);
            control->release(2, {});
            CHECK(await_atomic(*published, 2));
            CHECK(backend.coherence(backend.context, memory, 0, 0).ok());
            second.color_targets[0].address = 0x100000;
            CHECK(execute_metal_color_clear(nullptr, second, expected.memory()).ok());
            CHECK(backing->bytes == expected.bytes);
            CHECK(stats->completed_clears == 2);
        }
    }
}

void verify_engine_store_effect_pipeline(Fixtures& fixtures) {
    for (const unsigned batch : {1u, 8u}) {
        auto stats = std::make_shared<MetalEngineStatistics>();
        auto control = std::make_shared<StagedFeedback>();
        auto backend = engine_backend(std::make_shared<Fixtures>(fixtures), stats,
            control, 30'000'000'000ull, true, batch, true);
        auto backing = std::make_shared<Backing>();
        auto memory = backing->host_memory(backing);
        auto first = pitch_clear_snapshot(37);
        auto second = first;
        second.registers[0x360] = 0;
        second.registers[0x361] = std::bit_cast<std::uint32_t>(1.f);
        second.registers[0x362] = 0;
        Backing expected = *backing;
        CHECK(execute_metal_color_clear(nullptr, first, expected.memory()).ok());
        GuestEffect store;
        store.address = 0x10f000;
        store.size = 1;
        store.payload[0] = std::byte{0x5a};
        auto published = std::make_shared<std::atomic<unsigned>>(0);
        GuestEffect end;
        end.publish = [published]() -> std::optional<CommandResult> {
            published->store(1, std::memory_order_release);
            return CommandResult{};
        };
        CHECK(backend.clear(backend.context, first, memory).ok());
        CHECK(backend.ordered_effect(backend.context, memory, std::move(store)).ok());
        control->await(1);
        CHECK(backend.clear(backend.context, second, memory).ok());
        CHECK(backend.ordered_effect(backend.context, memory, std::move(end)).ok());
        CHECK(control->await_encoded(2));
        CHECK(backing->bytes[0xf000] == std::byte{0x19});
        control->release(1, {});
        control->await(2);
        expected.bytes[0xf000] = std::byte{0x5a};
        CHECK(backing->bytes == expected.bytes);
        CHECK(published->load(std::memory_order_acquire) == 0);
        control->release(2, {});
        CHECK(await_atomic(*published, 1));
        CHECK(backend.coherence(backend.context, memory, 0, 0).ok());
        CHECK(execute_metal_color_clear(nullptr, second, expected.memory()).ok());
        CHECK(backing->bytes == expected.bytes);
    }
}

void verify_engine_store_effect_shader_dependency(Fixtures& fixtures) {
    auto probe = std::make_shared<ThreadLookup>();
    probe->fixtures = std::make_shared<Fixtures>(fixtures);
    probe->observed_address = 0x20f000;
    auto control = std::make_shared<EngineFeedback>();
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = threaded_backend(probe, stats, control);
    auto backing = std::make_shared<Backing>();
    const auto host = reinterpret_cast<std::uint64_t>(backing->bytes.data());
    const std::array<Mapping, 2> mappings{{
        {0x100000, backing->bytes.size(), 0, host, backing->bytes.size(), 0x1000, backing},
        {0x200000, backing->bytes.size(), 0, host, backing->bytes.size(), 0x1000, backing},
    }};
    auto memory = std::get<GpuAddressSpace>(make_address_space(
        mappings, {Backing::direct_read, Backing::direct_write, nullptr}));
    auto stop = snapshot(*backing);
    GuestEffect store;
    store.address = 0x10f000;
    store.size = 4;
    const std::array<std::byte, 4> expected{std::byte{1}, std::byte{3}, std::byte{5}, std::byte{7}};
    std::copy(expected.begin(), expected.end(), store.payload.begin());
    CHECK(backend.draw(backend.context, stop, memory).ok());
    CHECK(backend.ordered_effect(backend.context, memory, std::move(store)).ok());
    control->await_entry();
    CHECK(backend.draw(backend.context, stop, memory).ok());
    {
        std::unique_lock lock(probe->mutex);
        CHECK(probe->ready.wait_for(lock, std::chrono::seconds(10), [&] { return probe->blocked_reads != 0; }));
        CHECK(probe->observations.size() == 1);
    }
    control->release();
    CHECK(backend.coherence(backend.context, memory, 0, 0).ok());
    {
        std::lock_guard lock(probe->mutex);
        CHECK(probe->observations.size() == 2);
        if (probe->observations.size() == 2) CHECK(probe->observations[1] == expected);
    }
    CHECK(stats->completed_draws == 2);
}

void verify_engine_pipelined_read_barrier(Fixtures& fixtures) {
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto control = std::make_shared<StagedFeedback>();
    auto backend = engine_backend(std::make_shared<Fixtures>(fixtures), stats,
        control, 30'000'000'000ull, true, 8, true);
    auto backing = std::make_shared<Backing>();
    auto memory = backing->host_memory(backing);
    auto gate = std::make_shared<RetryPublicationGate>();
    GuestEffect effect;
    effect.publish = [gate] { return gate->publish(); };
    CHECK(backend.ordered_effect(backend.context, memory, std::move(effect)).ok());
    gate->await_attempts(1);
    CHECK(backend.clear(backend.context, pitch_clear_snapshot(37), memory).ok());
    CHECK(control->await_encoded(1));
    unsigned checkpoint;
    {
        std::lock_guard lock(gate->mutex);
        checkpoint = gate->attempts;
    }
    gate->await_attempts(checkpoint + 1);
    CHECK(backend.read_coherence(backend.context, memory, 0x10f000, 4).ok());
    std::array<AsyncCommand, 8> reads;
    std::vector<std::thread> readers;
    for (auto& read : reads) {
        readers.emplace_back([&, command = &read] {
            command->begin();
            command->complete(backend.read_coherence(backend.context, memory, 0x100000, 4));
        });
    }
    for (auto& read : reads) read.await_started();
    {
        std::lock_guard lock(gate->mutex);
        checkpoint = gate->attempts;
    }
    gate->await_attempts(checkpoint + 1);
    for (auto& read : reads) CHECK(!read.completed());
    CHECK(backend.status(backend.context).ok());
    gate->release();
    control->await(1);
    control->release(1, {});
    for (auto& read : reads) CHECK(read.await_done());
    for (auto& reader : readers) reader.join();
    for (const auto& read : reads) CHECK(read.result.ok());
    CHECK(backend.coherence(backend.context, memory, 0, 0).ok());
    {
        std::lock_guard lock(gate->mutex);
        CHECK(gate->publications == 1);
    }
}

void verify_engine_ordered_effect_retry_and_failure(Fixtures& fixtures) {
    auto package=std::make_shared<Fixtures>(fixtures);
    {
        auto stats=std::make_shared<MetalEngineStatistics>();
        auto control=std::make_shared<EngineFeedback>();
        auto backend=engine_backend(package,stats,control,30'000'000'000ull,true,8,true);
        if(!backend.draw||!backend.coherence||!backend.ordered_effect) {
            CHECK(!"host-addressable deferred backend missing ordered effects");return;
        }
        auto backing=std::make_shared<Backing>();
        auto stop=snapshot(*backing);
        auto memory=backing->host_memory(backing);
        auto attempts=std::make_shared<std::atomic<unsigned>>(0);
        auto published=std::make_shared<std::atomic<unsigned>>(0);
        auto permit=std::make_shared<std::atomic<bool>>(false);
        std::atomic<bool> returned=false;
        GuestEffect effect;
        effect.publish=[attempts,published,permit]() -> std::optional<CommandResult> {
            attempts->fetch_add(1,std::memory_order_release);
            if(!permit->load(std::memory_order_acquire)) return std::nullopt;
            published->store(1,std::memory_order_release);return CommandResult{};
        };
        CHECK(backend.draw(backend.context,stop,memory).ok());
        CHECK(backend.ordered_effect(backend.context,memory,std::move(effect)).ok());
        CommandResult coherent;
        std::thread waiter([&] {
            coherent=backend.coherence(backend.context,memory,0,0);
            returned.store(true,std::memory_order_release);
        });
        control->await_entry();control->release();
        CHECK(await_atomic(*attempts,1));
        CHECK(!returned.load(std::memory_order_acquire));
        permit->store(true,std::memory_order_release);
        waiter.join();
        CHECK(coherent.ok());
        CHECK(returned.load(std::memory_order_acquire));
        CHECK(published->load(std::memory_order_acquire)==1);
    }
    {
        auto stats=std::make_shared<MetalEngineStatistics>();
        auto control=std::make_shared<EngineFeedback>();control->fail=true;
        auto backend=engine_backend(package,stats,control,30'000'000'000ull,true,8,true);
        if(!backend.draw||!backend.coherence||!backend.ordered_effect) return;
        auto backing=std::make_shared<Backing>();
        auto stop=snapshot(*backing);
        auto memory=backing->host_memory(backing);
        const auto before=backing->bytes;
        auto published=std::make_shared<std::atomic<unsigned>>(0);
        GuestEffect effect;
        effect.publish=[published]() -> std::optional<CommandResult> {
            published->store(1,std::memory_order_release);return CommandResult{};
        };
        CHECK(backend.draw(backend.context,stop,memory).ok());
        CHECK(backend.ordered_effect(backend.context,memory,std::move(effect)).ok());
        control->await_entry();control->release();
        const auto failed=backend.coherence(backend.context,memory,0,0);
        CHECK(failed.status==CommandStatus::MemoryFault);
        CHECK(failed.detail=="injected engine GPU failure");
        CHECK(published->load(std::memory_order_acquire)==0);
        CHECK(backing->bytes==before);
        CHECK(stats->completed_draws==0);
    }
}

void verify_engine_publication_releases_last_owner(Fixtures& fixtures) {
    auto package=std::make_shared<Fixtures>(fixtures);
    auto stats=std::make_shared<MetalEngineStatistics>();
    auto backend=engine_backend(package,stats,{},30'000'000'000ull,true,8,true);
    if(!backend.ordered_effect||!backend.status) {
        CHECK(!"host-addressable deferred backend missing threaded callbacks");return;
    }
    auto backing=std::make_shared<Backing>();
    auto memory=backing->host_memory(backing);
    std::weak_ptr<void> backend_lifetime=backend.owner;
    auto captured_owner=backend.owner;
    backend.owner.reset();
    auto published=std::make_shared<std::atomic<unsigned>>(0);
    GuestEffect effect;
    effect.publish=[captured_owner=std::move(captured_owner),published]() -> std::optional<CommandResult> {
        published->store(1,std::memory_order_release);
        return CommandResult{};
    };
    CHECK(backend.ordered_effect(backend.context,memory,std::move(effect)).ok());
    if(!await_atomic(*published,1)) return;
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
    while(!backend_lifetime.expired() && std::chrono::steady_clock::now()<deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    CHECK(backend_lifetime.expired());
}

void verify_engine_command_read_preserves_write_generation(Fixtures& fixtures) {
    using armigo::memory::WriteTracker;
    auto package = std::make_shared<Fixtures>(fixtures);
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = engine_backend(package, stats, {}, 30'000'000'000ull, true, 8, true);
    if (!backend.draw || !backend.coherence || !backend.read_coherence) {
        CHECK(!"deferred backend missing read coherence");return;
    }
    Backing initial;
    const auto stop = snapshot(initial);
    const auto size = initial.bytes.size();
    // Dedicated host pages keep allocator writes outside the tracked range.
    void* storage = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(storage != MAP_FAILED);
    if (storage == MAP_FAILED) return;
    auto owner = std::shared_ptr<void>(storage, [size](void* address) { munmap(address, size); });
    std::memcpy(storage, initial.bytes.data(), size);
    const Mapping mapping{0x100000, size, 0, reinterpret_cast<std::uint64_t>(storage),
                          size, 0x1000, owner};
    auto memory = std::get<GpuAddressSpace>(make_address_space({&mapping, 1},
        {Backing::direct_read, Backing::direct_write, nullptr}));
    // Pin a Metal view before registration creates the writable Mach alias.
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLBuffer> guest_buffer = [device newBufferWithBytesNoCopy:storage length:size
        options:MTLResourceStorageModeShared deallocator:nil];
    CHECK(guest_buffer != nil);
    CHECK(backend.draw(backend.context, stop, memory).ok());
    const auto* command = static_cast<const std::byte*>(storage) + 0x5000;
    // The first read barrier also proves that the asynchronous draw registered
    // this view before the test inspects its write-tracker state.
    CHECK(backend.read_coherence(backend.context, memory, 0x105000, 4).ok());
    CHECK(WriteTracker::tracked(command, 4));
    CHECK(WriteTracker::protect_checked(command, 4));
    const auto stamp = WriteTracker::now();
    CHECK(!WriteTracker::written_since(command, 4, stamp));
    CHECK(backend.read_coherence(backend.context, memory, 0x105000, 4).ok());
    std::array<std::byte, 4> fetched{};
    CHECK(memory.read(0x105000, fetched).ok());
    CHECK(std::equal(fetched.begin(), fetched.end(), initial.bytes.begin() + 0x5000));
    CHECK(WriteTracker::now() == stamp);
    CHECK(!WriteTracker::written_since(command, 4, stamp));
    CHECK(stats->command_buffers == 0);
    CHECK(stats->completed_draws == 0);

    // The write path must still invalidate that same certificate.
    CHECK(backend.coherence(backend.context, memory, 0x105000, 4).ok());
    CHECK(WriteTracker::written_since(command, 4, stamp));

    // A read overlapping GPU output still completes and resolves the draw.
    Backing expected = initial;
    auto expected_memory = expected.memory();
    MetalDrawContext reference{Fixtures::lookup, package.get(), nullptr};
    CHECK(execute_metal_draw(&reference, stop, expected_memory).ok());
    CHECK(backend.read_coherence(backend.context, memory, 0x100000, 512).ok());
    CHECK(stats->command_buffers == 1);
    CHECK(stats->completed_draws == 1);
    CHECK(std::equal(expected.bytes.begin(), expected.bytes.begin() + 512,
                     static_cast<const std::byte*>(storage)));

    CHECK(WriteTracker::has_writable_alias(command, 4));
    CHECK(WriteTracker::protect_checked(command, 4));
    const auto alias_stamp = WriteTracker::now();
    const auto alias_stats = WriteTracker::stats();
    const std::array<std::byte, 4> payload{
        std::byte{0x91}, std::byte{0x82}, std::byte{0x73}, std::byte{0x64}};
    CHECK(backend.ordered_write(backend.context, memory, 0x105000, payload).ok());
    CHECK(backend.read_coherence(backend.context, memory, 0x105000, 4).ok());
    CHECK(std::equal(payload.begin(), payload.end(), command));
    CHECK(WriteTracker::written_since(command, 4, alias_stamp));
    CHECK(WriteTracker::stats().protect_calls == alias_stats.protect_calls);
    CHECK(WriteTracker::stats().faults == alias_stats.faults);
    CHECK(WriteTracker::stats().prepared == alias_stats.prepared);

    // A Metal no-copy view of the original guest VA sees alias writes on the
    // device, without replacing its buffer or relaxing guest protection.
    id<MTLBuffer> copied = [device newBufferWithLength:4 options:MTLResourceStorageModeShared];
    CHECK(guest_buffer != nil && copied != nil);
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> copy = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [copy blitCommandEncoder];
    [blit copyFromBuffer:guest_buffer sourceOffset:0x5000 toBuffer:copied destinationOffset:0 size:4];
    [blit endEncoding];
    [copy commit];
    [copy waitUntilCompleted];
    CHECK(copy.status == MTLCommandBufferStatusCompleted);
    CHECK(std::memcmp(copied.contents, payload.data(), payload.size()) == 0);

    const auto effect_stamp = WriteTracker::now();
    GuestEffect effect;
    effect.address = 0x105008;
    effect.size = payload.size();
    std::copy(payload.begin(), payload.end(), effect.payload.begin());
    CHECK(backend.ordered_effect(backend.context, memory, std::move(effect)).ok());
    CHECK(backend.read_coherence(backend.context, memory, 0x105008, 4).ok());
    CHECK(std::equal(payload.begin(), payload.end(), command + 8));
    CHECK(WriteTracker::written_since(command + 8, 4, effect_stamp));
    CHECK(WriteTracker::stats().protect_calls == alias_stats.protect_calls);
    const auto guest_stamp = WriteTracker::now();
    static_cast<volatile std::byte*>(storage)[0x5010] = std::byte{0x55};
    CHECK(WriteTracker::stats().faults == alias_stats.faults + 1);
    CHECK(WriteTracker::written_since(command, 4, guest_stamp));
}

void verify_engine_synchronous_alias_write(Fixtures& fixtures) {
    using armigo::memory::WriteTracker;
    auto package = std::make_shared<Fixtures>(fixtures);
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto backend = engine_backend(package, stats, {}, 30'000'000'000ull, false, 8, true);
    CHECK(backend.ordered_write && !backend.ordered_effect);
    Backing initial;
    const auto stop = snapshot(initial);
    const auto size = initial.bytes.size();
    void* storage = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    CHECK(storage != MAP_FAILED);
    if (storage == MAP_FAILED || !backend.ordered_write) return;
    auto owner = std::shared_ptr<void>(storage, [size](void* address) { munmap(address, size); });
    std::memcpy(storage, initial.bytes.data(), size);
    const Mapping mapping{0x100000, size, 0, reinterpret_cast<std::uint64_t>(storage),
                          size, 0x1000, owner};
    auto memory = std::get<GpuAddressSpace>(make_address_space({&mapping, 1},
        {Backing::direct_read, Backing::direct_write, nullptr}));
    CHECK(backend.draw(backend.context, stop, memory).ok());
    auto* target = static_cast<std::byte*>(storage) + 0x5000;
    CHECK(WriteTracker::has_writable_alias(target, 4));
    CHECK(WriteTracker::protect_checked(target, 4));
    const auto stamp = WriteTracker::now();
    const auto before = WriteTracker::stats();
    const std::array<std::byte, 4> payload{std::byte{1}, std::byte{3}, std::byte{5}, std::byte{7}};
    CHECK(backend.ordered_write(backend.context, memory, 0x105000, payload).ok());
    CHECK(std::equal(payload.begin(), payload.end(), target));
    CHECK(WriteTracker::written_since(target, 4, stamp));
    CHECK(WriteTracker::stats().protect_calls == before.protect_calls);
    CHECK(WriteTracker::stats().faults == before.faults);
    static_cast<volatile std::byte*>(storage)[0x5008] = std::byte{0x55};
    CHECK(WriteTracker::stats().faults == before.faults + 1);
    CHECK(!backend.ordered_write(backend.context, memory, 0, payload).ok());
    CHECK(!backend.ordered_write(backend.context, memory, 0x105000, payload).ok());

}

void verify_engine_published_mapping_releases_owner(Fixtures& fixtures) {
    auto package=std::make_shared<Fixtures>(fixtures);
    auto stats=std::make_shared<MetalEngineStatistics>();
    auto backend=engine_backend(package,stats,{},30'000'000'000ull,true,8,true);
    if(!backend.ordered_effect||!backend.memory_changed) {
        CHECK(!"host-addressable deferred backend missing mapping callbacks");return;
    }
    auto backing=std::make_shared<Backing>();
    std::weak_ptr<Backing> backing_lifetime=backing;
    auto memory=std::make_optional(backing->host_memory(backing));
    auto published=std::make_shared<std::atomic<unsigned>>(0);
    GuestEffect effect;
    effect.publish=[published]() -> std::optional<CommandResult> {
        published->store(1,std::memory_order_release);
        return CommandResult{};
    };
    CHECK(backend.ordered_effect(backend.context,*memory,std::move(effect)).ok());
    if(!await_atomic(*published,1)) return;
    {
        const std::array<std::shared_ptr<void>,1> released{backing};
        CHECK(backend.memory_changed(backend.context,memory->mappings(),released,
                                     memory->backing_callbacks()).ok());
    }
    memory.reset();
    backing.reset();
    CHECK(backing_lifetime.expired());
    CHECK(static_cast<bool>(backend.owner));
}

// Two outstanding deferred batches, the second's GPU feedback injected to
// fail: the first batch's receipts still complete successfully and convert
// into statistics, and the injected failure surfaces once with its original
// diagnostic, attributed to the second batch.
void verify_engine_batch_failure_attribution(Fixtures& fixtures) {
    auto package = std::make_shared<Fixtures>(fixtures);
    auto stats = std::make_shared<MetalEngineStatistics>();
    auto control = std::make_shared<StagedFeedback>();
    auto backend = engine_backend(package, stats, control, 30'000'000'000ull, true, 2);
    if (!backend.draw) {CHECK(!"deferred backend missing draw");return;}
    auto backing = std::make_shared<Backing>();
    auto stop = snapshot(*backing);
    auto memory = backing->memory(backing);
    // The budget cap commits the first batch; its feedback blocks unpublished.
    CHECK(backend.draw(backend.context, stop, memory).ok());
    CHECK(backend.draw(backend.context, stop, memory).ok());
    control->await(1);
    CHECK(stats->command_buffers == 1);
    // The second batch commits while the first is still outstanding.
    CHECK(backend.draw(backend.context, stop, memory).ok());
    CHECK(backend.draw(backend.context, stop, memory).ok());
    CHECK(stats->command_buffers == 2);
    CHECK(stats->completed_draws == 0);
    CHECK(backing->writes == 0);
    // The first batch completes only after the second batch's failure is
    // recorded, so its retirement runs against a settled second-batch failure.
    control->release(1, {});
    control->release(2, {CommandStatus::MemoryFault, "injected second batch failure"});
    // The failure surfaces once for the second batch and then stays sticky.
    CommandResult surfaced, sticky;
    try {
        auto surfaced_memory = backing->memory(backing);
        surfaced = backend.draw(backend.context, stop, surfaced_memory);
    } catch (...) {CHECK(!"batch failure surface threw");return;}
    CHECK(surfaced.status == CommandStatus::MemoryFault);
    CHECK(surfaced.detail == "injected second batch failure");
    auto sticky_memory = backing->memory(backing);
    sticky = backend.draw(backend.context, stop, sticky_memory);
    CHECK(sticky.status == surfaced.status && sticky.detail == surfaced.detail);
    CHECK(backing->writes == 0);
    // Destruction retires the receipts: both first-batch draws completed and
    // convert, both second-batch draws stay failed.
    backend = {};
    CHECK(stats->completed_draws == 2);
    CHECK(stats->command_buffers == 2);
}

void verify_mrt_admission(Fixtures& fixtures) {
    const auto saved=fixtures.fragment;
    const auto stopped=[&](const CompiledGraphicsStage& fragment) {
        fixtures.fragment=std::make_shared<CompiledGraphicsStage>(fragment);
        Backing backing;
        const auto stop=snapshot(backing);
        auto memory=backing.memory();
        MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
        const auto result=execute_metal_draw(&context,stop,memory);
        if(!result.ok()&&result.status!=CommandStatus::Unsupported)
            std::fprintf(stderr,"MRT stop status: %.*s\n",int(result.detail.size()),result.detail.data());
        CHECK(!result.ok()&&result.status==CommandStatus::Unsupported);
        CHECK(stats.completed_draws==0&&backing.writes==0);
    };
    auto targets=*saved;
    targets.color_outputs={{0,15,ColorOutputScalar::f32,4},{2,15,ColorOutputScalar::f32,4}};
    targets.interface.color_output_mask=0x0f0f;
    stopped(targets);
    auto typed=*saved;
    typed.color_outputs={{0,15,ColorOutputScalar::u32,4}};
    stopped(typed);
    auto sparse=*saved;
    sparse.color_outputs={{0,5,ColorOutputScalar::f32,4}};
    sparse.interface.color_output_mask=5;
    stopped(sparse);
    auto depth=*saved;
    depth.writes_fragment_depth=true;
    stopped(depth);
    auto sample=*saved;
    sample.writes_sample_mask=true;
    stopped(sample);
    auto disagree=*saved;
    disagree.color_outputs={};
    stopped(disagree);
    fixtures.fragment=saved;
    Backing backing;
    const auto stop=snapshot(backing);
    auto memory=backing.memory();
    MetalDrawStatistics stats;MetalDrawContext context{Fixtures::lookup,&fixtures,&stats};
    const auto result=execute_metal_draw(&context,stop,memory);
    if(!result.ok())std::fprintf(stderr,"MRT baseline: %.*s\n",int(result.detail.size()),result.detail.data());
    CHECK(result.ok()&&stats.completed_draws==1);
}
}  // namespace
int main(int argc,char** argv) {
    try {
        Fixtures fixtures;
        ClassFixtures class_fixtures;
        if(argc==3 && std::string_view(argv[1])=="--emit-fixtures") {
            std::filesystem::create_directories(argv[2]);
            std::ofstream(std::filesystem::path(argv[2])/"vertex.metal")<<fixtures.vertex_source;
            std::ofstream(std::filesystem::path(argv[2])/"fragment.metal")
                <<fixtures.fragment_source<<class_fixtures.class_vertex_source
                <<class_fixtures.class_fragment_source;
            std::ofstream(std::filesystem::path(argv[2])/"depth_fragment.metal")<<depth_fragment_source;
            return 0;
        }
        if(argc!=2){std::fputs("Usage: armigo_metal_draw_tests [--emit-fixtures] DIRECTORY\n",stderr);return 2;}
        fixtures.load(argv[1]);class_fixtures.load(argv[1]);
        verify(fixtures);verify_texture_handle_fact(fixtures);
        verify_texture_descriptor_facts(fixtures);verify_rgba8_sampling(fixtures);
        verify_persistent_target_views(fixtures);
        verify_depth_sampling(fixtures,argv[1]);
        verify_readback(fixtures);verify_bc1_sampling(fixtures);verify_alpha_blending(fixtures);
        verify_runtime_constant_buffers(fixtures);verify_class_inputs(class_fixtures);
        verify_mrt_admission(fixtures);
        verify_engine_parity(fixtures);verify_engine_mapping_notifications(fixtures);
        verify_null_texture_remap_sequence(fixtures);
        verify_engine_completion(fixtures);verify_engine_selection(fixtures);
        verify_engine_deferred_batches(fixtures);
        verify_engine_batch_freshness(fixtures);
        verify_engine_threaded_owned_fifo(fixtures);
        verify_engine_queued_render_snapshots(fixtures);
        verify_engine_ordered_write_fifo(fixtures);
        verify_engine_ordered_write_read_coherence(fixtures);
        verify_engine_ordered_write_byte_backpressure(fixtures);
        verify_engine_ordered_write_teardown(fixtures);
        verify_engine_ordered_write_failure(fixtures);
        verify_engine_threaded_status_and_backpressure(fixtures);
        verify_engine_ordered_effect_idle_pump(fixtures);
        verify_engine_ordered_effect_fifo(fixtures);
        verify_engine_ordered_semaphore_acquire(fixtures);
        verify_engine_effect_pipeline(fixtures);
        verify_engine_store_effect_pipeline(fixtures);
        verify_engine_store_effect_shader_dependency(fixtures);
        verify_engine_pipelined_read_barrier(fixtures);
        verify_engine_ordered_effect_retry_and_failure(fixtures);
        verify_engine_publication_releases_last_owner(fixtures);
        verify_engine_command_read_preserves_write_generation(fixtures);
        verify_engine_synchronous_alias_write(fixtures);
        verify_engine_published_mapping_releases_owner(fixtures);
        verify_engine_batch_failure_attribution(fixtures);
        verify_engine_pitch_clear_padding(fixtures);verify_engine_clear_scissor_parity(fixtures);
        std::printf("Metal draw verification: %u failures.\n",failures);return failures?1:0;
    } catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
