// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "../../src/runtime/native_preparation.h"
#include <armigo/aarch64/register_effects.h>
#include <armigo/execution/guest_entry.h>
#include <armigo/execution/gate_code.h>
#include <armigo/execution/reachable_code.h>
#include <armigo/platform/host_module.h>
#include <armigo/platform/thread_slot.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <optional>
#include <vector>

namespace {
using namespace armigo;
namespace a64 = aarch64;
unsigned failures = 0;
#define CHECK(x) do { if (!(x)) { ++failures; std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); } } while(false)
constexpr auto lr = a64::register_bit(30);
void effects() {
    struct Case { std::uint32_t word, reads, writes; bool sp; unsigned implicit; };
    const Case cases[] = {
        {0xd65f0fff,lr,0,true,2}, {0xd503237f,lr,lr,true,2},
        {0xd50323ff,lr,lr,true,2}, {0xd50323bf,lr,lr,true,2}, {0xd50320ff,lr,lr,false,1},
        {0xd503219f,(1u<<16)|(1u<<17),1u<<17,false,2},
        {0xd50321df,(1u<<16)|(1u<<17),1u<<17,false,2},
        {0xdac10020,3,1,false,0}, {0xdac10420,3,1,false,0}, {0xdac11420,3,1,false,0},
        {0xdac107fe,lr,lr,true,0}, {0xdac143e0,1,1,false,0},
        {0x9ac23020,6,1,false,0}, {0x9adf3020,2,1,true,0},
        {0xdac10432,(1u<<18)|2,1u<<18,false,0}, {0xdac107f2,1u<<18,1u<<18,true,0}
    };
    for (auto c : cases) {
        const auto e = a64::decode_register_effects(c.word);
        CHECK(e.decoded() && e.reads == c.reads && e.writes == c.writes);
        CHECK(a64::reads_stack_pointer(e) == c.sp && !a64::writes_stack_pointer(e));
        CHECK(!e.reads_flags && !e.writes_flags && e.memory.operation == a64::MemoryOperation::none);
        unsigned implicit = 0;
        for (unsigned i=0;i<e.operand_count;++i) implicit += e.operands[i].field == a64::implicit_operand;
        CHECK(implicit == c.implicit);
        CHECK(e.control_flow == (c.word == 0xd65f0fff ? a64::ControlFlow::return_ : a64::ControlFlow::none));
    }
    // Register 31 is SP only in the modifier position, never the destination.
    for (unsigned rd=0;rd<32;++rd) for (unsigned rn=0;rn<32;++rn) {
        const auto e=a64::decode_register_effects(0xdac10400u | (rn<<5) | rd);
        CHECK(e.decoded() && e.reads==(a64::register_bit(rd)|a64::register_bit(rn)) &&
            e.writes==a64::register_bit(rd) && a64::reads_stack_pointer(e)==(rn==31));
    }
    const auto ret = a64::decode(0xd65f0fff);
    CHECK(ret.classified() && ret.reg == 30 && ret.reads == lr && ret.writes == 0);
}

void return_flow() {
    const std::uint32_t code[]{0xd65f0fff,0xd4200000};
    const execution::ReachableImage image{0x1000,std::as_bytes(std::span(code)),{{0x1000,0x1008}}};
    const std::uint64_t entry=0x1000;
    const auto flow=execution::collect_reachable_code(std::span(&image,1),std::span(&entry,1));
    CHECK(flow.complete() && flow.sites.size()==1 && flow.sites[0].kind==execution::SiteKind::return_);
}

std::uint64_t host_x18() {
#if defined(__aarch64__)
    std::uint64_t value; asm volatile("mov %0, x18" : "=r"(value)); return value;
#else
    return 0;
#endif
}

struct Image {
    std::optional<platform::HostModule> mapping;
    runtime::native::ModuleCodePreparation preparation;
};
Image prepare(std::span<const std::uint32_t> words, platform::ThreadSlotKey& key, bool rewrite) {
    const auto page = platform::HostModule::native_page_size().value();
    std::vector<std::byte> bytes(2*page);
    std::memcpy(bytes.data(),words.data(),words.size_bytes());
    const platform::ModuleRegion region{0,bytes.size(),memory::Permission::read|memory::Permission::execute};
    auto made = platform::HostModule::create(bytes,std::span(&region,1));
    CHECK(made.ok()); if(!made.ok())return {};
    Image result{std::move(made.module),{}};
    if (rewrite) {
        auto& prep=result.preparation;
        prep.slot=key.native_access()->tsd_byte_offset;
        prep.text_size=words.size_bytes(); prep.gate=execution::make_local_gate_code();
        linker::DynamicModule metadata;
        metadata.symbols.push_back({"authored_entry",0,words.size_bytes(),1,0x12,0});
        auto writable=result.mapping->preparation_bytes();
        CHECK(prep.scan_text(writable.first(words.size_bytes()),1,writable.first(page)) <= page);
        const std::uintptr_t bases[]{result.mapping->base_address()};
        const runtime::ModulePreparationView view{0,"synthetic",bases[0],writable.first(page),
            writable.subspan(page),page,metadata,bases};
        prep.prepare(view);
    }
    return result;
}

void refused_forms() {
    // Independent ISA encodings for unobserved forms; none may become a native passthrough.
    const std::uint32_t refused[]{
        0xd65f0bff,0xd69f0bff,0xd69f0fff, // RETAA, ERETAA, ERETAB
        0xd71f0801,0xd71f0c01,0xd73f0801,0xd73f0c01, // BRAA/BRAB, BLRAA/BLRAB
        0xd61f081f,0xd61f0c1f,0xd63f081f,0xd63f0c1f, // zero-modifier branches
        0xf8200420,0xf8a00420, // LDRAA, LDRAB
        0xdac10820,0xdac10c20,0xdac11020,0xdac11820,0xdac11c20, // unobserved register forms
        0xdac123e0,0xdac127e0,0xdac12be0,0xdac12fe0, // PACIZA/PACIZB/PACDZA/PACDZB
        0xdac133e0,0xdac137e0,0xdac13be0,0xdac13fe0, // AUTIZA/AUTIZB/AUTDZA/AUTDZB
        0xdac147e0, // XPACD
        0xd503211f,0xd503215f,0xd503231f,0xd503233f, // PACIA1716/PACIB1716/PACIAZ/PACIASP
        0xd503235f,0xd503239f,0xd50323df // PACIBZ, AUTIAZ, AUTIBZ
    };
    platform::ThreadSlotKey key;
    for(auto word:refused) {
        const std::uint32_t words[]{word,0xd65f03c0};
        auto image=prepare(words,key,true);
        CHECK(image.preparation.stops.size()==1);
        if (!image.preparation.stops.empty())
            CHECK(image.preparation.stops[0].reason.find("authentication")!=std::string::npos);
    }
}

execution::GateStatus supervisor(std::uint16_t svc,execution::GateCall& call,
    execution::GuestThreadContext&,void* p) {
    CHECK(svc==42); ++*static_cast<unsigned*>(p); call.set_x(0,7); return execution::GateStatus::resume;
}

std::optional<std::pair<std::uint64_t,std::uint64_t>> execute(std::span<const std::uint32_t> words,
    bool rewrite, platform::ThreadSlotKey& key, platform::GuardedStack& stack,
    std::uint64_t x0,std::uint64_t x1,bool gate=false,bool virtualized=false) {
    auto image=prepare(words,key,rewrite);
    CHECK(image.preparation.stops.empty());
    if (!image.mapping || !image.preparation.stops.empty())return {};
    if (rewrite && !gate && !virtualized) {
        const auto original=image.mapping->preparation_bytes().first(words.size_bytes());
        CHECK(std::memcmp(original.data(),words.data(),words.size_bytes())==0);
    }
    CHECK(image.mapping->seal().ok());
    auto table=execution::GuestSiteTable::build(image.preparation.sites); CHECK(table.ok());
    execution::GuestThreadContext thread; thread.set_read_only_thread_pointer(0x123450);
    unsigned calls=0;
    execution::GuestDispatchScope dispatch(table.table,thread,{supervisor,&calls});
    platform::ThreadSlotBinding binding(key,thread.bound_register_cell());
    CHECK(binding.native_access());
    const auto platform_before=host_x18();
    const auto result=execution::invoke_guest_entry(stack,{.function=image.mapping->base_address(),.x0=x0,.x1=x1});
    CHECK(result.returned() && dispatch.failure_count()==0 && host_x18()==platform_before);
    if(virtualized)CHECK(thread.guest_x18()==result.x0);
    if(gate)CHECK(calls==1 && result.x0==0x123450 && thread.guest_x18()==0x123450);
    return std::pair{result.x0,result.x1};
}
void native_cases() {
    platform::ThreadSlotKey key;
    auto stack=platform::GuardedStack::create({.usable_size=1u<<20}); CHECK(stack.ok()); if(!stack.ok())return;
    const std::vector<std::vector<std::uint32_t>> cases{
        {0xd503237f,0xaa1e03e1,0xd28009a0,0xd65f0fff}, // signed LR snapshot and return
        {0xdac103fe,0xd50323bf,0xd65f03c0}, // PACIA LR,SP; AUTIASP; RET
        {0xdac107fe,0xd50323ff,0xd65f03c0}, // PACIB LR,SP; AUTIBSP; RET
        {0xd503237f,0xdac10420,0xdac11420,0xd65f0fff},
        {0xd503237f,0x9ac13000,0xd65f0fff}, // PACGA x0,x0,x1
        {0xd503237f,0xdac143e0,0xd65f0fff},
        {0xd503237f,0xaa0103f0,0xaa0003f1,0xdac10211,0xd503219f,0xaa1103e0,0xd65f0fff},
        {0xd503237f,0xaa0103f0,0xaa0003f1,0xdac10611,0xd50321df,0xaa1103e0,0xd65f0fff},
        {0xb24b03de,0xd50320ff,0xd65f03c0} // ORR LR,LR,#(1<<53); XPACLRI; RET
    };
    for(const auto& words:cases) {
        // Refuse before any native execution while the production classifier is incomplete.
        auto probe=prepare(words,key,true);
        CHECK(probe.preparation.stops.empty()); if(!probe.preparation.stops.empty())continue;
        const auto raw=execute(words,false,key,*stack.stack,0x12340,0x55);
        const auto prepared=execute(words,true,key,*stack.stack,0x12340,0x55);
        CHECK(raw && prepared && raw==prepared);
    }
    const std::uint32_t gated[]{0xd503237f,0xa9bf7bfd,0x910003fd,0xd53bd060,
        0xaa0003f2,0xd4000541,0xaa1203e0,0xa8c17bfd,0xd65f0fff};
    (void)execute(gated,true,key,*stack.stack,0,0,true);
    const std::uint32_t virtual_pair[]{0xd503237f,0xaa0003f2,0xdac10432,0xdac11432,0xaa1203e0,0xd65f0fff};
    const auto paired=execute(virtual_pair,true,key,*stack.stack,0x12340,0x55,false,true);
    CHECK(paired && paired->first==0x12340);
    const std::uint32_t generic[]{0xd503237f,0x9ac13000,0xd65f0fff};
    const std::uint32_t virtual_generic[]{0xd503237f,0xaa0003f2,0x9ac13252,0xaa1203e0,0xd65f0fff};
    const auto plain=execute(generic,false,key,*stack.stack,0x12340,0x55);
    const auto virtual_result=execute(virtual_generic,true,key,*stack.stack,0x12340,0x55,false,true);
    CHECK(plain && virtual_result && plain==virtual_result);
    const std::uint32_t sp_modifier[]{0xdac107f2,0xd65f03c0};
    auto refused=prepare(sp_modifier,key,true);
    CHECK(refused.preparation.stops.size()==1);
}
}
int main(){effects();return_flow();refused_forms();native_cases();return failures?1:0;}
