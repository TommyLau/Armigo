// ============================================================================
// Armigo Shader - Native Shader Translation
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================
// Armigo uses owned C++ values and checked contracts rather than managed nodes.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace armigo::shader::pipeline {
// Vocabulary only: presence in this enum never implies implemented support.
enum class InstructionName : std::uint16_t {
    invalid,
    al2p,
    ald,
    ast,
    atom,
    atom_cas,
    atoms,
    atoms_cas,
    b2r,
    bar,
    bfe,
    bfi,
    bpt,
    bra,
    brk,
    brx,
    cal,
    cctl,
    cctll,
    cctlt,
    cont,
    cset,
    csetp,
    cs2r,
    dadd,
    depbar,
    dfma,
    dmnmx,
    dmul,
    dset,
    dsetp,
    exit,
    f2f,
    f2i,
    fadd,
    fadd32i,
    fchk,
    fcmp,
    ffma,
    ffma32i,
    flo,
    fmnmx,
    fmul,
    fmul32i,
    fset,
    fsetp,
    fswzadd,
    getcrsptr,
    getlmembase,
    hadd2,
    hadd232i,
    hfma2,
    hmul2,
    hmul232i,
    hset2,
    hsetp2,
    i2f,
    i2i,
    iadd,
    iadd32i,
    iadd3,
    icmp,
    ide,
    idp,
    imad,
    imad32i,
    imadsp,
    imnmx,
    imul,
    imul32i,
    ipa,
    isberd,
    iscadd,
    iscadd32i,
    iset,
    isetp,
    jcal,
    jmp,
    jmx,
    kil,
    ld,
    ldc,
    ldg,
    ldl,
    lds,
    lea,
    lea_hi,
    lepc,
    longjmp,
    lop,
    lop3,
    lop32i,
    membar,
    mov,
    mov32i,
    mufu,
    nop,
    out,
    p2r,
    pbk,
    pcnt,
    pexit,
    pixld,
    plongjmp,
    popc,
    pret,
    prmt,
    pset,
    psetp,
    r2b,
    r2p,
    ram,
    red,
    ret,
    rro,
    rtt,
    s2r,
    sam,
    sel,
    setcrsptr,
    setlmembase,
    shf,
    shf_2,
    shf_3,
    shf_4,
    shfl,
    shl,
    shr,
    ssy,
    st,
    stg,
    stl,
    stp,
    sts,
    suatom_b,
    suatom,
    suatom_b2,
    suatom_cas_b,
    suatom_cas,
    suld_db,
    suld_d,
    suld_b,
    suld,
    sured_b,
    sured,
    sust_db,
    sust_d,
    sust_b,
    sust,
    sync,
    tex,
    tex_b,
    texs,
    texs_f16,
    tld,
    tld_b,
    tlds,
    tlds_f16,
    tld4,
    tld4_b,
    tld4s,
    tld4s_f16,
    tmml,
    tmml_b,
    txa,
    txd,
    txd_b,
    txq,
    txq_b,
    vabsdiff,
    vabsdiff4,
    vadd,
    vmad,
    vmnmx,
    vote,
    votevtg,
    vset,
    vsetp,
    vshl,
    vshr,
    xmad,
    hfma2_32i,
    count
};
inline constexpr std::size_t instruction_name_count =
    static_cast<std::size_t>(InstructionName::count);
[[nodiscard]] constexpr std::string_view instruction_name(InstructionName name) {
    switch (name) {
    case InstructionName::invalid: return "INVALID";
    case InstructionName::al2p: return "AL2P";
    case InstructionName::ald: return "ALD";
    case InstructionName::ast: return "AST";
    case InstructionName::atom: return "ATOM";
    case InstructionName::atom_cas: return "ATOMCAS";
    case InstructionName::atoms: return "ATOMS";
    case InstructionName::atoms_cas: return "ATOMSCAS";
    case InstructionName::b2r: return "B2R";
    case InstructionName::bar: return "BAR";
    case InstructionName::bfe: return "BFE";
    case InstructionName::bfi: return "BFI";
    case InstructionName::bpt: return "BPT";
    case InstructionName::bra: return "BRA";
    case InstructionName::brk: return "BRK";
    case InstructionName::brx: return "BRX";
    case InstructionName::cal: return "CAL";
    case InstructionName::cctl: return "CCTL";
    case InstructionName::cctll: return "CCTLL";
    case InstructionName::cctlt: return "CCTLT";
    case InstructionName::cont: return "CONT";
    case InstructionName::cset: return "CSET";
    case InstructionName::csetp: return "CSETP";
    case InstructionName::cs2r: return "CS2R";
    case InstructionName::dadd: return "DADD";
    case InstructionName::depbar: return "DEPBAR";
    case InstructionName::dfma: return "DFMA";
    case InstructionName::dmnmx: return "DMNMX";
    case InstructionName::dmul: return "DMUL";
    case InstructionName::dset: return "DSET";
    case InstructionName::dsetp: return "DSETP";
    case InstructionName::exit: return "EXIT";
    case InstructionName::f2f: return "F2F";
    case InstructionName::f2i: return "F2I";
    case InstructionName::fadd: return "FADD";
    case InstructionName::fadd32i: return "FADD32I";
    case InstructionName::fchk: return "FCHK";
    case InstructionName::fcmp: return "FCMP";
    case InstructionName::ffma: return "FFMA";
    case InstructionName::ffma32i: return "FFMA32I";
    case InstructionName::flo: return "FLO";
    case InstructionName::fmnmx: return "FMNMX";
    case InstructionName::fmul: return "FMUL";
    case InstructionName::fmul32i: return "FMUL32I";
    case InstructionName::fset: return "FSET";
    case InstructionName::fsetp: return "FSETP";
    case InstructionName::fswzadd: return "FSWZADD";
    case InstructionName::getcrsptr: return "GETCRSPTR";
    case InstructionName::getlmembase: return "GETLMEMBASE";
    case InstructionName::hadd2: return "HADD2";
    case InstructionName::hadd232i: return "HADD232I";
    case InstructionName::hfma2: return "HFMA2";
    case InstructionName::hmul2: return "HMUL2";
    case InstructionName::hmul232i: return "HMUL232I";
    case InstructionName::hset2: return "HSET2";
    case InstructionName::hsetp2: return "HSETP2";
    case InstructionName::i2f: return "I2F";
    case InstructionName::i2i: return "I2I";
    case InstructionName::iadd: return "IADD";
    case InstructionName::iadd32i: return "IADD32I";
    case InstructionName::iadd3: return "IADD3";
    case InstructionName::icmp: return "ICMP";
    case InstructionName::ide: return "IDE";
    case InstructionName::idp: return "IDP";
    case InstructionName::imad: return "IMAD";
    case InstructionName::imad32i: return "IMAD32I";
    case InstructionName::imadsp: return "IMADSP";
    case InstructionName::imnmx: return "IMNMX";
    case InstructionName::imul: return "IMUL";
    case InstructionName::imul32i: return "IMUL32I";
    case InstructionName::ipa: return "IPA";
    case InstructionName::isberd: return "ISBERD";
    case InstructionName::iscadd: return "ISCADD";
    case InstructionName::iscadd32i: return "ISCADD32I";
    case InstructionName::iset: return "ISET";
    case InstructionName::isetp: return "ISETP";
    case InstructionName::jcal: return "JCAL";
    case InstructionName::jmp: return "JMP";
    case InstructionName::jmx: return "JMX";
    case InstructionName::kil: return "KIL";
    case InstructionName::ld: return "LD";
    case InstructionName::ldc: return "LDC";
    case InstructionName::ldg: return "LDG";
    case InstructionName::ldl: return "LDL";
    case InstructionName::lds: return "LDS";
    case InstructionName::lea: return "LEA";
    case InstructionName::lea_hi: return "LEAHI";
    case InstructionName::lepc: return "LEPC";
    case InstructionName::longjmp: return "LONGJMP";
    case InstructionName::lop: return "LOP";
    case InstructionName::lop3: return "LOP3";
    case InstructionName::lop32i: return "LOP32I";
    case InstructionName::membar: return "MEMBAR";
    case InstructionName::mov: return "MOV";
    case InstructionName::mov32i: return "MOV32I";
    case InstructionName::mufu: return "MUFU";
    case InstructionName::nop: return "NOP";
    case InstructionName::out: return "OUT";
    case InstructionName::p2r: return "P2R";
    case InstructionName::pbk: return "PBK";
    case InstructionName::pcnt: return "PCNT";
    case InstructionName::pexit: return "PEXIT";
    case InstructionName::pixld: return "PIXLD";
    case InstructionName::plongjmp: return "PLONGJMP";
    case InstructionName::popc: return "POPC";
    case InstructionName::pret: return "PRET";
    case InstructionName::prmt: return "PRMT";
    case InstructionName::pset: return "PSET";
    case InstructionName::psetp: return "PSETP";
    case InstructionName::r2b: return "R2B";
    case InstructionName::r2p: return "R2P";
    case InstructionName::ram: return "RAM";
    case InstructionName::red: return "RED";
    case InstructionName::ret: return "RET";
    case InstructionName::rro: return "RRO";
    case InstructionName::rtt: return "RTT";
    case InstructionName::s2r: return "S2R";
    case InstructionName::sam: return "SAM";
    case InstructionName::sel: return "SEL";
    case InstructionName::setcrsptr: return "SETCRSPTR";
    case InstructionName::setlmembase: return "SETLMEMBASE";
    case InstructionName::shf: return "SHF";
    case InstructionName::shf_2: return "SHF_2";
    case InstructionName::shf_3: return "SHF_3";
    case InstructionName::shf_4: return "SHF_4";
    case InstructionName::shfl: return "SHFL";
    case InstructionName::shl: return "SHL";
    case InstructionName::shr: return "SHR";
    case InstructionName::ssy: return "SSY";
    case InstructionName::st: return "ST";
    case InstructionName::stg: return "STG";
    case InstructionName::stl: return "STL";
    case InstructionName::stp: return "STP";
    case InstructionName::sts: return "STS";
    case InstructionName::suatom_b: return "SUATOMB";
    case InstructionName::suatom: return "SUATOM";
    case InstructionName::suatom_b2: return "SUATOMB2";
    case InstructionName::suatom_cas_b: return "SUATOMCASB";
    case InstructionName::suatom_cas: return "SUATOMCAS";
    case InstructionName::suld_db: return "SULDDB";
    case InstructionName::suld_d: return "SULDD";
    case InstructionName::suld_b: return "SULDB";
    case InstructionName::suld: return "SULD";
    case InstructionName::sured_b: return "SUREDB";
    case InstructionName::sured: return "SURED";
    case InstructionName::sust_db: return "SUSTDB";
    case InstructionName::sust_d: return "SUSTD";
    case InstructionName::sust_b: return "SUSTB";
    case InstructionName::sust: return "SUST";
    case InstructionName::sync: return "SYNC";
    case InstructionName::tex: return "TEX";
    case InstructionName::tex_b: return "TEXB";
    case InstructionName::texs: return "TEXS";
    case InstructionName::texs_f16: return "TEXSF16";
    case InstructionName::tld: return "TLD";
    case InstructionName::tld_b: return "TLDB";
    case InstructionName::tlds: return "TLDS";
    case InstructionName::tlds_f16: return "TLDSF16";
    case InstructionName::tld4: return "TLD4";
    case InstructionName::tld4_b: return "TLD4B";
    case InstructionName::tld4s: return "TLD4S";
    case InstructionName::tld4s_f16: return "TLD4SF16";
    case InstructionName::tmml: return "TMML";
    case InstructionName::tmml_b: return "TMMLB";
    case InstructionName::txa: return "TXA";
    case InstructionName::txd: return "TXD";
    case InstructionName::txd_b: return "TXDB";
    case InstructionName::txq: return "TXQ";
    case InstructionName::txq_b: return "TXQB";
    case InstructionName::vabsdiff: return "VABSDIFF";
    case InstructionName::vabsdiff4: return "VABSDIFF4";
    case InstructionName::vadd: return "VADD";
    case InstructionName::vmad: return "VMAD";
    case InstructionName::vmnmx: return "VMNMX";
    case InstructionName::vote: return "VOTE";
    case InstructionName::votevtg: return "VOTEVTG";
    case InstructionName::vset: return "VSET";
    case InstructionName::vsetp: return "VSETP";
    case InstructionName::vshl: return "VSHL";
    case InstructionName::vshr: return "VSHR";
    case InstructionName::xmad: return "XMAD";
    case InstructionName::hfma2_32i: return "HFMA2_32I";
    case InstructionName::count: break;
    }
    return "INVALID";
}
static_assert(instruction_name(InstructionName::ssy) == "SSY");
static_assert(instruction_name_count < 65536);
} // namespace armigo::shader::pipeline
