// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_native/input.h>
#include <armigo/shader_native/program.h>
#include <armigo/shader_native/opcode_inventory.h>
#include <armigo/loader/nup.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <tuple>

namespace sn = armigo::shader_native;
namespace ld = armigo::loader;
namespace {
std::string hex(sn::Bytes bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string s;
    for (auto b : bytes) { const auto v = std::to_integer<unsigned>(b); s += digits[v >> 4]; s += digits[v & 15]; }
    return s;
}
std::string label(const std::string& s) { return hex(std::as_bytes(std::span(s))); }
template<class T> T unwrap(std::variant<ld::NupFailure, T> result) {
    if (auto e = std::get_if<ld::NupFailure>(&result)) throw sn::InputError(e->operation + ": " + e->detail);
    return std::get<T>(std::move(result));
}
struct Source {
    std::shared_ptr<ld::NupArchive> archive;
    sn::Buffer read(std::uint32_t index, const sn::Dictionaries& dictionaries, std::size_t limit) {
        const auto& file = archive->files().at(index);
        if (file.codec || file.stored_size != file.size) throw sn::InputError("Unsupported NUP extent codec");
        if (auto e = archive->verify(index)) throw sn::InputError(e->operation + ": " + e->detail);
        std::uint64_t position = 0;
        auto reader = [&](std::span<std::byte> buffer) {
            const auto count = unwrap(archive->read(index, position, buffer));
            position += count; return count;
        };
        std::array<std::byte, 4> prefix{};
        const auto count = unwrap(archive->read(index, 0, prefix));
        if (count == 4 && sn::little_endian(prefix, 0, 4) == 0xfd2fb528)
            return sn::read_zstandard(reader, dictionaries, limit);
        return sn::read_raw_member(reader, file.size, limit);
    }
};
}
int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "Usage: armigo-native-shader-inventory NUP DICTIONARY_MEMBER NEW_OUTPUT_DIRECTORY\n"; return 2; }
    try {
        const auto output = std::filesystem::weakly_canonical(argv[3]);
        for (auto parent = output; !parent.empty(); parent = parent.parent_path()) {
            if (std::filesystem::exists(parent / ".git")) throw sn::InputError("Inventory output must be outside Git work trees");
            if (parent == parent.root_path()) break;
        }
        if (!std::filesystem::create_directory(output)) throw sn::InputError("Output directory must be new");
        std::ofstream records(output / "records.tsv"), members(output / "members.tsv"), identity(output / "selection.tsv");
        records.exceptions(std::ios::failbit | std::ios::badbit);
        members.exceptions(std::ios::failbit | std::ios::badbit);
        identity.exceptions(std::ios::failbit | std::ios::badbit);
        Source source{unwrap(ld::NupArchive::open(argv[1]))};
        const auto selection = unwrap(source.archive->select());
        std::ifstream input(argv[1], std::ios::binary);
        std::array<std::byte, 128> header{};
        if (!input.read(reinterpret_cast<char*>(header.data()), header.size())) throw sn::InputError("Cannot reread NUP identity");
        identity << "schema\tnative-input-v1\nheader_sha256\t" << sn::digest(header)
                 << "\ntitle_id\t" << std::hex << selection.title_id << std::dec << "\nversion\t" << selection.version
                 << "\nupdate_applied\t" << selection.update_applied << "\n";
        for (auto& [name, id] : selection.exefs) identity << "exefs\t" << label(name) << '\t' << id << '\t' << hex(source.archive->files()[id].hash) << '\n';
        auto dictionary_entry = selection.romfs.find(argv[2]);
        if (dictionary_entry == selection.romfs.end()) throw sn::InputError("Dictionary archive member was not selected");
        sn::Limits limits;
        auto dictionary_bytes = source.read(dictionary_entry->second, {}, limits.member_bytes);
        auto dictionaries = sn::read_dictionaries(dictionary_bytes, 16 << 20);
        identity << "dictionary_member\t" << label(argv[2]) << '\n';
        for (const auto& [id, bytes] : dictionaries)
            identity << "dictionary\t" << id << '\t' << sn::digest(bytes) << '\n';
        std::uint32_t current = 0;
        std::string expanded_hash;
        std::set<std::pair<std::string, std::string>> pairs, refused_stage_pairs;
        std::set<std::string> program_images;
        sn::OpcodeInventory opcode_inventory;
        std::size_t fully_decoded_images = 0;
        records << "file\texpanded_sha256\tformat\tancestry_hex_offsets\tcontainer\tdescriptor\tcontrol\tcode\tvariation\tslot\tcontrol_size\tcode_size\tcontrol_sha256\tcode_sha256\tstage_selector\tstage_status\timage_sha256\tstage_identity\tallocated_slots\tdecoded_slots\trefused_slots\tfirst_refused_opcode\tfirst_refused_pc\tfirst_refusal\n";
        sn::Enumerator enumerator([&](const sn::Record& r) {
            auto control_hash = sn::digest(r.control), code_hash = sn::digest(r.code);
            pairs.emplace(control_hash, code_hash);
            auto parsed = sn::parse_program(r);
            std::string stage_status = "metadata-accepted", image_hash, stage_identity;
            std::optional<sn::ProgramAudit> audit;
            if (auto failure = std::get_if<sn::ProgramFailure>(&parsed)) {
                if (failure->code != sn::ProgramError::unsupported_stage) throw sn::InputError(failure->message, r.descriptor_offset);
                refused_stage_pairs.emplace(control_hash, code_hash);
                stage_status = failure->message;
            } else {
                const auto& program = std::get<sn::Program>(parsed);
                image_hash = program.image_digest; stage_identity = program.stage_identity;
                program_images.insert(image_hash);
                const auto before = opcode_inventory.programs();
                audit = opcode_inventory.inspect(program);
                if (opcode_inventory.programs() != before && !audit->refused) ++fully_decoded_images;
            }
            records << current << '\t' << expanded_hash << '\t' << (r.kind == sn::ContainerKind::bnsh ? "BNSH" : "BAHS") << '\t';
            for (auto& location : r.ancestry) records << label(location.member) << ':' << location.offset << '/';
            records << '\t' << r.container_offset << '\t' << r.descriptor_offset << '\t' << r.control_offset << '\t' << r.code_offset << '\t';
            if (r.variation) records << *r.variation;
            records << '\t'; if (r.slot) records << *r.slot;
            records << '\t' << r.control.size() << '\t' << r.code.size() << '\t' << control_hash << '\t' << code_hash << '\t' << sn::little_endian(r.control, 0x56, 1)
                    << '\t' << stage_status << '\t' << image_hash << '\t' << stage_identity;
            if (audit) {
                records << '\t' << audit->slots << '\t' << audit->decoded << '\t' << audit->refused;
                if (audit->first_failure) {
                    const auto& failure = *audit->first_failure;
                    records << '\t' << std::hex << failure.opcode << std::dec << '\t' << failure.pc << '\t' << failure.message;
                } else records << "\t\t\t";
            } else records << "\t\t\t\t\t\t";
            records << '\n';
        }, limits, dictionaries);
        using Extent = std::tuple<std::uint64_t, std::uint64_t, std::array<std::byte, 32>>;
        std::map<Extent, std::pair<std::uint32_t, std::string>> extents;
        members << "path_hex\tfile\toffset\tstored_size\tsha256\n";
        for (auto& [name, id] : selection.romfs) {
            auto& f = source.archive->files()[id];
            members << label(name) << '\t' << id << '\t' << f.offset << '\t' << f.stored_size << '\t' << hex(f.hash) << '\n';
            extents.try_emplace(Extent{f.offset, f.stored_size, f.hash}, id, name);
        }
        std::size_t scanned = 0;
        for (auto& [extent, entry] : extents) {
            current = entry.first;
            auto bytes = source.read(current, dictionaries, limits.member_bytes);
            expanded_hash = sn::digest(bytes);
            if (entry.second != argv[2]) enumerator.scan(bytes, entry.second);
            if (++scanned % 5000 == 0) std::cerr << "Scanned " << scanned << " extents; records=" << enumerator.inventory().records << '\n';
        }
        std::ofstream opcodes(output / "opcodes.tsv");
        opcodes.exceptions(std::ios::failbit | std::ios::badbit);
        opcodes << "opcode_hex\tallocated_slots\tdecoded_slots\trefused_slots\n";
        std::uint64_t decoded_slots = 0, refused_slots = 0;
        for (std::size_t opcode = 0; opcode < opcode_inventory.counts().size(); ++opcode) {
            const auto& count = opcode_inventory.counts()[opcode];
            decoded_slots += count.decoded; refused_slots += count.refused;
            if (count.slots) opcodes << std::hex << opcode << std::dec << '\t' << count.slots << '\t' << count.decoded << '\t' << count.refused << '\n';
        }
        records.flush(); members.flush(); identity.flush(); opcodes.flush();
        const auto& i = enumerator.inventory();
        std::ofstream summary(output / "summary.tsv");
        summary.exceptions(std::ios::failbit | std::ios::badbit);
        summary << "scope\tstatic-container-census\nrunnable\tfalse\npaths\t" << selection.romfs.size()
                << "\nextents\t" << extents.size() << "\nbnsh\t" << i.bnsh << "\nbahs\t" << i.bahs
                << "\nrecords\t" << i.records << "\ncontrol_code_pairs\t" << pairs.size()
                << "\naccepted_stage_images\t" << program_images.size() << "\nrefused_stage_pairs\t" << refused_stage_pairs.size()
                << "\nduplicate_candidates\t" << i.duplicate_candidates << "\nrejected_markers\t" << i.rejected_markers
                << "\nunrecognized_spans\t" << i.unrecognized_spans << "\nvisited_bytes\t" << i.visited_bytes << '\n';
        summary << "decoded_slots\t" << decoded_slots << "\nrefused_slots\t" << refused_slots
                << "\nfully_decoded_images\t" << fully_decoded_images << '\n';
        constexpr const char* failure_names[]{"unknown_opcode", "unvalidated_opcode", "unvalidated_mode", "invalid_pc", "branch_target"};
        for (unsigned i = 0; i < opcode_inventory.failures().size(); ++i)
            summary << failure_names[i] << '\t' << opcode_inventory.failures()[i] << '\n';
        summary.flush();
        std::cout << "Enumerated " << i.records << " records and " << pairs.size() << " complete control/code pairs\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "Native shader inventory failed: " << e.what() << '\n'; return 1; }
}
