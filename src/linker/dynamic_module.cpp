// ============================================================================
// Armigo Linker - Guest Dynamic Linking
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/linker/dynamic_module.h>

#include <algorithm>
#include <bit>
#include <limits>
#include <map>
#include <new>
#include <utility>

namespace armigo::linker {
namespace {

struct ParseFailure {
    DynamicError error;
    const char* message;
};

[[noreturn]] void reject(DynamicError error, const char* message) {
    throw ParseFailure{error, message};
}

class Reader {
public:
    Reader(std::span<const std::byte> image, const DynamicLimits& limits)
        : image_(image), limits_(limits) {}

    DynamicModule read() {
        require_range(0, 8);
        const std::uint64_t header = word(4);
        require_range(header, 8);
        if (word(header) != 0x30444f4du) {
            reject(DynamicError::invalid_header, "The module has no valid MOD0 header.");
        }
        const auto displacement = std::bit_cast<std::int32_t>(word(header + 4));
        const auto dynamic_signed = static_cast<std::int64_t>(header) + displacement;
        if (dynamic_signed < 0 || dynamic_signed % 8 != 0) {
            reject(DynamicError::invalid_header, "The dynamic table address is invalid.");
        }
        std::uint64_t cursor = static_cast<std::uint64_t>(dynamic_signed);
        bool terminated = false;
        for (std::size_t index = 0; index < limits_.max_entries; ++index) {
            require_range(cursor, 16);
            const auto tag = wide(cursor);
            const auto value = wide(cursor + 8);
            cursor += 16;
            if (tag == 0) {
                terminated = true;
                break;
            }
            if (tag == 1) {
                needed_.push_back(value);
            } else if (!tags_.emplace(tag, value).second) {
                reject(DynamicError::invalid_table, "A singleton dynamic tag is duplicated.");
            }
        }
        if (!terminated) {
            reject(DynamicError::resource_limit, "The dynamic table exceeds its entry limit.");
        }
        if (required(11) != 24) {
            reject(DynamicError::invalid_table, "The dynamic symbol entry size is invalid.");
        }
        strings_ = required(5);
        strings_size_ = required(10);
        require_range(strings_, strings_size_);
        if (strings_size_ == 0 || byte(strings_) != 0) {
            reject(DynamicError::invalid_table, "The dynamic string table is invalid.");
        }
        const auto symbol_table = required(6);
        const auto count = symbol_count();
        if (count == 0) {
            reject(DynamicError::invalid_symbol, "The mandatory null symbol is missing.");
        }
        require_range(symbol_table, count * 24);
        result_.symbols.reserve(static_cast<std::size_t>(count));
        for (std::uint64_t index = 0; index < count; ++index) {
            const auto at = symbol_table + index * 24;
            Symbol symbol;
            symbol.name = string(word(at));
            symbol.info = byte(at + 4);
            symbol.visibility = byte(at + 5);
            symbol.section = half(at + 6);
            symbol.value = wide(at + 8);
            symbol.size = wide(at + 16);
            if (index == 0 && (word(at) != 0 || symbol.info != 0 ||
                               symbol.visibility != 0 || symbol.section != 0 ||
                               symbol.value != 0 || symbol.size != 0)) {
                reject(DynamicError::invalid_symbol, "Symbol zero is not the undefined null entry.");
            }
            // Absolute symbols are values, not offsets into this module.
            if (symbol.defined() && symbol.section != 0xfff1) {
                require_range(symbol.value, symbol.size);
            }
            result_.symbols.push_back(std::move(symbol));
        }
        for (const auto offset : needed_) {
            result_.dependencies.push_back(string(offset));
        }
        read_rela(7, 8, false);
        read_rela(23, 2, true);
        read_relr();
        result_.init = optional(12);
        result_.fini = optional(13);
        for (const auto entry : {result_.init, result_.fini}) {
            if (entry != 0) {
                if (entry % 4 != 0) {
                    reject(DynamicError::invalid_table, "An initialization function is misaligned.");
                }
                require_range(entry, 4);
            }
        }
        read_array(25, 27, result_.init_array, result_.init_array_size);
        read_array(26, 28, result_.fini_array, result_.fini_array_size);
        if (tags_.contains(17) || tags_.contains(18)) {
            reject(DynamicError::unsupported_relocation, "REL relocation tables are not implemented.");
        }
        return std::move(result_);
    }

private:
    void require_range(std::uint64_t offset, std::uint64_t size) const {
        if (offset > image_.size() || size > image_.size() - offset) {
            reject(DynamicError::invalid_table, "A module table or range exceeds the image.");
        }
    }

    std::uint8_t byte(std::uint64_t offset) const {
        require_range(offset, 1);
        return std::to_integer<std::uint8_t>(image_[static_cast<std::size_t>(offset)]);
    }

    std::uint16_t half(std::uint64_t offset) const {
        return static_cast<std::uint16_t>(byte(offset) | (byte(offset + 1) << 8));
    }

    std::uint32_t word(std::uint64_t offset) const {
        require_range(offset, 4);
        std::uint32_t value = 0;
        for (unsigned index = 0; index < 4; ++index) {
            value |= static_cast<std::uint32_t>(byte(offset + index)) << (index * 8);
        }
        return value;
    }

    std::uint64_t wide(std::uint64_t offset) const {
        require_range(offset, 8);
        return word(offset) | (static_cast<std::uint64_t>(word(offset + 4)) << 32);
    }

    std::uint64_t required(std::uint64_t tag) const {
        const auto found = tags_.find(tag);
        if (found == tags_.end()) {
            reject(DynamicError::invalid_table, "A required dynamic tag is missing.");
        }
        return found->second;
    }

    std::uint64_t optional(std::uint64_t tag) const {
        const auto found = tags_.find(tag);
        return found == tags_.end() ? 0 : found->second;
    }

    std::string string(std::uint64_t offset) {
        if (offset >= strings_size_) {
            reject(DynamicError::invalid_symbol, "A dynamic string offset is out of range.");
        }
        const auto available = strings_size_ - offset;
        const auto bound = std::min<std::uint64_t>(available, limits_.max_name_bytes);
        const auto start = strings_ + offset;
        for (std::uint64_t size = 0; size < bound; ++size) {
            if (byte(start + size) == 0) {
                if (size > limits_.max_total_name_bytes - name_bytes_) {
                    reject(DynamicError::resource_limit, "Dynamic names exceed their aggregate byte limit.");
                }
                name_bytes_ += static_cast<std::size_t>(size);
                return {reinterpret_cast<const char*>(image_.data() + start),
                        static_cast<std::size_t>(size)};
            }
        }
        reject(DynamicError::invalid_symbol, "A dynamic string is unterminated or exceeds its limit.");
    }

    std::uint64_t symbol_count() const {
        if (tags_.contains(4)) {
            const auto at = required(4);
            require_range(at, 8);
            const auto buckets = word(at);
            const auto chains = word(at + 4);
            if (chains == 0 || chains > limits_.max_symbols ||
                buckets > limits_.max_symbols) {
                reject(DynamicError::resource_limit, "The symbol hash table exceeds its limit.");
            }
            const auto entries = static_cast<std::uint64_t>(buckets) + chains;
            require_range(at + 8, entries * 4);
            for (std::uint64_t index = 0; index < entries; ++index) {
                if (word(at + 8 + index * 4) >= chains) {
                    reject(DynamicError::invalid_table, "A symbol hash index is out of range.");
                }
            }
            return chains;
        }
        constexpr std::uint64_t gnu_hash = 0x6ffffef5;
        const auto at = required(gnu_hash);
        require_range(at, 16);
        const auto buckets = word(at);
        const auto first_symbol = word(at + 4);
        const auto bloom_words = word(at + 8);
        if (buckets == 0 || bloom_words == 0 ||
            buckets > limits_.max_symbols || bloom_words > limits_.max_symbols ||
            first_symbol > limits_.max_symbols) {
            reject(DynamicError::resource_limit, "The GNU symbol hash table exceeds its limit.");
        }
        const auto bucket_at = at + 16 + static_cast<std::uint64_t>(bloom_words) * 8;
        require_range(at + 16, static_cast<std::uint64_t>(bloom_words) * 8 + buckets * 4ull);
        std::uint32_t highest = 0;
        for (std::uint64_t index = 0; index < buckets; ++index) {
            const auto value = word(bucket_at + index * 4);
            if (value != 0 && value < first_symbol) {
                reject(DynamicError::invalid_table, "A GNU symbol bucket precedes its symbol range.");
            }
            highest = std::max(highest, value);
        }
        if (highest == 0) {
            return first_symbol;
        }
        const auto chain_at = bucket_at + buckets * 4ull;
        for (std::uint64_t index = highest; index < limits_.max_symbols; ++index) {
            if ((word(chain_at + (index - first_symbol) * 4) & 1u) != 0) {
                return index + 1;
            }
        }
        reject(DynamicError::resource_limit, "The GNU symbol hash chain exceeds its limit.");
    }

    void check_target(std::uint64_t target) const {
        if (target % 8 != 0) {
            reject(DynamicError::invalid_table, "A relocation target is not eight-byte aligned.");
        }
        require_range(target, 8);
    }

    void check_relocation_budget() const {
        if (result_.relocations.size() + result_.packed_relative_targets.size() >=
            limits_.max_relocations) {
            reject(DynamicError::resource_limit, "The relocation count exceeds its limit.");
        }
    }

    void reserve_record_budget(std::uint64_t count) {
        if (count > limits_.max_relocation_records - relocation_records_) {
            reject(DynamicError::resource_limit, "Encoded relocation records exceed their limit.");
        }
        relocation_records_ += static_cast<std::size_t>(count);
    }

    void read_rela(std::uint64_t address_tag, std::uint64_t size_tag, bool plt) {
        if (!tags_.contains(address_tag) && !tags_.contains(size_tag)) {
            return;
        }
        const auto address = required(address_tag);
        const auto size = required(size_tag);
        if (size % 24 != 0 || (plt ? required(20) != 7 : required(9) != 24)) {
            reject(DynamicError::invalid_table, "A RELA table has an invalid entry format.");
        }
        require_range(address, size);
        reserve_record_budget(size / 24);
        for (std::uint64_t offset = 0; offset < size; offset += 24) {
            const auto target = wide(address + offset);
            const auto info = wide(address + offset + 8);
            const auto kind = static_cast<std::uint32_t>(info);
            if (kind == 0 || kind == 256) {
                continue;
            }
            if (kind != 257 && kind != 1025 && kind != 1026 && kind != 1027) {
                reject(DynamicError::unsupported_relocation, "An ARM64 relocation kind is not implemented.");
            }
            const auto symbol = static_cast<std::uint32_t>(info >> 32);
            if (symbol >= result_.symbols.size() || (kind == 1027 && symbol != 0)) {
                reject(DynamicError::invalid_symbol, "A relocation references an invalid symbol.");
            }
            check_target(target);
            check_relocation_budget();
            result_.relocations.push_back(
                {target, static_cast<RelocationKind>(kind), symbol,
                 std::bit_cast<std::int64_t>(wide(address + offset + 16))});
        }
    }

    void read_relr() {
        if (!tags_.contains(36) && !tags_.contains(35) && !tags_.contains(37)) {
            return;
        }
        const auto address = required(36);
        const auto size = required(35);
        if (required(37) != 8 || size % 8 != 0) {
            reject(DynamicError::invalid_table, "A RELR table has an invalid entry format.");
        }
        require_range(address, size);
        reserve_record_budget(size / 8);
        std::optional<std::uint64_t> cursor;
        for (std::uint64_t offset = 0; offset < size; offset += 8) {
            const auto entry = wide(address + offset);
            if ((entry & 1u) == 0) {
                check_target(entry);
                check_relocation_budget();
                result_.packed_relative_targets.push_back(entry);
                cursor = entry + 8;
            } else {
                if (!cursor || *cursor > std::numeric_limits<std::uint64_t>::max() - 63 * 8) {
                    reject(DynamicError::invalid_table, "A RELR bitmap has no valid base.");
                }
                for (unsigned bit = 0; bit < 63; ++bit) {
                    if ((entry & (std::uint64_t{1} << (bit + 1))) != 0) {
                        const auto target = *cursor + bit * 8;
                        check_target(target);
                        check_relocation_budget();
                        result_.packed_relative_targets.push_back(target);
                    }
                }
                *cursor += 63 * 8;
            }
        }
    }

    void read_array(std::uint64_t address_tag, std::uint64_t size_tag,
                    std::uint64_t& address, std::uint64_t& size) const {
        if (!tags_.contains(address_tag) && !tags_.contains(size_tag)) {
            return;
        }
        address = required(address_tag);
        size = required(size_tag);
        if (address % 8 != 0 || size % 8 != 0) {
            reject(DynamicError::invalid_table, "An initialization array is misaligned.");
        }
        require_range(address, size);
    }

    std::span<const std::byte> image_;
    const DynamicLimits& limits_;
    std::map<std::uint64_t, std::uint64_t> tags_;
    std::vector<std::uint64_t> needed_;
    std::uint64_t strings_ = 0;
    std::uint64_t strings_size_ = 0;
    std::size_t name_bytes_ = 0;
    std::size_t relocation_records_ = 0;
    DynamicModule result_;
};

} // namespace

DynamicResult read_dynamic_module(std::span<const std::byte> image,
                                  const DynamicLimits& limits) {
    try {
        return {DynamicError::none, {}, Reader(image, limits).read()};
    } catch (const ParseFailure& failure) {
        return {failure.error, failure.message, std::nullopt};
    } catch (const std::bad_alloc&) {
        return {DynamicError::resource_limit, "Module metadata allocation failed.", std::nullopt};
    }
}

} // namespace armigo::linker
