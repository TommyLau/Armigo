// ============================================================================
// Armigo Shader Input - Shader Input Processing
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/shader_input/gfd_cache.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <optional>
#include <unordered_set>
#include <utility>

namespace armigo::shader_input {
namespace {

constexpr std::size_t kUtfRowLimit = 1'000'000;
constexpr std::size_t kUtfColumnLimit = 1024;
// Cells built for one table. The row and column caps are independent, so
// their product would materialize about a billion values from a table only
// a few kilobytes long. A real contents table stays well under this.
constexpr std::size_t kUtfValueLimit = 2'000'000;
// Bytes copied into names and string values, counting each row's copy of a
// constant. The source packet is already bounded; this bounds the copies.
constexpr std::size_t kUtfStringBudget = gfd_table_limit;
constexpr std::size_t kOptionLimit = 16;
constexpr std::size_t kProgramLimit = 100'000;
constexpr std::size_t kGscRecordSize = 36;
constexpr std::size_t kDescriptorMinimum = 0x91;
constexpr std::uint32_t kGscMagic = 0x47534330u;
constexpr std::uint32_t kGscExpectedCount = 8;
constexpr std::array<std::uint32_t, 2> kGscVersions{0x10000000u, 0x01087165u};

std::uint8_t byte_at(std::span<const std::byte> input, std::size_t offset) {
    return std::to_integer<std::uint8_t>(input[offset]);
}

std::uint16_t read_be16(std::span<const std::byte> input, std::size_t offset) {
    return static_cast<std::uint16_t>(byte_at(input, offset) << 8) |
           byte_at(input, offset + 1);
}

std::uint32_t read_be32(std::span<const std::byte> input, std::size_t offset) {
    return (static_cast<std::uint32_t>(byte_at(input, offset)) << 24) |
           (static_cast<std::uint32_t>(byte_at(input, offset + 1)) << 16) |
           (static_cast<std::uint32_t>(byte_at(input, offset + 2)) << 8) |
           byte_at(input, offset + 3);
}

std::uint32_t read_le32(std::span<const std::byte> input, std::size_t offset) {
    return static_cast<std::uint32_t>(byte_at(input, offset)) |
           (static_cast<std::uint32_t>(byte_at(input, offset + 1)) << 8) |
           (static_cast<std::uint32_t>(byte_at(input, offset + 2)) << 16) |
           (static_cast<std::uint32_t>(byte_at(input, offset + 3)) << 24);
}

bool within(std::size_t position, std::size_t count, std::size_t end) {
    return count <= end && position <= end - count;
}

GfdCacheError error_of(std::string message) { return {std::move(message)}; }

std::vector<std::byte> unmask_utf(std::span<const std::byte> input) {
    std::vector<std::byte> out(input.begin(), input.end());
    std::uint32_t mask = 0x655f;
    for (std::byte& value : out) {
        value ^= static_cast<std::byte>(mask & 0xff);
        mask = (mask * 0x4115u) & 0xffffu;
    }
    return out;
}

struct UtfValue {
    enum class Kind : std::uint8_t { Missing, Unsigned, Signed, Real, Text, Blob };
    Kind kind = Kind::Missing;
    std::uint64_t unsigned_value = 0;
    std::int64_t signed_value = 0;
    std::string text;
};

struct UtfColumn {
    std::string name;
    std::uint8_t storage = 0;
    std::uint8_t type = 0;
    UtfValue constant;
};

struct UtfTable {
    std::vector<UtfColumn> columns;
    std::vector<std::vector<UtfValue>> rows;
};

struct UtfFailure {
    std::string message;
};

std::optional<std::uint64_t> as_size(const UtfValue& value) {
    if (value.kind == UtfValue::Kind::Missing) return 0;
    if (value.kind == UtfValue::Kind::Unsigned) return value.unsigned_value;
    if (value.kind == UtfValue::Kind::Signed && value.signed_value >= 0) {
        return static_cast<std::uint64_t>(value.signed_value);
    }
    return std::nullopt;
}

const UtfValue* column_value(const UtfTable& table, const std::vector<UtfValue>& row,
                             std::string_view name) {
    for (std::size_t index = 0; index < table.columns.size(); ++index) {
        if (table.columns[index].name == name) return &row[index];
    }
    return nullptr;
}

// `single_row_error`, when set, requires exactly one row and is returned
// before any column or row storage is built.
std::variant<UtfTable, UtfFailure> parse_utf(std::span<const std::byte> packet, std::string_view label,
                                             std::string_view single_row_error = {}) {
    std::vector<std::byte> owned;
    std::span<const std::byte> view = packet;
    if (packet.size() < 0x14 || std::memcmp(packet.data() + 0x10, "@UTF", 4) != 0) {
        if (packet.size() < 0x10) return UtfFailure{std::string(label) + ": no @UTF table"};
        owned = std::vector<std::byte>(packet.begin(), packet.begin() + 0x10);
        const auto masked = unmask_utf(packet.subspan(0x10));
        owned.insert(owned.end(), masked.begin(), masked.end());
        view = owned;
        if (view.size() < 0x14 || std::memcmp(view.data() + 0x10, "@UTF", 4) != 0) {
            return UtfFailure{std::string(label) + ": no @UTF table"};
        }
    }
    if (view.size() < 0x18) return UtfFailure{std::string(label) + ": truncated @UTF header"};
    const auto table_size = static_cast<std::size_t>(read_be32(view, 0x14));
    if (table_size < 24 || !within(0x18, table_size, view.size())) {
        return UtfFailure{std::string(label) + ": @UTF table is truncated"};
    }
    const auto table = view.subspan(0x18, table_size);
    const auto rows_off = static_cast<std::size_t>(read_be16(table, 2));
    const auto string_off = static_cast<std::size_t>(read_be32(table, 4));
    const auto data_off = static_cast<std::size_t>(read_be32(table, 8));
    const auto column_count = static_cast<std::size_t>(read_be16(table, 16));
    const auto row_length = static_cast<std::size_t>(read_be16(table, 18));
    const auto row_count = static_cast<std::size_t>(read_be32(table, 20));
    if (column_count > kUtfColumnLimit || row_count > kUtfRowLimit) {
        return UtfFailure{std::string(label) + ": @UTF dimensions exceed the resource limit"};
    }
    if (rows_off < 24 || rows_off > string_off || string_off > table.size() || data_off > table.size()) {
        return UtfFailure{std::string(label) + ": @UTF table offsets are outside the table"};
    }
    // Constant-only tables store a row length of 0. A header is one such row,
    // so a zero length is a valid table rather than a truncated one.
    if (row_length != 0 && row_count > (string_off - rows_off) / row_length) {
        return UtfFailure{std::string(label) + ": @UTF table offsets are outside the table"};
    }
    if (!single_row_error.empty() && row_count != 1) {
        return UtfFailure{std::string(single_row_error)};
    }
    if (column_count != 0 && row_count > kUtfValueLimit / column_count) {
        return UtfFailure{std::string(label) + ": @UTF allocation exceeds the resource limit"};
    }

    std::size_t string_bytes = 0;
    auto charge_string = [&](std::size_t bytes) {
        if (bytes > kUtfStringBudget || string_bytes > kUtfStringBudget - bytes) return false;
        string_bytes += bytes;
        return true;
    };
    auto cstring = [&](std::size_t offset) -> std::variant<std::string, UtfFailure> {
        const auto at = string_off + offset;
        if (at >= table.size()) return UtfFailure{std::string(label) + ": @UTF string outside the table"};
        const auto end = std::find(table.begin() + static_cast<std::ptrdiff_t>(at), table.end(), std::byte{0});
        if (end == table.end()) return UtfFailure{std::string(label) + ": @UTF string outside the table"};
        const auto length = static_cast<std::size_t>(end - (table.begin() + static_cast<std::ptrdiff_t>(at)));
        if (!charge_string(length)) {
            return UtfFailure{std::string(label) + ": @UTF allocation exceeds the resource limit"};
        }
        std::string text;
        text.resize(length);
        for (std::size_t index = 0; index < text.size(); ++index) {
            text[index] = static_cast<char>(byte_at(table, at + index));
        }
        return text;
    };

    auto value_at = [&](std::uint8_t type, std::size_t position,
                        std::size_t limit) -> std::variant<std::pair<UtfValue, std::size_t>, UtfFailure> {
        auto fixed = [&](std::size_t width, bool negative) -> std::variant<std::pair<UtfValue, std::size_t>, UtfFailure> {
            if (!within(position, width, limit)) {
                return UtfFailure{std::string(label) + ": @UTF value outside its row"};
            }
            std::uint64_t bits = 0;
            for (std::size_t index = 0; index < width; ++index) {
                bits = (bits << 8) | byte_at(table, position + index);
            }
            UtfValue value;
            if (!negative) {
                value.kind = UtfValue::Kind::Unsigned;
                value.unsigned_value = bits;
            } else {
                const int bit_count = static_cast<int>(width * 8);
                value.kind = UtfValue::Kind::Signed;
                if (bit_count >= 64) {
                    value.signed_value = static_cast<std::int64_t>(bits);
                } else {
                    const auto sign = std::uint64_t{1} << (bit_count - 1);
                    value.signed_value = (bits & sign) == 0
                        ? static_cast<std::int64_t>(bits)
                        : static_cast<std::int64_t>(bits) -
                              static_cast<std::int64_t>(std::uint64_t{1} << bit_count);
                }
            }
            return std::pair{value, width};
        };
        switch (type) {
            case 0: return fixed(1, false);
            case 1: return fixed(1, true);
            case 2: return fixed(2, false);
            case 3: return fixed(2, true);
            case 4: return fixed(4, false);
            case 5: return fixed(4, true);
            case 6: return fixed(8, false);
            case 7: return fixed(8, true);
            case 8:
            case 9: {
                const std::size_t width = type == 8 ? 4 : 8;
                if (!within(position, width, limit)) {
                    return UtfFailure{std::string(label) + ": @UTF value outside its row"};
                }
                UtfValue value;
                value.kind = UtfValue::Kind::Real;
                return std::pair{value, width};
            }
            case 0x0a: {
                if (!within(position, 4, limit)) {
                    return UtfFailure{std::string(label) + ": @UTF value outside its row"};
                }
                auto text = cstring(read_be32(table, position));
                if (std::holds_alternative<UtfFailure>(text)) return std::get<UtfFailure>(text);
                UtfValue value;
                value.kind = UtfValue::Kind::Text;
                value.text = std::get<std::string>(std::move(text));
                return std::pair{std::move(value), std::size_t{4}};
            }
            case 0x0b: {
                if (!within(position, 8, limit)) {
                    return UtfFailure{std::string(label) + ": @UTF value outside its row"};
                }
                const auto offset = static_cast<std::size_t>(read_be32(table, position));
                const auto length = static_cast<std::size_t>(read_be32(table, position + 4));
                if (offset > table.size() - data_off || length > table.size() - data_off - offset) {
                    return UtfFailure{std::string(label) + ": @UTF data outside the table"};
                }
                UtfValue value;
                value.kind = UtfValue::Kind::Blob;
                return std::pair{value, std::size_t{8}};
            }
            default:
                return UtfFailure{std::string(label) + ": unsupported @UTF column type"};
        }
    };

    UtfTable parsed;
    std::size_t cursor = 24;
    parsed.columns.reserve(column_count);
    for (std::size_t index = 0; index < column_count; ++index) {
        if (!within(cursor, 5, rows_off)) {
            return UtfFailure{std::string(label) + ": @UTF columns overrun the rows"};
        }
        const auto flags = byte_at(table, cursor);
        const auto name_offset = read_be32(table, cursor + 1);
        cursor += 5;
        UtfColumn column;
        column.storage = static_cast<std::uint8_t>(flags & 0xf0);
        column.type = static_cast<std::uint8_t>(flags & 0x0f);
        auto name = cstring(name_offset);
        if (std::holds_alternative<UtfFailure>(name)) return std::get<UtfFailure>(name);
        column.name = std::get<std::string>(std::move(name));
        if (column.storage == 0x30) {
            auto constant = value_at(column.type, cursor, rows_off);
            if (std::holds_alternative<UtfFailure>(constant)) return std::get<UtfFailure>(constant);
            auto [value, width] = std::get<std::pair<UtfValue, std::size_t>>(std::move(constant));
            column.constant = std::move(value);
            cursor += width;
        } else if (column.storage != 0x10 && column.storage != 0x50) {
            return UtfFailure{std::string(label) + ": unsupported @UTF column storage"};
        }
        parsed.columns.push_back(std::move(column));
    }
    parsed.rows.reserve(row_count);
    for (std::size_t row_index = 0; row_index < row_count; ++row_index) {
        const auto row_at = rows_off + row_index * row_length;
        const auto row_end = row_at + row_length;
        std::size_t row_cursor = row_at;
        std::vector<UtfValue> row(parsed.columns.size());
        for (std::size_t column_index = 0; column_index < parsed.columns.size(); ++column_index) {
            const auto& column = parsed.columns[column_index];
            if (column.storage == 0x50) {
                auto read = value_at(column.type, row_cursor, row_end);
                if (std::holds_alternative<UtfFailure>(read)) return std::get<UtfFailure>(read);
                auto [value, width] = std::get<std::pair<UtfValue, std::size_t>>(std::move(read));
                row[column_index] = std::move(value);
                row_cursor += width;
            } else if (column.storage == 0x30) {
                if (column.constant.kind == UtfValue::Kind::Text &&
                    !charge_string(column.constant.text.size())) {
                    return UtfFailure{std::string(label) + ": @UTF allocation exceeds the resource limit"};
                }
                row[column_index] = column.constant;
            }
        }
        parsed.rows.push_back(std::move(row));
    }
    return parsed;
}

std::variant<std::vector<std::byte>, GfdCacheError> read_file_at(std::ifstream& input, std::uint64_t offset,
                                                                std::uint64_t size, std::string_view label) {
    if (size > gfd_table_limit && size > gfd_member_limit) {
        return error_of(std::string(label) + " exceeds the in-memory limit");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    input.clear();
    input.seekg(static_cast<std::streamoff>(offset));
    if (!input || (size != 0 && !input.read(reinterpret_cast<char*>(bytes.data()),
                                            static_cast<std::streamsize>(size)))) {
        return error_of(std::string(label) + " runs past the end of the file");
    }
    return bytes;
}

std::string member_name(const UtfTable& table, const std::vector<UtfValue>& row, std::string& error) {
    auto text = [&](std::string_view name) -> std::string {
        const auto* value = column_value(table, row, name);
        if (value == nullptr || value->kind == UtfValue::Kind::Missing) return {};
        if (value->kind != UtfValue::Kind::Text) {
            error = "CPK member name is not a string";
            return {};
        }
        return value->text;
    };
    auto name = text("DirName");
    if (!error.empty()) return {};
    const auto file = text("FileName");
    if (!error.empty()) return {};
    if (!name.empty()) name.push_back('/');
    name += file;
    while (!name.empty() && name.front() == '/') name.erase(name.begin());
    return name;
}

std::optional<std::uint64_t> numeric_column(const UtfTable& table, const std::vector<UtfValue>& row,
                                           std::string_view name, std::string& error) {
    const auto* value = column_value(table, row, name);
    if (value == nullptr) return 0;
    auto size = as_size(*value);
    if (!size) error = std::string(name) + " is not an integer";
    return size;
}

struct NeedleReader {
    std::span<const std::byte> data;
    std::size_t position = 0;
    std::size_t end = 0;
    std::string label;
    std::string error;

    void fail(std::string message) {
        if (error.empty()) error = label + ": " + std::move(message);
    }

    std::uint32_t u32() {
        if (!error.empty()) return 0;
        if (!within(position, 4, end)) {
            fail("u32 outside the package");
            return 0;
        }
        const auto value = read_be32(data, position);
        position += 4;
        return value;
    }

    std::string name() {
        if (!error.empty()) return {};
        const auto nul = std::find(data.begin() + static_cast<std::ptrdiff_t>(position),
                                   data.begin() + static_cast<std::ptrdiff_t>(end), std::byte{0});
        if (nul == data.begin() + static_cast<std::ptrdiff_t>(end)) {
            fail("unterminated name");
            return {};
        }
        const auto length = static_cast<std::size_t>(nul - (data.begin() + static_cast<std::ptrdiff_t>(position)));
        std::string text(length, '\0');
        for (std::size_t index = 0; index < length; ++index) {
            text[index] = static_cast<char>(byte_at(data, position + index));
        }
        const auto advance = (length + 4) & ~std::size_t{3};
        if (!within(position, advance, end)) {
            fail("name padding outside the package");
            return {};
        }
        position += advance;
        return text;
    }

    // Returns the chunk end. `tags` is a list of accepted 8-byte tags.
    std::size_t chunk(std::initializer_list<std::string_view> tags) {
        if (!error.empty()) return 0;
        if (!within(position, 12, end)) {
            fail("expected package chunk");
            return 0;
        }
        const auto tag = data.subspan(position, 8);
        bool matched = false;
        for (const auto expected : tags) {
            if (expected.size() == 8 && std::memcmp(tag.data(), expected.data(), 8) == 0) matched = true;
        }
        if (!matched) {
            fail("expected package chunk");
            return 0;
        }
        const auto chunk_end = position + 8 + static_cast<std::size_t>(read_be32(data, position + 8));
        if (chunk_end < position || chunk_end > end) {
            fail("chunk ends outside the package");
            return 0;
        }
        position += 12;
        name();
        if (!error.empty()) return 0;
        if (position > chunk_end) {
            fail("name runs past the chunk");
            return 0;
        }
        return chunk_end;
    }
};

std::variant<OwnedShaderStage, std::string> descriptor_stage(std::span<const std::byte> data, std::size_t start,
                                                            std::size_t size, std::string where) {
    if (size >= 4 && std::memcmp(data.data() + start, "BNSH", 4) == 0) {
        return where + ": BNSH program body is not a descriptor";
    }
    if (size < kDescriptorMinimum || !within(start, size, data.size())) {
        return where + ": descriptor is below its minimum size";
    }
    const auto view = data.subspan(start, size);
    const auto head = read_le32(view, 0);
    const auto control_offset = static_cast<std::size_t>(read_le32(view, 0x30));
    const auto code_offset = static_cast<std::size_t>(read_le32(view, 0x34));
    const auto code_size = static_cast<std::size_t>(read_le32(view, 0x38));
    const auto control_size = static_cast<std::size_t>(read_le32(view, 0x3c));
    if (head > size || control_offset > size || code_offset > size || code_size > size || control_size > size ||
        control_offset > size - control_size || code_offset > size - code_size) {
        return where + ": descriptor field exceeds the descriptor";
    }
    const auto control = view.subspan(control_offset, control_size);
    const auto code = view.subspan(code_offset, code_size);
    const auto parsed = parse_shader(
        {ByteSpan{reinterpret_cast<const std::uint8_t*>(control.data()), control.size()},
         ByteSpan{reinterpret_cast<const std::uint8_t*>(code.data()), code.size()},
         {}});
    if (const auto* failure = std::get_if<ParseError>(&parsed)) {
        return where + ": " + std::string(failure->message());
    }
    const auto& shader = std::get<ParsedShader>(parsed);
    if (shader.control_size != control.size() || shader.code_size != code.size()) {
        return where + ": control or code length does not match its header";
    }
    OwnedShaderStage stage;
    stage.source = std::move(where);
    stage.stage = shader.stage;
    stage.control.assign(control.begin(), control.begin() + static_cast<std::ptrdiff_t>(shader.control_size));
    stage.code.assign(code.begin(), code.begin() + static_cast<std::ptrdiff_t>(shader.code_size));
    return stage;
}

std::variant<std::vector<OwnedShaderStage>, std::string> parse_needle(std::span<const std::byte> data,
                                                                      std::string label) {
    NeedleReader reader{data, 0, data.size(), std::move(label), {}};
    const auto needle_end = reader.chunk({"HHNEEDLE"});
    if (!reader.error.empty()) return reader.error;
    const auto saved_end = reader.end;
    const auto tag_at = reader.position;
    if (!within(tag_at, 8, reader.end)) return reader.label + ": expected package chunk";
    char tag_text[8];
    std::memcpy(tag_text, data.data() + tag_at, 8);
    const std::string_view tag(tag_text, 8);
    int mode = -1;
    if (tag == "HNSHV001") mode = 0;
    else if (tag == "HNSHV002") mode = 2;
    else return reader.label + ": expected package chunk";
    const auto hnsh_end = reader.chunk({tag});
    if (!reader.error.empty()) return reader.error;
    reader.end = std::min(saved_end, hnsh_end);
    const auto options = reader.u32();
    if (!reader.error.empty()) return reader.error;
    if (options > kOptionLimit) return reader.label + ": option count exceeds the resource limit";
    for (std::uint32_t index = 0; index < options; ++index) {
        reader.u32();
        reader.name();
        if (!reader.error.empty()) return reader.error;
    }
    std::uint32_t programs = 0;
    if (mode == 2) {
        const auto entries = std::size_t{1} << options;
        for (std::size_t index = 0; index < std::max<std::size_t>(entries, 1); ++index) reader.u32();
        programs = reader.u32();
    } else {
        programs = 1u << options;
    }
    if (!reader.error.empty()) return reader.error;
    if (programs > kProgramLimit) return reader.label + ": program count exceeds the resource limit";
    std::vector<OwnedShaderStage> stages;
    stages.reserve(programs);
    for (std::uint32_t program = 0; program < programs; ++program) {
        const auto length = reader.u32();
        if (!reader.error.empty()) return reader.error;
        if (length < 8 || !within(reader.position, static_cast<std::size_t>(length) - 4, reader.end)) {
            return reader.label + ": program length is outside the package";
        }
        const auto body = reader.position;
        const auto body_end = reader.position + static_cast<std::size_t>(length) - 4;
        reader.position = body_end;
        if (!within(body, 4, body_end)) return reader.label + ": program descriptor size runs past its body";
        const auto descriptor_size = static_cast<std::size_t>(read_be32(data, body));
        if (descriptor_size > body_end - body - 4) {
            return reader.label + ": program descriptor size runs past its body";
        }
        auto stage = descriptor_stage(data, body + 4, descriptor_size,
                                      reader.label + "/prog" + std::to_string(program));
        if (std::holds_alternative<std::string>(stage)) return std::get<std::string>(std::move(stage));
        stages.push_back(std::get<OwnedShaderStage>(std::move(stage)));
    }
    if (reader.position != hnsh_end) return reader.label + ": programs do not end with the package chunk";
    if (hnsh_end != needle_end || needle_end != data.size()) {
        return reader.label + ": package chunks do not end together";
    }
    return stages;
}

}  // namespace

CpkListResult list_cpk(const std::filesystem::path& path) {
    std::error_code filesystem_error;
    const auto file_size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error) return error_of("Cannot read the size of " + path.string());
    std::ifstream input(path, std::ios::binary);
    if (!input) return error_of("Cannot read " + path.string());
    const auto label = path.filename().string();
    auto head = read_file_at(input, 0, 0x10, label);
    if (std::holds_alternative<GfdCacheError>(head)) return std::get<GfdCacheError>(std::move(head));
    const auto& head_bytes = std::get<std::vector<std::byte>>(head);
    if (std::memcmp(head_bytes.data(), "CPK ", 4) != 0) return error_of(label + ": not a CPK archive");
    const auto packet_size = static_cast<std::uint64_t>(read_le32(head_bytes, 8));
    if (packet_size < 8 || packet_size > gfd_table_limit || packet_size + 0x10 > file_size ||
        packet_size + 0x10 > gfd_table_limit) {
        return error_of(label + ": CPK header packet size is out of range");
    }
    auto header_packet = read_file_at(input, 0, packet_size + 0x10, label);
    if (std::holds_alternative<GfdCacheError>(header_packet)) return std::get<GfdCacheError>(std::move(header_packet));
    auto header = parse_utf(std::get<std::vector<std::byte>>(header_packet), label + " header",
                            label + ": CPK header table does not have one row");
    if (std::holds_alternative<UtfFailure>(header)) return error_of(std::get<UtfFailure>(header).message);
    const auto& header_table = std::get<UtfTable>(header);
    if (header_table.rows.size() != 1) return error_of(label + ": CPK header table does not have one row");
    std::string numeric_error;
    const auto toc_offset = numeric_column(header_table, header_table.rows[0], "TocOffset", numeric_error);
    const auto toc_size = numeric_column(header_table, header_table.rows[0], "TocSize", numeric_error);
    const auto content_offset = numeric_column(header_table, header_table.rows[0], "ContentOffset", numeric_error);
    if (!numeric_error.empty() || !toc_offset || !toc_size || !content_offset) {
        return error_of(label + ": " + (numeric_error.empty() ? "CPK header field is not an integer" : numeric_error));
    }
    if (*toc_offset == 0 || *toc_size == 0) return error_of(label + ": CPK has no TOC");
    if (*toc_size > gfd_table_limit || *toc_offset > file_size || *toc_size > file_size - *toc_offset ||
        *toc_offset + 0x10 + *toc_size > file_size) {
        return error_of(label + ": TOC is outside the archive");
    }
    auto toc_packet = read_file_at(input, *toc_offset, *toc_size + 0x10, label);
    if (std::holds_alternative<GfdCacheError>(toc_packet)) return std::get<GfdCacheError>(std::move(toc_packet));
    const auto& toc_bytes = std::get<std::vector<std::byte>>(toc_packet);
    if (std::memcmp(toc_bytes.data(), "TOC ", 4) != 0) return error_of(label + ": no TOC packet");
    auto toc = parse_utf(toc_bytes, label + " TOC");
    if (std::holds_alternative<UtfFailure>(toc)) return error_of(std::get<UtfFailure>(toc).message);
    const auto& toc_table = std::get<UtfTable>(toc);
    const auto base = *content_offset == 0 ? *toc_offset : std::min(*toc_offset, *content_offset);
    CpkListing listing;
    listing.members.reserve(toc_table.rows.size());
    for (const auto& row : toc_table.rows) {
        std::string name_error;
        auto name = member_name(toc_table, row, name_error);
        if (!name_error.empty()) return error_of(label + ": " + name_error);
        const auto file_offset = numeric_column(toc_table, row, "FileOffset", name_error);
        const auto stored = numeric_column(toc_table, row, "FileSize", name_error);
        auto extracted = numeric_column(toc_table, row, "ExtractSize", name_error);
        if (!name_error.empty() || !file_offset || !stored || !extracted) {
            return error_of(label + ": " + (name_error.empty() ? "CPK member size is not an integer" : name_error));
        }
        if (*extracted == 0) extracted = stored;
        if (*file_offset > UINT64_MAX - base) return error_of(label + "/" + name + ": member is outside the archive");
        const auto offset = base + *file_offset;
        if (offset > file_size || *stored > file_size - offset) {
            return error_of(label + "/" + name + ": member is outside the archive");
        }
        listing.members.push_back({std::move(name), offset, *stored, *extracted});
    }
    return listing;
}

GfdBytesResult decompress_crilayla(std::span<const std::byte> input, std::size_t limit) {
    if (input.size() < 0x110 || std::memcmp(input.data(), "CRILAYLA", 8) != 0) {
        return error_of("malformed CRILAYLA header");
    }
    const auto payload = static_cast<std::size_t>(read_le32(input, 8));
    const auto packed = static_cast<std::size_t>(read_le32(input, 12));
    if (packed > input.size() || 0x10 + packed + 0x100 != input.size()) {
        return error_of("CRILAYLA compressed length does not match the stored bytes");
    }
    if (payload > limit || 0x100 > limit - payload) {
        return error_of("CRILAYLA decoded size is above the in-memory limit");
    }
    std::vector<std::byte> output(payload + 0x100);
    std::memcpy(output.data(), input.data() + 0x10 + packed, 0x100);
    std::size_t source = 0x10 + packed - 1;
    std::size_t destination = output.size() - 1;
    std::uint64_t pool = 0;
    int bits = 0;
    std::string failure;
    auto take = [&](int count) -> std::uint64_t {
        if (!failure.empty()) return 0;
        while (bits < count) {
            if (source < 0x10) {
                failure = "truncated CRILAYLA stream";
                return 0;
            }
            pool = (pool << 8) | byte_at(input, source);
            --source;
            bits += 8;
        }
        bits -= count;
        const auto value = pool >> bits;
        pool = bits == 0 ? 0 : pool & ((std::uint64_t{1} << bits) - 1);
        return value;
    };
    while (failure.empty() && destination >= 0x100) {
        if (take(1) == 0) {
            output[destination] = static_cast<std::byte>(take(8));
            --destination;
            continue;
        }
        const auto distance = static_cast<std::size_t>(take(13)) + 3;
        std::size_t length = 3;
        bool extended = true;
        for (const int width : {2, 3, 5, 8}) {
            const auto piece = take(width);
            length += static_cast<std::size_t>(piece);
            if (piece != (std::uint64_t{1} << width) - 1) {
                extended = false;
                break;
            }
        }
        if (extended) {
            for (;;) {
                const auto piece = take(8);
                if (length > output.size() - piece) {
                    failure = "CRILAYLA copy is too long";
                    break;
                }
                length += static_cast<std::size_t>(piece);
                if (piece != 0xff) break;
            }
        }
        if (!failure.empty()) break;
        if (distance > output.size() - 1 - destination) {
            failure = "CRILAYLA back-reference past the end of the output";
            break;
        }
        if (length == 0 || destination < 0x100 + length - 1) {
            failure = "CRILAYLA copy runs into the stored prefix";
            break;
        }
        for (std::size_t left = 0; left < length; ++left) {
            output[destination] = output[destination + distance];
            --destination;
        }
    }
    if (!failure.empty()) return error_of(std::move(failure));
    return output;
}

GfdBytesResult read_cpk_member(const std::filesystem::path& path, const CpkMember& member) {
    if (member.stored > gfd_member_limit || member.extracted > gfd_member_limit) {
        return error_of(member.name + " exceeds the in-memory limit");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) return error_of("Cannot read " + path.string());
    auto stored = read_file_at(input, member.offset, member.stored, member.name);
    if (std::holds_alternative<GfdCacheError>(stored)) return std::get<GfdCacheError>(std::move(stored));
    auto& bytes = std::get<std::vector<std::byte>>(stored);
    std::vector<std::byte> decoded;
    if (bytes.size() >= 8 && std::memcmp(bytes.data(), "CRILAYLA", 8) == 0) {
        auto expanded = decompress_crilayla(bytes, gfd_member_limit);
        if (std::holds_alternative<GfdCacheError>(expanded)) return expanded;
        decoded = std::get<std::vector<std::byte>>(std::move(expanded));
    } else if (member.stored == member.extracted) {
        decoded = std::move(bytes);
    } else {
        return error_of(member.name + " is compressed without a CRILAYLA header");
    }
    if (decoded.size() != member.extracted) return error_of(member.name + ": decoded length does not match the TOC");
    return decoded;
}

GfdStagesResult parse_gsc(std::span<const std::byte> input, std::string_view label) {
    if (input.size() < 0x20 || std::memcmp(input.data(), "GFS0", 4) != 0) {
        return error_of(std::string(label) + ": no GFS0 header");
    }
    const auto magic = read_be32(input, 0x10);
    const auto version = read_be32(input, 0x14);
    const auto count = read_be32(input, 0x18);
    if (magic != kGscMagic ||
        std::find(kGscVersions.begin(), kGscVersions.end(), version) == kGscVersions.end() ||
        count != kGscExpectedCount) {
        return error_of(std::string(label) + ": GSC0 header is not one the parser accepts");
    }
    std::vector<OwnedShaderStage> stages;
    std::size_t position = 0x20;
    std::size_t records = 0;
    while (within(position, kGscRecordSize, input.size())) {
        const auto type = read_be16(input, position + 4);
        const auto payload = static_cast<std::size_t>(read_be32(input, position + 6));
        const auto payload_at = position + kGscRecordSize;
        if (payload > input.size() - payload_at) {
            return error_of(std::string(label) + ": shader-cache record runs past the end");
        }
        if ((type == 0 || type == 1) && payload != 0) {
            const auto where = std::string(label) + "#rec" + std::to_string(records);
            auto programs = parse_needle(input.subspan(payload_at, payload), where);
            if (std::holds_alternative<std::string>(programs)) {
                return error_of(std::get<std::string>(std::move(programs)));
            }
            for (auto& stage : std::get<std::vector<OwnedShaderStage>>(programs)) {
                const auto expected = type == 0 ? Stage::Vertex : Stage::Fragment;
                if (stage.stage != expected) {
                    return error_of(stage.source + ": shader-cache record type does not match the stage");
                }
                stages.push_back(std::move(stage));
            }
        }
        position = payload_at + payload;
        ++records;
    }
    if (position != input.size()) return error_of(std::string(label) + ": bytes remain after the last record");
    return stages;
}

NeedleScan scan_hhneedle(std::span<const std::byte> image, std::string_view label,
                         std::span<const std::uint64_t> ignored_offsets) {
    NeedleScan scan;
    constexpr std::array<std::byte, 8> kTag{std::byte{'H'}, std::byte{'H'}, std::byte{'N'}, std::byte{'E'},
                                            std::byte{'E'}, std::byte{'D'}, std::byte{'L'}, std::byte{'E'}};
    std::unordered_set<std::string> covered;
    auto ignored = [&](std::size_t offset) {
        return std::find(ignored_offsets.begin(), ignored_offsets.end(), offset) != ignored_offsets.end();
    };
    auto cursor = image.begin();
    while (cursor != image.end()) {
        const auto found = std::search(cursor, image.end(), kTag.begin(), kTag.end());
        if (found == image.end()) break;
        const auto offset = static_cast<std::size_t>(found - image.begin());
        const auto where = std::string(label) + "@" + std::to_string(offset);
        auto reject = [&](std::string reason) {
            if (!ignored(offset)) scan.rejected.push_back(where + ": " + std::move(reason));
        };
        if (!within(offset, 12, image.size())) {
            reject("truncated chunk header");
        } else {
            const auto chunk_end = offset + 8 + static_cast<std::size_t>(read_be32(image, offset + 8));
            if (chunk_end < offset || chunk_end > image.size()) {
                reject("HHNEEDLE chunk runs past the image");
            } else {
                auto programs = parse_needle(image.subspan(offset, chunk_end - offset), where);
                if (std::holds_alternative<std::string>(programs)) {
                    reject(std::get<std::string>(programs));
                } else {
                    for (auto& stage : std::get<std::vector<OwnedShaderStage>>(programs)) {
                        covered.emplace(reinterpret_cast<const char*>(stage.control.data()), stage.control.size());
                        scan.stages.push_back(std::move(stage));
                    }
                }
            }
        }
        cursor = found + static_cast<std::ptrdiff_t>(kTag.size());
    }
    constexpr std::array<std::byte, 4> kMagic{std::byte{0x34}, std::byte{0x12}, std::byte{0x76}, std::byte{0x98}};
    cursor = image.begin();
    while (cursor != image.end()) {
        const auto found = std::search(cursor, image.end(), kMagic.begin(), kMagic.end());
        if (found == image.end()) break;
        const auto offset = static_cast<std::size_t>(found - image.begin());
        std::size_t declared = 0;
        if (within(offset, format::control_size_offset + 4, image.size())) {
            declared = read_le32(image, offset + format::control_size_offset);
        }
        const bool accounted = declared != 0 && within(offset, declared, image.size()) &&
            covered.contains(std::string(reinterpret_cast<const char*>(image.data() + offset), declared));
        if (!accounted) scan.uncovered_controls.push_back(offset);
        cursor = found + static_cast<std::ptrdiff_t>(kMagic.size());
    }
    return scan;
}

std::vector<std::string> missing_shader_caches(
    const std::vector<std::string>& present, const std::vector<std::string>& required,
    const std::vector<std::pair<std::string, std::vector<std::string>>>& fallbacks) {
    std::unordered_set<std::string> have(present.begin(), present.end());
    std::vector<std::string> missing;
    for (const auto& name : required) {
        if (have.contains(name)) continue;
        bool satisfied = false;
        for (const auto& [cache, alternatives] : fallbacks) {
            if (cache != name) continue;
            satisfied = std::any_of(alternatives.begin(), alternatives.end(),
                                    [&](const std::string& alternative) { return have.contains(alternative); });
            break;
        }
        if (!satisfied) missing.push_back(name);
    }
    return missing;
}

}  // namespace armigo::shader_input
