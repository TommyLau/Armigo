// ============================================================================
// Armigo Compatibility - Guest Instruction Observers
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/compat/observer.h>
#include <armigo/aarch64/instruction.h>
#include <armigo/common/diagnostics.h>
#include <armigo/execution/gate.h>
#include <armigo/horizon/ipc.h>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <pthread.h>

namespace armigo::compat {
namespace {

std::chrono::steady_clock::time_point observe_epoch() {
    static const auto epoch = std::chrono::steady_clock::now();
    return epoch;
}

bool blank(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

std::string_view trim(std::string_view text) {
    while (!text.empty() && blank(text.front())) text.remove_prefix(1);
    while (!text.empty() && blank(text.back())) text.remove_suffix(1);
    return text;
}

class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    // The whole text as one expression.
    std::optional<ObserveExpression> expression(std::string& error) {
        const auto root = binary(0);
        skip();
        if (root >= 0 && at_ != text_.size()) fail("unexpected text");
        if (!error_.empty()) {
            error = "'" + std::string(text_) + "': " + error_;
            return std::nullopt;
        }
        out_.root = root;
        return std::move(out_);
    }

private:
    static constexpr int levels = 7;

    void skip() {
        while (at_ < text_.size() && blank(text_[at_])) ++at_;
    }
    bool take(std::string_view token) {
        skip();
        if (text_.substr(at_, token.size()) != token) return false;
        at_ += token.size();
        return true;
    }
    bool next_is(std::string_view token) {
        skip();
        return text_.substr(at_, token.size()) == token;
    }
    std::int32_t fail(const char* message) {
        if (error_.empty()) error_ = message;
        return -1;
    }
    std::int32_t add(ObserveNode node) {
        out_.nodes.push_back(node);
        return static_cast<std::int32_t>(out_.nodes.size() - 1);
    }
    std::optional<std::uint64_t> number() {
        skip();
        int base = 10;
        if (text_.substr(at_, 2) == "0x" || text_.substr(at_, 2) == "0X") {
            base = 16;
            at_ += 2;
        }
        std::uint64_t value = 0;
        const auto* begin = text_.data() + at_;
        const auto parsed = std::from_chars(begin, text_.data() + text_.size(), value, base);
        if (parsed.ec != std::errc{} || parsed.ptr == begin) return std::nullopt;
        at_ += static_cast<std::size_t>(parsed.ptr - begin);
        return value;
    }
    std::int32_t primary() {
        if (take("(")) {
            const auto inner = binary(0);
            if (inner < 0) return -1;
            return take(")") ? inner : fail("missing ')'");
        }
        static constexpr std::pair<std::string_view, std::uint8_t> loads[] = {
            {"[", 8}, {"w[", 4}, {"h[", 2}, {"b[", 1}};
        for (const auto& [prefix, width] : loads)
            if (take(prefix)) {
                const auto address = binary(0);
                if (address < 0) return -1;
                if (!take("]")) return fail("missing ']'");
                return add({.kind = ObserveNode::Kind::load, .width = width, .left = address});
            }
        if (take("sp")) return add({.kind = ObserveNode::Kind::sp});
        skip();
        if (at_ < text_.size() && (text_[at_] == 'x' || text_[at_] == 'w')) {
            const bool wide = text_[at_] == 'x';
            ++at_;
            const auto index = number();
            if (!index || *index > 30) return fail("registers are x0..x30 and w0..w30");
            return add({.kind = wide ? ObserveNode::Kind::x : ObserveNode::Kind::w, .value = *index});
        }
        if (const auto value = number()) return add({.kind = ObserveNode::Kind::constant, .value = *value});
        return fail("expected a number, register, load or '('");
    }
    std::int32_t unary() {
        for (const char op : {'!', '~', '-'}) {
            if (op == '!' && next_is("!=")) break;
            if (take(std::string_view(&op, 1))) {
                const auto operand = unary();
                if (operand < 0) return -1;
                return add({.kind = ObserveNode::Kind::unary, .op = op, .left = operand});
            }
        }
        return primary();
    }
    // Level 0 binds loosest. Each entry: token, operator code.
    std::optional<char> binary_operator(int level) {
        switch (level) {
        case 0:
            if (take("==")) return 'e';
            if (take("!=")) return 'n';
            if (!next_is("<<") && take("<")) return '<';
            if (!next_is(">>") && take(">")) return '>';
            return std::nullopt;
        case 1: return take("|") ? std::optional<char>('|') : std::nullopt;
        case 2: return take("^") ? std::optional<char>('^') : std::nullopt;
        case 3: return take("&") ? std::optional<char>('&') : std::nullopt;
        case 4:
            if (take("<<")) return 'l';
            if (take(">>")) return 'r';
            return std::nullopt;
        case 5:
            if (take("+")) return '+';
            if (take("-")) return '-';
            return std::nullopt;
        default: return take("*") ? std::optional<char>('*') : std::nullopt;
        }
    }
    std::int32_t binary(int level) {
        if (level == levels) return unary();
        auto left = binary(level + 1);
        while (left >= 0) {
            const auto op = binary_operator(level);
            if (!op) break;
            const auto right = binary(level + 1);
            if (right < 0) return -1;
            left = add({.kind = ObserveNode::Kind::binary, .op = *op, .left = left, .right = right});
        }
        return left;
    }

    std::string_view text_;
    std::size_t at_ = 0;
    ObserveExpression out_;
    std::string error_;
};

std::optional<std::uint64_t> evaluate_node(const ObserveExpression& expression, std::int32_t index,
                                           const ObserveState& state) noexcept {
    if (index < 0 || static_cast<std::size_t>(index) >= expression.nodes.size()) return std::nullopt;
    const auto& node = expression.nodes[static_cast<std::size_t>(index)];
    switch (node.kind) {
    case ObserveNode::Kind::constant: return node.value;
    case ObserveNode::Kind::x: return state.x(static_cast<unsigned>(node.value));
    case ObserveNode::Kind::w: return state.x(static_cast<unsigned>(node.value)) & 0xffffffffu;
    case ObserveNode::Kind::sp: return state.sp();
    case ObserveNode::Kind::load: {
        const auto address = evaluate_node(expression, node.left, state);
        if (!address) return std::nullopt;
        std::array<std::byte, 8> bytes{};
        if (!state.read(*address, std::span(bytes).first(node.width))) return std::nullopt;
        std::uint64_t value = 0;
        for (unsigned i = 0; i < node.width; ++i) value |= std::uint64_t(bytes[i]) << (8 * i);
        return value;
    }
    case ObserveNode::Kind::unary: {
        const auto operand = evaluate_node(expression, node.left, state);
        if (!operand) return std::nullopt;
        if (node.op == '!') return std::uint64_t{*operand == 0};
        if (node.op == '~') return ~*operand;
        return std::uint64_t{0} - *operand;
    }
    case ObserveNode::Kind::binary: {
        const auto left = evaluate_node(expression, node.left, state);
        const auto right = left ? evaluate_node(expression, node.right, state) : std::nullopt;
        if (!right) return std::nullopt;
        switch (node.op) {
        case '*': return *left * *right;
        case '+': return *left + *right;
        case '-': return *left - *right;
        case 'l': return *right < 64 ? *left << *right : 0;
        case 'r': return *right < 64 ? *left >> *right : 0;
        case '&': return *left & *right;
        case '^': return *left ^ *right;
        case '|': return *left | *right;
        case 'e': return std::uint64_t{*left == *right};
        case 'n': return std::uint64_t{*left != *right};
        case '<': return std::uint64_t{*left < *right};
        case '>': return std::uint64_t{*left > *right};
        }
        return std::nullopt;
    }
    }
    return std::nullopt;
}

std::string hex(std::uint64_t value) {
    char text[24];
    std::snprintf(text, sizeof text, "%#llx", static_cast<unsigned long long>(value));
    return text;
}

class GateState final : public ObserveState {
public:
    GateState(const execution::GateCall& call, const horizon::GuestBuffers& memory) : call_(call), memory_(memory) {}
    // The stub keeps the guest's x30 at the gate's sp and moved sp down by 16.
    std::uint64_t x(unsigned index) const noexcept override {
        if (index < 30) return call_.x(index);
        std::uint64_t value = 0;
        read(call_.sp(), std::as_writable_bytes(std::span(&value, 1)));
        return value;
    }
    std::uint64_t sp() const noexcept override {
        return call_.sp() + execution::CallHookPatch::stack_adjustment;
    }
    bool read(std::uint64_t address, std::span<std::byte> out) const noexcept override {
        return memory_.read_memory(address, out);
    }

private:
    const execution::GateCall& call_;
    const horizon::GuestBuffers& memory_;
};

bool condition_holds(const ObservePoint& point, const ObserveState& state) noexcept {
    if (!point.condition) return true;
    const auto value = evaluate(*point.condition, state);
    return value && *value != 0;
}

std::string describe_items(const ObservePoint& point, const ObserveState& state) {
    std::string line;
    for (const auto& item : point.items) {
        if (!line.empty()) line += ' ';
        if (item.registers) {
            for (unsigned i = 0; i <= 30; ++i) line += "x" + std::to_string(i) + "=" + hex(state.x(i)) + " ";
            line += "sp=" + hex(state.sp());
            continue;
        }
        line += item.text + "=";
        const auto value = evaluate(item.expression, state);
        if (!value) {
            line += '?';
            continue;
        }
        if (!item.string) {
            line += hex(*value);
            continue;
        }
        std::array<char, 96> text{};
        // A string may end before the page that would hold 96 bytes does.
        std::size_t length = 0;
        for (; length < text.size() - 1; ++length) {
            std::byte byte{};
            if (!state.read(*value + length, std::span(&byte, 1)) || byte == std::byte{0}) break;
            const auto c = static_cast<char>(byte);
            text[length] = c >= 0x20 && c < 0x7f ? c : '?';
        }
        line += "'" + std::string(text.data(), length) + "'";
    }
    return line;
}

} // namespace

std::optional<std::uint64_t> evaluate(const ObserveExpression& expression, const ObserveState& state) noexcept {
    return evaluate_node(expression, expression.root, state);
}

std::optional<std::string> describe_hit(const ObservePoint& point, const ObserveState& state) {
    if (!condition_holds(point, state)) return std::nullopt;
    return describe_items(point, state);
}

std::variant<std::vector<ObservePoint>, std::string> parse_observe_points(std::string_view text) {
    observe_epoch();
    std::vector<ObservePoint> points;
    while (!text.empty()) {
        const auto end = text.find(';');
        const auto entry = trim(text.substr(0, end));
        text = end == std::string_view::npos ? std::string_view{} : text.substr(end + 1);
        if (entry.empty()) continue;
        ObservePoint point;
        point.text = std::string(entry);
        auto head = entry;
        std::string_view items;
        if (const auto colon = entry.find(':'); colon != std::string_view::npos) {
            // "s:" inside an item is not the separator; the first ':' after the
            // offset and condition is, and neither of them contains one.
            head = trim(entry.substr(0, colon));
            items = entry.substr(colon + 1);
        }
        std::string_view condition;
        if (const auto keyword = head.find(" if "); keyword != std::string_view::npos) {
            condition = trim(head.substr(keyword + 4));
            head = trim(head.substr(0, keyword));
        }
        std::string_view limit;
        if (const auto slash = head.find('/'); slash != std::string_view::npos) {
            limit = head.substr(slash + 1);
            head = head.substr(0, slash);
        }
        int base = 10;
        if (head.starts_with("0x") || head.starts_with("0X")) {
            head.remove_prefix(2);
            base = 16;
        }
        const auto offset = std::from_chars(head.data(), head.data() + head.size(), point.offset, base);
        if (head.empty() || offset.ec != std::errc{} || offset.ptr != head.data() + head.size() || point.offset % 4)
            return "Observer '" + point.text + "' needs an aligned main-module offset.";
        if (!limit.empty()) {
            const auto parsed = std::from_chars(limit.data(), limit.data() + limit.size(), point.limit);
            if (parsed.ec != std::errc{} || parsed.ptr != limit.data() + limit.size())
                return "Observer '" + point.text + "' has an invalid hit limit.";
        }
        std::string error;
        if (!condition.empty()) {
            point.condition = Parser(condition).expression(error);
            if (!point.condition) return "Observer condition " + error;
        }
        while (!trim(items).empty()) {
            const auto comma = items.find(',');
            const auto spelled = trim(items.substr(0, comma));
            items = comma == std::string_view::npos ? std::string_view{} : items.substr(comma + 1);
            ObserveItem item;
            item.text = std::string(spelled);
            if (spelled == "regs") {
                item.registers = true;
            } else {
                auto expression = spelled;
                if (expression.starts_with("s:")) {
                    item.string = true;
                    expression.remove_prefix(2);
                }
                auto parsed = Parser(expression).expression(error);
                if (!parsed) return "Observer item " + error;
                item.expression = std::move(*parsed);
            }
            point.items.push_back(std::move(item));
        }
        for (const auto& other : points)
            if (other.offset == point.offset) return "Observer offsets must be distinct.";
        points.push_back(std::move(point));
    }
    if (points.empty()) return std::string("ARMIGO_OBSERVE names no instruction.");
    return points;
}

std::variant<ObserveHook, std::string> make_observe_hook(std::uint32_t instruction, std::uint64_t address,
                                                         std::uint64_t trampoline,
                                                         execution::CallHookCodeRegion gate) noexcept {
    if (address % 4 || trampoline % 4 || gate.address % 8)
        return std::string("Observer stub layout is misaligned.");
    if (trampoline < address + 8 && address < trampoline + observe_stub_bytes)
        return std::string("Observer stub overlaps its instruction.");
    const auto sign_extend = [](std::uint32_t field, unsigned bits) {
        return static_cast<std::int64_t>(static_cast<std::uint64_t>(field) << (64 - bits)) >> (64 - bits);
    };
    // The branch forms whose test is kept and whose target is reached through
    // a nearby branch: the field holds the word offset of that branch.
    std::optional<std::uint64_t> branch_target;
    std::uint32_t displaced = instruction;
    if ((instruction & 0x7e000000u) == 0x36000000u) { // TBZ, TBNZ
        branch_target = address + static_cast<std::uint64_t>(sign_extend((instruction >> 5) & 0x3fff, 14) * 4);
        displaced = (instruction & ~(0x3fffu << 5)) | (2u << 5);
    } else if ((instruction & 0x7e000000u) == 0x34000000u || // CBZ, CBNZ
               (instruction & 0xff000000u) == 0x54000000u) { // B.cond, BC.cond
        branch_target = address + static_cast<std::uint64_t>(sign_extend((instruction >> 5) & 0x7ffff, 19) * 4);
        displaced = (instruction & ~(0x7ffffu << 5)) | (2u << 5);
    } else if ((instruction & 0x7c000000u) == 0x14000000u || // B, BL
               (instruction & 0x1f000000u) == 0x10000000u || // ADR, ADRP
               (instruction & 0x3b000000u) == 0x18000000u || // literal loads
               (instruction & 0xfe000000u) == 0xd6000000u || // register branches
               (instruction & 0xfe000000u) == 0xd4000000u || // exceptions, system
               (instruction & 0x3f000000u) == 0x08000000u) { // exclusive and ordered accesses
        return std::string("Observer instruction is PC-relative, a branch, exclusive or a system instruction.");
    }
    const auto redirect = aarch64::encode_b(address, trampoline);
    const auto call = aarch64::encode_bl(trampoline + 4, gate.address);
    const auto resume = aarch64::encode_b(trampoline + 16, address + 4);
    const auto taken = branch_target ? aarch64::encode_b(trampoline + 20, *branch_target) : aarch64::Encoded{};
    if (!redirect.ok() || !call.ok() || !resume.ok() || !taken.ok())
        return std::string("Observer stub is out of branch range.");
    // STR X30,[SP,#-16]!; BL gate; LDR X30,[SP],#16; displaced; B next; B target or UDF.
    ObserveHook hook;
    hook.replacement_instruction = redirect.word;
    hook.trampoline = {0xf81f0ffeu, call.word, 0xf84107feu, displaced, resume.word,
                       branch_target ? taken.word : 0u};
    hook.site = {trampoline + 8, address + 4, address + 4};
    return hook;
}

void observe_hit(const ObservePoint& point, const execution::GateCall& call,
                 const horizon::GuestBuffers& memory) noexcept {
    try {
        const GateState state(call, memory);
        if (!condition_holds(point, state)) return;
        const auto hit = point.hits->fetch_add(1, std::memory_order_relaxed);
        if (hit > point.limit) return;
        std::uint64_t thread = 0;
        pthread_threadid_np(nullptr, &thread);
        const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - observe_epoch()).count();
        if (hit == point.limit) {
            armigo::diag::write(armigo::diag::Level::info, "Armigo observe: main+%#llx reached its %u-hit limit\n",
                                static_cast<unsigned long long>(point.offset), point.limit);
            return;
        }
        const auto line = describe_items(point, state);
        armigo::diag::write(armigo::diag::Level::info, "Armigo observe: %.3f s main+%#llx hit %u thread %llu: %s\n",
                            seconds, static_cast<unsigned long long>(point.offset), hit + 1,
                            static_cast<unsigned long long>(thread), line.c_str());
    } catch (...) {
    }
}

} // namespace armigo::compat
