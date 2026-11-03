// ============================================================================
// Armigo Linker - Guest Dynamic Linking
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/linker/link_plan.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace armigo::linker {
namespace {

constexpr std::uint64_t module_alignment = 0x1000;
constexpr std::uint16_t section_reserved_low = 0xff00;
constexpr std::uint16_t section_absolute = 0xfff1;

constexpr std::uint8_t binding_local = 0;
constexpr std::uint8_t binding_global = 1;
constexpr std::uint8_t binding_weak = 2;

constexpr std::uint8_t type_section = 3;
constexpr std::uint8_t type_tls = 6;
constexpr std::uint8_t type_ifunc = 10;

constexpr std::uint8_t visibility_default = 0;

struct Failure {
    LinkError error;
    std::string message;
};

[[noreturn]] void fail(LinkError error, std::string message) {
    throw Failure{error, std::move(message)};
}

std::uint64_t load_le64(const std::byte* bytes) noexcept {
    std::uint64_t value = 0;
    for (unsigned index = 0; index < 8; ++index) {
        value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(bytes[index]))
                 << (index * 8);
    }
    return value;
}

void store_le64(std::byte* bytes, std::uint64_t value) noexcept {
    for (unsigned index = 0; index < 8; ++index) {
        bytes[index] = static_cast<std::byte>((value >> (index * 8)) & 0xff);
    }
}

// A non-cryptographic content check that detects stale or substituted images
// between planning and application. Exact preimages are checked separately.
std::uint64_t fingerprint(std::span<const std::byte> image) noexcept {
    constexpr std::uint64_t multiplier = 0x9e3779b97f4a7c15ull;
    std::uint64_t lanes[4] = {0x243f6a8885a308d3ull, 0x13198a2e03707344ull,
                              0xa4093822299f31d0ull, 0x082efa98ec4e6c89ull};
    const std::size_t size = image.size();
    std::size_t at = 0;
    for (; size - at >= 32; at += 32) {
        for (unsigned lane = 0; lane < 4; ++lane) {
            const auto word = load_le64(image.data() + at + lane * 8);
            lanes[lane] = (lanes[lane] ^ word) * multiplier;
            lanes[lane] ^= lanes[lane] >> 29;
        }
    }
    std::uint64_t hash = static_cast<std::uint64_t>(size) * multiplier;
    for (const auto lane : lanes) {
        hash = (hash ^ lane) * multiplier;
        hash ^= hash >> 31;
    }
    for (; at < size; ++at) {
        hash = (hash ^ std::to_integer<std::uint8_t>(image[at])) * multiplier;
        hash ^= hash >> 31;
    }
    return hash;
}

bool add_signed(std::uint64_t value, std::int64_t addend, std::uint64_t& result) noexcept {
    if (addend >= 0) {
        const auto magnitude = static_cast<std::uint64_t>(addend);
        if (value > std::numeric_limits<std::uint64_t>::max() - magnitude) {
            return false;
        }
        result = value + magnitude;
        return true;
    }
    // Negating through unsigned arithmetic is defined for the minimum value.
    const auto magnitude = std::uint64_t{0} - static_cast<std::uint64_t>(addend);
    if (value < magnitude) {
        return false;
    }
    result = value - magnitude;
    return true;
}

std::uint8_t binding_of(const Symbol& symbol) noexcept { return symbol.info >> 4; }
std::uint8_t type_of(const Symbol& symbol) noexcept { return symbol.info & 0xf; }
std::uint8_t visibility_of(const Symbol& symbol) noexcept { return symbol.visibility & 3; }

bool exportable(const Symbol& symbol) noexcept {
    // Hidden (2) and internal (1) definitions stay within their module.
    const auto visibility = visibility_of(symbol);
    return symbol.defined() && binding_of(symbol) != binding_local && !symbol.name.empty() &&
           (visibility == visibility_default || visibility == 3);
}

struct Definition {
    std::uint32_t module = 0;
    std::uint32_t symbol = 0;
};

struct Resolution {
    enum class State : std::uint8_t { unseen, resolved, unresolved };
    State state = State::unseen;
    WriteSource source = WriteSource::null_symbol;
    std::uint32_t provider = 0;
    std::uint32_t provider_symbol = 0;
    std::uint64_t value = 0;
};

} // namespace

namespace detail {

class LinkPlanner {
public:
    LinkPlanner(std::span<const LinkModule> modules, const LinkOptions& options,
            const LinkLimits& limits, PlanResult& result)
        : modules_(modules), options_(options), limits_(limits), result_(result) {}

    LinkPlan run() {
        LinkPlan plan;
        if (options_.preemption != PreemptionModel::load_order &&
            options_.preemption != PreemptionModel::own_definition_first) {
            fail(LinkError::invalid_argument, "The symbol preemption model is unsupported.");
        }
        validate_modules();
        index_host(options_.forced, forced_);
        index_host(options_.fallback, fallback_);
        index_exports();
        plan.forced_uses_.assign(options_.forced.size(), 0);
        forced_uses_ = &plan.forced_uses_;
        plan.modules_.reserve(modules_.size());
        resolutions_.resize(modules_.size());
        for (std::size_t index = 0; index < modules_.size(); ++index) {
            plan_module(index, plan);
        }
        if (result_.unresolved_count != 0) {
            std::string message = std::to_string(result_.unresolved_count) +
                                  " strong symbol reference(s) are unresolved";
            if (!result_.unresolved.empty()) {
                message += "; first: " + describe(result_.unresolved.front().module) + " '" +
                           result_.unresolved.front().name + "'";
            }
            fail(LinkError::unresolved_symbol, message + ".");
        }
        if (options_.require_forced_bindings_used) {
            for (std::size_t index = 0; index < options_.forced.size(); ++index) {
                if (plan.forced_uses_[index] == 0) {
                    result_.unused_forced.emplace_back(options_.forced[index].name);
                }
            }
            if (!result_.unused_forced.empty()) {
                fail(LinkError::unused_forced_binding,
                     "Forced host binding '" + result_.unused_forced.front() +
                         "' is not used by any preemptible relocation.");
            }
        }
        return plan;
    }

private:
    std::string describe(std::size_t module) const {
        std::string text = "module " + std::to_string(module);
        if (module < modules_.size() && !modules_[module].name.empty()) {
            text += " (";
            text += modules_[module].name;
            text += ')';
        }
        return text;
    }

    [[noreturn]] void fail_module(LinkError error, std::size_t module, const std::string& what) const {
        fail(error, describe(module) + ": " + what);
    }

    [[noreturn]] void fail_symbol(LinkError error, std::size_t module, std::uint32_t symbol,
                                  const char* what) const {
        const auto& name = modules_[module].metadata->symbols[symbol].name;
        fail(error, describe(module) + ": symbol " + std::to_string(symbol) + " '" + name +
                        "' " + what);
    }

    void validate_modules() {
        if (modules_.empty()) {
            fail(LinkError::invalid_argument, "A link plan requires at least one module.");
        }
        if (modules_.size() > limits_.max_modules ||
            modules_.size() > std::numeric_limits<std::uint32_t>::max()) {
            fail(LinkError::resource_limit, "The module count exceeds its limit.");
        }
        std::size_t total_symbols = 0;
        ranges_.resize(modules_.size());
        for (std::size_t index = 0; index < modules_.size(); ++index) {
            const auto& module = modules_[index];
            if (module.metadata == nullptr) {
                fail_module(LinkError::invalid_argument, index, "dynamic metadata is missing.");
            }
            if (module.image.empty()) {
                fail_module(LinkError::invalid_argument, index, "the module image is empty.");
            }
            if (module.base % module_alignment != 0) {
                fail_module(LinkError::invalid_layout, index, "the base is not page aligned.");
            }
            if (module.span_size < module.image.size()) {
                fail_module(LinkError::invalid_layout, index, "the address span is smaller than the image.");
            }
            if (module.base > std::numeric_limits<std::uint64_t>::max() - module.span_size) {
                fail_module(LinkError::address_overflow, index, "the address span overflows.");
            }
            const auto& symbols = module.metadata->symbols;
            if (symbols.empty() || symbols[0].defined() || !symbols[0].name.empty() ||
                symbols[0].value != 0 || symbols[0].size != 0 ||
                symbols[0].info != 0 || symbols[0].visibility != 0) {
                fail_module(LinkError::invalid_metadata, index, "symbol zero is not the null symbol.");
            }
            if (symbols.size() > limits_.max_total_symbols - total_symbols ||
                symbols.size() > std::numeric_limits<std::uint32_t>::max()) {
                fail(LinkError::resource_limit, "The aggregate symbol count exceeds its limit.");
            }
            total_symbols += symbols.size();
            if (module.writable.size() > limits_.max_writable_ranges) {
                fail_module(LinkError::resource_limit, index, "writable ranges exceed their limit.");
            }
            auto& ranges = ranges_[index];
            ranges.assign(module.writable.begin(), module.writable.end());
            for (const auto& range : ranges) {
                if (range.size == 0 || range.offset > module.image.size() ||
                    range.size > module.image.size() - range.offset) {
                    fail_module(LinkError::invalid_argument, index,
                                "a writable range is empty or exceeds the image.");
                }
            }
            std::sort(ranges.begin(), ranges.end(), [](const ModuleRange& left, const ModuleRange& right) {
                return left.offset < right.offset;
            });
            // Coalesce so containment can be checked against a single interval.
            std::size_t kept = 0;
            for (const auto& range : ranges) {
                if (kept != 0 && range.offset <= ranges[kept - 1].offset + ranges[kept - 1].size) {
                    auto& last = ranges[kept - 1];
                    last.size = std::max(last.offset + last.size, range.offset + range.size) - last.offset;
                } else {
                    ranges[kept++] = range;
                }
            }
            ranges.resize(kept);
        }
        total_symbols_ = total_symbols;

        std::vector<std::size_t> order(modules_.size());
        for (std::size_t index = 0; index < order.size(); ++index) {
            order[index] = index;
        }
        std::sort(order.begin(), order.end(), [this](std::size_t left, std::size_t right) {
            return modules_[left].base < modules_[right].base;
        });
        for (std::size_t index = 1; index < order.size(); ++index) {
            const auto& previous = modules_[order[index - 1]];
            if (modules_[order[index]].base < previous.base + previous.span_size) {
                fail(LinkError::invalid_layout, describe(order[index - 1]) + " and " +
                                                     describe(order[index]) +
                                                     " have overlapping address ranges.");
            }
        }

        std::vector<std::pair<std::uintptr_t, std::size_t>> images(modules_.size());
        for (std::size_t index = 0; index < modules_.size(); ++index) {
            images[index] = {reinterpret_cast<std::uintptr_t>(modules_[index].image.data()), index};
        }
        std::sort(images.begin(), images.end());
        for (std::size_t index = 1; index < images.size(); ++index) {
            const auto& previous = images[index - 1];
            if (images[index].first - previous.first < modules_[previous.second].image.size()) {
                fail(LinkError::aliased_images, describe(previous.second) + " and " +
                                                    describe(images[index].second) +
                                                    " have overlapping input images.");
            }
        }
    }

    void index_host(std::span<const HostBinding> bindings,
                    std::unordered_map<std::string_view, std::uint32_t>& index) {
        if (bindings.size() > limits_.max_host_bindings) {
            fail(LinkError::resource_limit, "Host bindings exceed their limit.");
        }
        index.reserve(bindings.size());
        for (std::size_t at = 0; at < bindings.size(); ++at) {
            const auto& binding = bindings[at];
            if (binding.name.size() > limits_.max_host_name_bytes) {
                fail(LinkError::resource_limit, "A host binding name exceeds its limit.");
            }
            if (binding.name.empty() || binding.name.find('\0') != std::string_view::npos) {
                fail(LinkError::invalid_argument, "A host binding name is empty or contains NUL.");
            }
            if (binding.address == 0) {
                fail(LinkError::invalid_argument,
                     "Host binding '" + std::string(binding.name) + "' has a null address.");
            }
            if (!index.emplace(binding.name, static_cast<std::uint32_t>(at)).second) {
                fail(LinkError::duplicate_host_binding,
                     "Host binding '" + std::string(binding.name) + "' is duplicated.");
            }
        }
    }

    void index_exports() {
        exports_.reserve(total_symbols_);
        for (std::size_t module = 0; module < modules_.size(); ++module) {
            const auto& symbols = modules_[module].metadata->symbols;
            for (std::size_t symbol = 1; symbol < symbols.size(); ++symbol) {
                if (exportable(symbols[symbol])) {
                    // The first definition in load order, then symbol order, wins.
                    exports_.try_emplace(symbols[symbol].name,
                                         Definition{static_cast<std::uint32_t>(module),
                                                    static_cast<std::uint32_t>(symbol)});
                }
            }
        }
    }

    void check_symbol_kind(std::size_t module, std::uint32_t symbol) const {
        const auto& entry = modules_[module].metadata->symbols[symbol];
        const auto type = type_of(entry);
        if (type == type_tls) {
            fail_symbol(LinkError::unsupported_symbol, module, symbol,
                        "is thread-local; TLS relocation is not implemented.");
        }
        if (type == type_ifunc) {
            fail_symbol(LinkError::unsupported_symbol, module, symbol,
                        "is an indirect function; IFUNC resolution is not implemented.");
        }
        if (type > type_section) {
            fail_symbol(LinkError::unsupported_symbol, module, symbol, "has an unsupported type.");
        }
        const auto binding = binding_of(entry);
        if (binding != binding_local && binding != binding_global && binding != binding_weak) {
            fail_symbol(LinkError::unsupported_symbol, module, symbol, "has an unsupported binding.");
        }
        if (entry.section >= section_reserved_low && entry.section != section_absolute) {
            fail_symbol(LinkError::unsupported_symbol, module, symbol,
                        "uses a reserved section index.");
        }
    }

    Resolution definition(std::size_t module, std::uint32_t symbol) const {
        check_symbol_kind(module, symbol);
        const auto& link_module = modules_[module];
        const auto& entry = link_module.metadata->symbols[symbol];
        Resolution resolution;
        resolution.state = Resolution::State::resolved;
        resolution.provider = static_cast<std::uint32_t>(module);
        resolution.provider_symbol = symbol;
        if (entry.section == section_absolute) {
            resolution.source = WriteSource::absolute_symbol;
            resolution.value = entry.value;
            return resolution;
        }
        if (entry.value > link_module.span_size || entry.size > link_module.span_size - entry.value) {
            fail_symbol(LinkError::invalid_symbol, module, symbol,
                        "is defined outside its module address span.");
        }
        resolution.source = WriteSource::module_symbol;
        resolution.value = link_module.base + entry.value;
        return resolution;
    }

    const Resolution& resolve(std::size_t module, std::uint32_t symbol) {
        auto& cache = resolutions_[module];
        if (cache.empty()) {
            cache.resize(modules_[module].metadata->symbols.size());
        }
        auto& resolution = cache[symbol];
        if (resolution.state != Resolution::State::unseen) {
            return resolution;
        }
        if (symbol == 0) {
            resolution.state = Resolution::State::resolved;
            resolution.source = WriteSource::null_symbol;
            return resolution;
        }
        check_symbol_kind(module, symbol);
        const auto& entry = modules_[module].metadata->symbols[symbol];
        const bool preemptible =
            binding_of(entry) != binding_local && visibility_of(entry) == visibility_default;
        if (!preemptible) {
            if (!entry.defined()) {
                fail_symbol(LinkError::invalid_symbol, module, symbol,
                            "is a local or non-default-visibility reference without a definition.");
            }
            resolution = definition(module, symbol);
            return resolution;
        }
        if (entry.name.empty()) {
            fail_symbol(LinkError::invalid_symbol, module, symbol, "is a preemptible reference without a name.");
        }
        if (const auto forced = forced_.find(entry.name); forced != forced_.end()) {
            ++(*forced_uses_)[forced->second];
            resolution.state = Resolution::State::resolved;
            resolution.source = WriteSource::forced_host;
            resolution.provider = forced->second;
            resolution.value = options_.forced[forced->second].address;
            return resolution;
        }
        if (options_.preemption == PreemptionModel::own_definition_first && entry.defined()) {
            resolution = definition(module, symbol);
            return resolution;
        }
        if (const auto found = exports_.find(entry.name); found != exports_.end()) {
            resolution = definition(found->second.module, found->second.symbol);
            return resolution;
        }
        if (const auto host = fallback_.find(entry.name); host != fallback_.end()) {
            resolution.state = Resolution::State::resolved;
            resolution.source = WriteSource::host_fallback;
            resolution.provider = host->second;
            resolution.value = options_.fallback[host->second].address;
            return resolution;
        }
        if (binding_of(entry) == binding_weak) {
            resolution.state = Resolution::State::resolved;
            resolution.source = WriteSource::weak_unresolved;
            return resolution;
        }
        resolution.state = Resolution::State::unresolved;
        if (result_.unresolved.size() < limits_.max_reported_unresolved) {
            result_.unresolved.push_back({module, symbol, entry.name});
        }
        ++result_.unresolved_count;
        return resolution;
    }

    bool permitted(std::size_t module, std::uint64_t offset) const {
        const auto& ranges = ranges_[module];
        auto after = std::upper_bound(ranges.begin(), ranges.end(), offset,
                                      [](std::uint64_t value, const ModuleRange& range) {
                                          return value < range.offset;
                                      });
        if (after == ranges.begin()) {
            return false;
        }
        const auto& range = *(after - 1);
        return offset - range.offset <= range.size && range.size - (offset - range.offset) >= 8;
    }

    std::uint64_t preimage(std::size_t module, std::uint64_t target, const char* kind) const {
        const auto image = modules_[module].image;
        if (target % 8 != 0) {
            fail_module(LinkError::invalid_metadata, module,
                        std::string(kind) + " target " + std::to_string(target) + " is misaligned.");
        }
        if (image.size() < 8 || target > image.size() - 8) {
            fail_module(LinkError::invalid_metadata, module,
                        std::string(kind) + " target " + std::to_string(target) + " exceeds the image.");
        }
        if (!permitted(module, target)) {
            fail_module(LinkError::forbidden_target, module,
                        std::string(kind) + " target " + std::to_string(target) +
                            " is outside the writable ranges.");
        }
        return load_le64(image.data() + target);
    }

    void plan_module(std::size_t module, LinkPlan& plan) {
        const auto& link_module = modules_[module];
        const auto& metadata = *link_module.metadata;
        const std::size_t requested = metadata.packed_relative_targets.size();
        if (requested > limits_.max_writes - plan.writes_.size() ||
            metadata.relocations.size() > limits_.max_writes - plan.writes_.size() - requested) {
            fail_module(LinkError::resource_limit, module, "relocation writes exceed their limit.");
        }
        const std::size_t first = plan.writes_.size();
        plan.writes_.reserve(first + requested + metadata.relocations.size());

        for (const auto target : metadata.packed_relative_targets) {
            PlannedWrite write;
            write.offset = target;
            write.expected = preimage(module, target, "RELR");
            write.kind = RelocationKind::relative;
            write.source = WriteSource::packed_relative;
            write.provider = static_cast<std::uint32_t>(module);
            if (link_module.base > std::numeric_limits<std::uint64_t>::max() - write.expected) {
                fail_module(LinkError::address_overflow, module,
                            "RELR target " + std::to_string(target) + " overflows.");
            }
            write.value = link_module.base + write.expected;
            plan.writes_.push_back(write);
        }

        for (const auto& relocation : metadata.relocations) {
            PlannedWrite write;
            write.offset = relocation.target;
            write.kind = relocation.kind;
            write.symbol = relocation.symbol;
            if (relocation.symbol >= metadata.symbols.size()) {
                fail_module(LinkError::invalid_metadata, module,
                            "a relocation references symbol " + std::to_string(relocation.symbol) +
                                ", which does not exist.");
            }
            switch (relocation.kind) {
            case RelocationKind::relative:
                if (relocation.symbol != 0) {
                    fail_module(LinkError::invalid_metadata, module,
                                "a RELATIVE relocation references a symbol.");
                }
                write.expected = preimage(module, relocation.target, "RELATIVE");
                write.source = WriteSource::relative;
                write.provider = static_cast<std::uint32_t>(module);
                if (!add_signed(link_module.base, relocation.addend, write.value)) {
                    fail_module(LinkError::address_overflow, module,
                                "RELATIVE target " + std::to_string(relocation.target) + " overflows.");
                }
                break;
            case RelocationKind::absolute64:
            case RelocationKind::global_data:
            case RelocationKind::jump_slot: {
                write.expected = preimage(module, relocation.target, "symbolic");
                const auto& resolution = resolve(module, relocation.symbol);
                if (resolution.state == Resolution::State::unresolved) {
                    continue;
                }
                write.source = resolution.source;
                write.provider = resolution.provider;
                write.provider_symbol = resolution.provider_symbol;
                if (!add_signed(resolution.value, relocation.addend, write.value)) {
                    fail_module(LinkError::address_overflow, module,
                                "symbolic target " + std::to_string(relocation.target) + " overflows.");
                }
                break;
            }
            default:
                fail_module(LinkError::unsupported_relocation, module,
                            "relocation kind " +
                                std::to_string(static_cast<std::uint32_t>(relocation.kind)) +
                                " is not supported.");
            }
            plan.writes_.push_back(write);
        }

        const auto begin = plan.writes_.begin() + static_cast<std::ptrdiff_t>(first);
        std::sort(begin, plan.writes_.end(), [](const PlannedWrite& left, const PlannedWrite& right) {
            return left.offset < right.offset;
        });
        for (auto at = begin; at != plan.writes_.end(); ++at) {
            if (at != begin && at->offset - (at - 1)->offset < 8) {
                fail_module(LinkError::conflicting_write, module,
                            "relocation target " + std::to_string(at->offset) +
                                " is written more than once.");
            }
        }

        LinkPlan::ModuleState state;
        state.base = link_module.base;
        state.span_size = link_module.span_size;
        state.image_size = link_module.image.size();
        state.fingerprint = fingerprint(link_module.image);
        state.first_write = first;
        state.write_count = plan.writes_.size() - first;
        plan.modules_.push_back(state);
    }

    std::span<const LinkModule> modules_;
    const LinkOptions& options_;
    const LinkLimits& limits_;
    PlanResult& result_;
    std::size_t total_symbols_ = 0;
    std::vector<std::vector<ModuleRange>> ranges_;
    std::unordered_map<std::string_view, std::uint32_t> forced_;
    std::unordered_map<std::string_view, std::uint32_t> fallback_;
    std::unordered_map<std::string_view, Definition> exports_;
    std::vector<std::vector<Resolution>> resolutions_;
    std::vector<std::uint64_t>* forced_uses_ = nullptr;
};

} // namespace detail

LinkPlan::LinkPlan(LinkPlan&& other) noexcept
    : modules_(std::move(other.modules_)), writes_(std::move(other.writes_)),
      forced_uses_(std::move(other.forced_uses_)) {
    other.modules_.clear();
    other.writes_.clear();
    other.forced_uses_.clear();
}

LinkPlan& LinkPlan::operator=(LinkPlan&& other) noexcept {
    if (this != &other) {
        modules_ = std::move(other.modules_);
        writes_ = std::move(other.writes_);
        forced_uses_ = std::move(other.forced_uses_);
        other.modules_.clear();
        other.writes_.clear();
        other.forced_uses_.clear();
    }
    return *this;
}

std::span<const PlannedWrite> LinkPlan::writes(std::size_t module) const {
    const auto& state = modules_.at(module);
    return std::span<const PlannedWrite>(writes_).subspan(state.first_write, state.write_count);
}

ApplyResult LinkPlan::apply(std::span<const std::span<std::byte>> images) const noexcept {
    if (images.size() != modules_.size()) {
        return {ApplyError::module_count_mismatch, 0, 0,
                "The image count does not match the plan."};
    }
    const auto descriptors_begin = reinterpret_cast<std::uintptr_t>(images.data());
    const auto descriptors_size = images.size_bytes();
    for (std::size_t module = 0; module < modules_.size(); ++module) {
        if (images[module].size() != modules_[module].image_size) {
            return {ApplyError::image_size_mismatch, module, 0,
                    "An image size does not match the plan."};
        }
        const auto image_begin = reinterpret_cast<std::uintptr_t>(images[module].data());
        const bool separate = image_begin < descriptors_begin
            ? descriptors_begin - image_begin >= images[module].size()
            : image_begin - descriptors_begin >= descriptors_size;
        if (!separate) {
            return {ApplyError::aliased_descriptors, module, 0,
                    "Destination bytes overlap the image descriptor array."};
        }
    }
    for (std::size_t left = 0; left < images.size(); ++left) {
        const auto left_begin = reinterpret_cast<std::uintptr_t>(images[left].data());
        for (std::size_t right = left + 1; right < images.size(); ++right) {
            const auto right_begin = reinterpret_cast<std::uintptr_t>(images[right].data());
            const bool disjoint = left_begin < right_begin
                                      ? right_begin - left_begin >= images[left].size()
                                      : left_begin - right_begin >= images[right].size();
            if (!disjoint) {
                return {ApplyError::aliased_images, right, 0, "Destination images overlap."};
            }
        }
    }
    for (std::size_t module = 0; module < modules_.size(); ++module) {
        if (fingerprint(images[module]) != modules_[module].fingerprint) {
            return {ApplyError::image_mismatch, module, 0,
                    "An image does not match the planned content."};
        }
    }
    for (std::size_t module = 0; module < modules_.size(); ++module) {
        for (const auto& write : writes(module)) {
            if (load_le64(images[module].data() + write.offset) != write.expected) {
                return {ApplyError::preimage_mismatch, module, write.offset,
                        "A relocation target does not hold its planned preimage."};
            }
        }
    }
    for (std::size_t module = 0; module < modules_.size(); ++module) {
        const auto& state = modules_[module];
        std::byte* const data = images[module].data();
        for (std::size_t at = state.first_write; at < state.first_write + state.write_count; ++at) {
            store_le64(data + writes_[at].offset, writes_[at].value);
        }
    }
    return {};
}

PlanResult plan_link(std::span<const LinkModule> modules, const LinkOptions& options,
                     const LinkLimits& limits) {
    PlanResult result;
    try {
        auto plan = detail::LinkPlanner(modules, options, limits, result).run();
        result.plan.emplace(std::move(plan));
    } catch (Failure& failure) {
        result.error = failure.error;
        result.message = std::move(failure.message);
        result.plan.reset();
    } catch (const std::bad_alloc&) {
        result = {};
        result.error = LinkError::resource_limit;
        result.message = "Link plan allocation failed.";
    } catch (const std::length_error&) {
        result = {};
        result.error = LinkError::resource_limit;
        result.message = "Link plan allocation failed.";
    }
    if (result.error != LinkError::unresolved_symbol) {
        result.unresolved.clear();
        result.unresolved_count = 0;
    }
    return result;
}

} // namespace armigo::linker
