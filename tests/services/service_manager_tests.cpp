// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include "armigo/services/service_manager.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string_view>

namespace {
using namespace armigo::services::sm;
unsigned failures = 0;
#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "Failed at line %d: %s\n", __LINE__, #condition); ++failures; \
} } while (false)

using Packet = std::array<std::byte, 32>;

void put(Packet& packet, std::size_t at, std::uint64_t value, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
        packet[at + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

std::uint64_t get(const DispatchResult& out, std::size_t at, std::size_t count) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < count; ++i)
        value |= std::uint64_t{std::to_integer<unsigned char>(out.reply[at + i])} << (8 * i);
    return value;
}

ServiceName name(std::string_view text) {
    ServiceName out{};
    for (std::size_t i = 0; i < text.size() && i < out.size(); ++i)
        out[i] = static_cast<std::byte>(text[i]);
    return out;
}

Packet registration() {
    Packet out{};
    put(out, 0, 0x8000000000000010ULL, 8);
    put(out, 8, 1, 4);
    put(out, 12, 0xdeadbeef, 8);
    return out;
}

Packet request(const ServiceName& service) {
    Packet out{};
    put(out, 0, 0x200000011ULL, 8);
    std::copy(service.begin(), service.end(), out.begin() + 8);
    return out;
}

ProcessConfig config(std::string_view allowed, bool server = false) {
    ProcessConfig out{37, true, {}};
    out.service_access_control.push_back(static_cast<std::byte>((allowed.size() - 1) |
                                                               (server ? 0x80U : 0U)));
    for (const char c : allowed) out.service_access_control.push_back(static_cast<std::byte>(c));
    return out;
}

void register_client(Session& session) {
    auto packet = registration();
    const auto original = packet;
    const auto result = session.dispatch(packet, 37);
    CHECK(result.disposition == Disposition::handled && result.reply_size == 12);
    CHECK(get(result, 0, 8) == 0x100000010ULL && get(result, 8, 4) == 0);
    CHECK(packet == original);
}

void check_error(const DispatchResult& out, std::uint32_t code) {
    CHECK(out.disposition == Disposition::handled && out.reply_size == 20);
    CHECK(get(out, 0, 8) == 0x8000000100000011ULL);
    CHECK(get(out, 8, 4) == 0x20 && get(out, 12, 4) == 0 && get(out, 16, 4) == code);
}

void check_no_reply(const DispatchResult& out, Disposition disposition) {
    CHECK(out.disposition == disposition && out.reply_size == 0);
    CHECK(std::all_of(out.reply.begin(), out.reply.end(), [](std::byte b) { return b == std::byte{0}; }));
}

struct SyntheticClient final : armigo::horizon::KernelObject {};

void test_registration_and_connection() {
    armigo::horizon::HandleTable handles(4);
    unsigned calls = 0;
    Session session(config("unit"), [&](std::uint64_t pid, const ServiceName& service) {
        ++calls;
        CHECK(pid == 37 && service == name("unit"));
        const auto handle = handles.insert(std::make_shared<SyntheticClient>());
        CHECK(handle.has_value());
        return LookupResult{LookupKind::connected, handle.value_or(0), 0};
    });
    check_error(session.dispatch(request(name("unit"))), 0x415);
    CHECK(calls == 0);
    check_no_reply(session.dispatch(registration()), Disposition::unsupported);
    check_no_reply(session.dispatch(registration(), 38), Disposition::unsupported);
    check_error(session.dispatch(request(name("unit"))), 0x415);
    register_client(session);
    register_client(session);
    const auto out = session.dispatch(request(name("unit")));
    CHECK(out.disposition == Disposition::handled && out.reply_size == 20);
    CHECK(get(out, 0, 8) == 0x8000000100000011ULL && get(out, 8, 4) == 0x20);
    CHECK(get(out, 16, 4) == 0 && calls == 1);
    const auto handle = static_cast<armigo::horizon::Handle>(get(out, 12, 4));
    CHECK(handles.lookup_as<SyntheticClient>(handle) != nullptr);
    CHECK(handles.close(handle));

    auto unregistered = config("unit");
    unregistered.registered = false;
    Session without_process(std::move(unregistered), {});
    register_client(without_process);
    check_error(without_process.dispatch(request(name("unit"))), 0x415);
}

void test_names_and_acl() {
    unsigned calls = 0;
    const ConnectService backend = [&](std::uint64_t, const ServiceName&) {
        ++calls;
        return LookupResult{LookupKind::failed, 0, 0x777};
    };
    Session exact(config("unit"), backend);
    register_client(exact);
    check_error(exact.dispatch(request(name("other"))), 0x1015);
    check_error(exact.dispatch(request(name(""))), 0xc15);
    auto gapped = name("unit");
    gapped[1] = std::byte{0};
    check_error(exact.dispatch(request(gapped)), 0xc15);
    CHECK(calls == 0);
    check_error(exact.dispatch(request(name("unit"))), 0x777);
    CHECK(calls == 1);

    Session prefix(config("ab*"), backend);
    register_client(prefix);
    check_error(prefix.dispatch(request(name("ab"))), 0x777);
    check_error(prefix.dispatch(request(name("abcdefgh"))), 0x777);
    check_error(prefix.dispatch(request(name("a"))), 0x1015);
    Session wildcard(config("*"), backend);
    register_client(wildcard);
    check_error(wildcard.dispatch(request(name("eightbyt"))), 0x777);
    auto arbitrary = name("abcdefgh");
    arbitrary[3] = std::byte{0xff};
    check_error(wildcard.dispatch(request(arbitrary)), 0x777);
    Session server(config("*", true), backend);
    register_client(server);
    check_error(server.dispatch(request(name("unit"))), 0x1015);

    auto incomplete = config("unit");
    incomplete.service_access_control.pop_back();
    Session truncated(std::move(incomplete), backend);
    register_client(truncated);
    check_error(truncated.dispatch(request(name("unit"))), 0x1015);
    auto multi = config("else", true);
    const auto client = config("unit");
    multi.service_access_control.insert(multi.service_access_control.end(),
        client.service_access_control.begin(), client.service_access_control.end());
    multi.service_access_control[5] |= std::byte{0x78};
    Session ignored_bits(std::move(multi), backend);
    register_client(ignored_bits);
    check_error(ignored_bits.dispatch(request(name("unit"))), 0x777);
    Session eight(config("eightbyt"), backend);
    register_client(eight);
    check_error(eight.dispatch(request(name("eightbyt"))), 0x777);
}

void test_pending_and_failures() {
    bool available = false;
    bool unsupported = false;
    unsigned calls = 0;
    Session session(config("unit"), [&](std::uint64_t, const ServiceName& service) {
        ++calls;
        CHECK(service == name("unit"));
        if (unsupported) return LookupResult{};
        if (!available) return LookupResult{LookupKind::pending, 0, 0};
        return LookupResult{LookupKind::failed, 0, 0xe01};
    });
    register_client(session);
    const auto packet = request(name("unit"));
    const auto original = packet;
    check_no_reply(session.dispatch(packet), Disposition::pending);
    CHECK(session.has_pending_request() && packet == original && calls == 1);
    check_no_reply(session.dispatch(registration(), 37), Disposition::unsupported);
    CHECK(calls == 1);
    check_no_reply(session.retry_pending(), Disposition::pending);
    unsupported = true;
    check_no_reply(session.retry_pending(), Disposition::unsupported);
    CHECK(session.has_pending_request());
    unsupported = false;
    available = true;
    check_error(session.retry_pending(), 0x615);
    CHECK(!session.has_pending_request());
    check_no_reply(session.retry_pending(), Disposition::unsupported);

    armigo::horizon::HandleTable handles(1);
    bool published = false;
    Session delayed(config("unit"), [&](std::uint64_t, const ServiceName&) {
        if (!published) return LookupResult{LookupKind::pending, 0, 0};
        const auto handle = handles.insert(std::make_shared<SyntheticClient>());
        CHECK(handle.has_value());
        return LookupResult{LookupKind::connected, handle.value_or(0), 0};
    });
    register_client(delayed);
    check_no_reply(delayed.dispatch(packet), Disposition::pending);
    CHECK(handles.size() == 0);
    published = true;
    const auto delivered = delayed.retry_pending();
    CHECK(delivered.disposition == Disposition::handled && get(delivered, 16, 4) == 0);
    CHECK(!delayed.has_pending_request() && handles.size() == 1);
    const auto moved = static_cast<armigo::horizon::Handle>(get(delivered, 12, 4));
    CHECK(handles.lookup_as<SyntheticClient>(moved) != nullptr && handles.close(moved));

    Session missing_backend(config("unit"), {});
    register_client(missing_backend);
    check_no_reply(missing_backend.dispatch(packet), Disposition::unsupported);
    for (const auto outcome : {
             LookupResult{LookupKind::connected, 0, 0},
             LookupResult{LookupKind::failed, 0, 0},
             LookupResult{LookupKind::failed, 0, 0xc823},
             LookupResult{LookupKind::pending, 0, 1}}) {
        Session invalid(config("unit"), [outcome](std::uint64_t, const ServiceName&) { return outcome; });
        register_client(invalid);
        check_no_reply(invalid.dispatch(packet), Disposition::unsupported);
        CHECK(!invalid.has_pending_request());
    }
}

void test_packet_validation() {
    Session session(config("unit"), {});
    auto reg = registration();
    for (std::size_t n = 0; n < 20; ++n) {
        const auto out = session.dispatch(std::span<const std::byte>(reg).first(n), 37);
        check_no_reply(out, Disposition::malformed);
        CHECK(out.dispatcher_result == 0x1e23);
    }
    auto malformed = reg;
    put(malformed, 8, 3, 4);
    check_no_reply(session.dispatch(malformed, 37), Disposition::malformed);
    malformed = reg;
    malformed[4] = std::byte{1};
    check_no_reply(session.dispatch(malformed, 37), Disposition::malformed);
    // Firmware masks unused header bits; this bit is outside that mask.
    reg[6] |= std::byte{1};
    CHECK(session.dispatch(reg, 37).disposition == Disposition::handled);
    auto lookup = request(name("unit"));
    for (std::size_t n = 8; n < 16; ++n)
        check_no_reply(session.dispatch(std::span<const std::byte>(lookup).first(n)), Disposition::malformed);
    lookup[7] = std::byte{0x80};
    check_no_reply(session.dispatch(lookup), Disposition::malformed);
    lookup = request(name("unit"));
    lookup[0] = std::byte{0x12};
    check_no_reply(session.dispatch(lookup), Disposition::unsupported);
}
}  // namespace

int main() {
    test_registration_and_connection();
    test_names_and_acl();
    test_pending_and_failures();
    test_packet_validation();
    if (failures) return 1;
    std::puts("service-manager checks passed");
    return 0;
}
