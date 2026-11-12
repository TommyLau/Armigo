// ============================================================================
// Armigo Services - Initial File-System Service Session
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#pragma once

#include <armigo/horizon/ipc.h>
#include <armigo/romfs/storage.h>
#include <armigo/services/addon_content_catalog.h>

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace armigo::services::fs {

// One firmware `fsp-srv` client session. The original SDK connects to
// this port during nnMain startup. This increment serves the CMIF controls the
// generated clients always request and reports every other command as an
// explicit unsupported operation, so the next required command is observable
// instead of silently succeeding.
// A read-only RomFS store the service exposes as the mounted file system. The
// store is read on demand; the service never holds the image's data region.
using RomFsStorage = std::shared_ptr<romfs::Storage>;

// Host-backed file contents are loaded only when the guest opens the file.
// Directory enumeration and mounting retain metadata without reading payloads.
struct SaveNode {
    bool directory = false;
    std::vector<std::byte> bytes;
    std::filesystem::path backing_file;
    std::uint64_t backing_size = 0;
    std::uint64_t size() const noexcept { return backing_file.empty() ? bytes.size() : backing_size; }
};

// The save-data tree mounted by fsp-srv command 51. A session clone and the
// file objects opened from it share one store, so the game's final commit sees
// every write. With a host save directory the tree is read from it at the
// first mount as metadata and each commit replaces it whole. Open files keep
// their node alive across rename, deletion and committed tree replacement.
struct SaveStore {
    std::mutex mutex;
    std::map<std::string, std::shared_ptr<SaveNode>> entries;
    bool loaded = false;
    std::map<std::uint32_t, std::string> files;
    std::map<std::uint32_t, std::string> directories;
    std::uint32_t next_id = 0;
};

// Host directory that keeps the application's save data between runs; empty
// (the default) keeps saves in memory for the run. Set before the application
// mounts save data. Thread-safe.
void set_host_save_directory(std::filesystem::path directory);

class FileSystemServiceSession final : public horizon::ClientSession {
public:
    // `addons` is the application's installed add-on content: command 202
    // (OpenDataStorageByDataId) opens an item's RomFS by its data id.
    explicit FileSystemServiceSession(std::uint64_t process_id,
                                      std::uint16_t pointer_buffer_capacity = 0,
                                      RomFsStorage romfs = {},
                                      std::shared_ptr<const AddOnContentCatalog> addons = {});

    horizon::IpcReply request(const horizon::IpcMessage& message,
                              std::uint64_t sender_pid) override;
    horizon::IpcReply request_with_buffers(const horizon::IpcMessage& message,
                                           std::uint64_t sender_pid,
                                           const horizon::GuestBuffers& buffers) override;

    // The application's own program id. A save-data attribute naming it, like
    // one naming program zero, opens this application's save; the host keeps
    // no other program's saves, so naming any other program finds none.
    // Shared by clones.
    void set_application_program_id(std::uint64_t program_id);

    std::size_t domain_object_count() const;
    // Object 1 is the service root; later objects are mounted file systems.
    static constexpr std::uint32_t root_object_id = 1;
    // Static text for the most recent unsupported request, otherwise null.
    const char* diagnostic() const;
    const std::string& last_path() const;

private:
    // One fs client process's shared service state. Firmware resolves domain
    // object ids across every clone of an fsp-srv session: a client that opens
    // the RomFS storage through one clone issues later reads through another,
    // so the id space and the object tables live behind one lock here instead
    // of in each session.
    struct DomainState {
        std::mutex mutex;
        bool converted = false;
        std::uint32_t next_object_id = 1;
        std::uint32_t object_count = 0;
        // Opened data storages: the application's RomFS or an add-on's.
        std::map<std::uint32_t, RomFsStorage> stores;
        std::vector<std::uint32_t> save_objects;
        std::map<std::uint32_t, std::shared_ptr<SaveNode>> file_objects;
        std::map<std::uint32_t, std::string> directory_objects;
        std::shared_ptr<SaveStore> save = std::make_shared<SaveStore>();
        std::uint64_t application_program_id = 0;
        std::uint32_t released_objects = 0;
        std::string last_path;
        const char* diagnostic = nullptr;
    };

    FileSystemServiceSession(std::uint64_t process_id, std::uint16_t pointer_buffer_capacity,
                             RomFsStorage romfs, std::shared_ptr<const AddOnContentCatalog> addons,
                             std::shared_ptr<DomainState> state);
    horizon::IpcReply open_store(RomFsStorage storage);
    horizon::IpcReply unsupported(const char* message);
    horizon::IpcReply save_command(std::uint32_t command, std::size_t cmif,
                                   const horizon::IpcMessage& message,
                                   const horizon::GuestBuffers* buffers);
    horizon::IpcReply file_command(std::uint32_t id, std::uint32_t command, std::size_t cmif,
                                   const horizon::IpcMessage& message,
                                   const horizon::GuestBuffers* buffers);
    horizon::IpcReply directory_command(std::uint32_t id, std::uint32_t command, std::size_t cmif,
                                        const horizon::IpcMessage& message,
                                        const horizon::GuestBuffers* buffers);
    const std::uint64_t process_id_;
    const std::uint16_t pointer_capacity_;
    const RomFsStorage romfs_;
    const std::shared_ptr<const AddOnContentCatalog> addons_;
    const std::shared_ptr<DomainState> state_;
};

} // namespace armigo::services::fs
