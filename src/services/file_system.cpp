// ============================================================================
// Armigo Services - Initial File-System Service Session
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <armigo/services/file_system.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <new>
#include <system_error>
#include <utility>
#include <vector>

namespace armigo::services::fs {
namespace {
using horizon::IpcDisposition;
using horizon::IpcMessage;
using horizon::IpcReply;

constexpr std::uint32_t sfci = 0x49434653;
constexpr std::uint32_t sfco = 0x4f434653;
constexpr std::uint32_t control_interface = 0xec6be3ff;

std::uint32_t read(const IpcMessage& message, std::size_t offset, unsigned width = 4) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < width; ++i)
        value |= std::uint32_t{std::to_integer<unsigned char>(message[offset + i])} << (8 * i);
    return value;
}

void write(IpcMessage& message, std::size_t offset, std::uint32_t value, unsigned width = 4) {
    for (unsigned i = 0; i < width; ++i)
        message[offset + i] = static_cast<std::byte>((value >> (8 * i)) & 255);
}

void cmif_header(IpcMessage& message, std::size_t offset, std::uint32_t interface_id) {
    write(message, offset, sfco);
    write(message, offset + 12, interface_id);
}

IpcReply control_reply(std::uint32_t value, unsigned width) {
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 44;
    write(reply.bytes, 4, 9);
    cmif_header(reply.bytes, 16, control_interface);
    write(reply.bytes, 32, value, width);
    return reply;
}
std::uint64_t read64(const IpcMessage& message, std::size_t offset) {
    return std::uint64_t{read(message, offset)} |
           (std::uint64_t{read(message, offset + 4)} << 32);
}

// Save files are small; this bound catches a bad size before allocating.
constexpr std::uint64_t save_file_limit = 64ull << 20;
constexpr std::uint64_t save_space_free = 512ull << 20;
constexpr std::uint64_t save_space_total = 512ull << 20;
constexpr std::uint32_t path_not_found = 0x202;
// FS ResultTargetNotFound: the requested save data does not exist.
constexpr std::uint32_t target_not_found = 0x7d402;
constexpr std::uint32_t path_already_exists = 0x402;
constexpr std::size_t directory_entry_size = 0x310;

std::string save_path(const horizon::GuestBuffers& buffers, std::size_t index) {
    if (index >= buffers.size()) return {};
    const auto text = buffers.read_string(index);
    if (!text) return {};
    std::string path = *text;
    for (auto& character : path)
        if (character == '\\') character = '/';
    while (!path.empty() && path.front() == '/') path.erase(path.begin());
    while (!path.empty() && path.back() == '/') path.pop_back();
    return path;
}

bool under_path(const std::string& entry, const std::string& prefix) {
    return entry == prefix ||
           (entry.size() > prefix.size() && entry.compare(0, prefix.size(), prefix) == 0 &&
            entry[prefix.size()] == '/');
}

std::size_t response_offset(std::size_t cmif) noexcept { return cmif >= 32 ? 32 : cmif; }

std::mutex host_save_mutex;
std::filesystem::path host_save_directory;

std::filesystem::path current_host_save_directory() {
    const std::lock_guard lock(host_save_mutex);
    return host_save_directory;
}

// A save path stays inside the save root: every component is a plain name.
bool host_safe(const std::string& path) {
    if (path.empty()) return false;
    for (std::size_t begin = 0; begin <= path.size();) {
        const auto end = std::min(path.find('/', begin), path.size());
        const auto component = path.substr(begin, end - begin);
        if (component.empty() || component == "." || component == "..") return false;
        begin = end + 1;
    }
    return true;
}

// A commit interrupted between its two renames (see commit_saves) leaves no
// directory, the previous tree beside it and, staged beside that, the new one.
// The previous tree is moved aside only after the new one is fully written,
// so the staged tree wins; without it the previous tree comes back. A staged
// tree with no previous one may be partial and is left for the next commit
// to discard.
void recover_interrupted_commit(const std::filesystem::path& root) {
    std::error_code error;
    if (std::filesystem::exists(root, error) || error) return;
    const auto staged = std::filesystem::path(root).concat(".commit");
    const auto previous = std::filesystem::path(root).concat(".previous");
    if (!std::filesystem::is_directory(previous, error)) return;
    if (std::filesystem::is_directory(staged, error)) {
        std::filesystem::rename(staged, root, error);
        if (!error) {
            std::filesystem::remove_all(previous, error);
            return;
        }
    }
    std::filesystem::rename(previous, root, error);
}

// Reads the committed tree from the host directory, if it exists.
void load_saves(SaveStore& store, const std::filesystem::path& root) {
    std::error_code error;
    recover_interrupted_commit(root);
    if (!std::filesystem::is_directory(root, error)) return;
    for (auto it = std::filesystem::recursive_directory_iterator(root, error);
         !error && it != std::filesystem::recursive_directory_iterator(); it.increment(error)) {
        const auto relative = it->path().lexically_relative(root).generic_string();
        if (!host_safe(relative)) continue;
        SaveNode node;
        if (it->is_directory(error)) {
            node.directory = true;
        } else if (it->is_regular_file(error) && it->file_size(error) <= save_file_limit) {
            std::ifstream file(it->path(), std::ios::binary);
            node.bytes.resize(static_cast<std::size_t>(it->file_size(error)));
            file.read(reinterpret_cast<char*>(node.bytes.data()), static_cast<std::streamsize>(node.bytes.size()));
            if (!file) continue;
        } else {
            continue;
        }
        store.entries[relative] = std::move(node);
    }
}

// Writes the whole tree beside the host directory, then swaps it in by
// renames, so the directory always holds one complete commit.
bool commit_saves(const SaveStore& store, const std::filesystem::path& root) {
    std::error_code error;
    const auto staged = std::filesystem::path(root).concat(".commit");
    const auto previous = std::filesystem::path(root).concat(".previous");
    std::filesystem::remove_all(staged, error);
    if (!std::filesystem::create_directories(staged, error) && error) return false;
    for (const auto& [path, node] : store.entries) {
        if (!host_safe(path)) continue;
        const auto target = staged / path;
        if (node.directory) {
            std::filesystem::create_directories(target, error);
            if (error) return false;
            continue;
        }
        std::filesystem::create_directories(target.parent_path(), error);
        std::ofstream file(target, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(node.bytes.data()), static_cast<std::streamsize>(node.bytes.size()));
        if (!file) return false;
    }
    std::filesystem::remove_all(previous, error);
    if (std::filesystem::exists(root, error)) {
        std::filesystem::rename(root, previous, error);
        if (error) return false;
    }
    std::filesystem::rename(staged, root, error);
    if (error) return false;
    std::filesystem::remove_all(previous, error);
    return true;
}

IpcReply save_reply(std::size_t cmif, std::size_t output = 0) {
    const auto response = response_offset(cmif);
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = response + 16 + output;
    write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
    write(reply.bytes, 16, 1);
    cmif_header(reply.bytes, response, 0);
    return reply;
}

IpcReply save_result(std::size_t cmif, std::uint32_t result) {
    IpcReply reply = save_reply(cmif);
    write(reply.bytes, response_offset(cmif) + 8, result);
    return reply;
}

IpcReply save_u32(std::size_t cmif, std::uint32_t value) {
    IpcReply reply = save_reply(cmif, 4);
    write(reply.bytes, response_offset(cmif) + 16, value);
    return reply;
}

IpcReply save_u64(std::size_t cmif, std::uint64_t value) {
    IpcReply reply = save_reply(cmif, 8);
    write(reply.bytes, response_offset(cmif) + 16, static_cast<std::uint32_t>(value));
    write(reply.bytes, response_offset(cmif) + 20, static_cast<std::uint32_t>(value >> 32));
    return reply;
}

IpcReply save_object(std::size_t cmif, std::uint32_t id) {
    IpcReply reply = save_reply(cmif, 12);
    write(reply.bytes, response_offset(cmif) + 16, id);
    return reply;
}
} // namespace

void set_host_save_directory(std::filesystem::path directory) {
    // "saves/" names the same directory as "saves"; the commit's sibling paths
    // (".commit", ".previous") are built from the form without the separator.
    directory = directory.lexically_normal();
    if (!directory.empty() && !directory.has_filename() && directory.has_relative_path())
        directory = directory.parent_path();
    const std::lock_guard lock(host_save_mutex);
    host_save_directory = std::move(directory);
}

FileSystemServiceSession::FileSystemServiceSession(std::uint64_t process_id,
                                                   std::uint16_t pointer_buffer_capacity,
                                                   RomFsStorage romfs,
                                                   std::shared_ptr<const AddOnContentCatalog> addons)
    : FileSystemServiceSession(process_id, pointer_buffer_capacity, std::move(romfs), std::move(addons),
                               std::make_shared<DomainState>()) {}

FileSystemServiceSession::FileSystemServiceSession(std::uint64_t process_id,
                                                   std::uint16_t pointer_buffer_capacity,
                                                   RomFsStorage romfs,
                                                   std::shared_ptr<const AddOnContentCatalog> addons,
                                                   std::shared_ptr<DomainState> state)
    : process_id_(process_id), pointer_capacity_(pointer_buffer_capacity),
      romfs_(std::move(romfs)), addons_(std::move(addons)), state_(std::move(state)) {}

void FileSystemServiceSession::set_application_program_id(std::uint64_t program_id) {
    const std::lock_guard lock(state_->mutex);
    state_->application_program_id = program_id;
}

IpcReply FileSystemServiceSession::unsupported(const char* message) {
    state_->diagnostic = message;
    return {};
}

// Creates a data storage domain object over `storage`. The reply carries the
// new object id and the size of the opened store: nn::fs::RomFsFileSystem::
// Initialize validates every region of the superblock against the size of the
// storage it reads from, so a store that reports no size fails the mount.
IpcReply FileSystemServiceSession::open_store(RomFsStorage storage) {
    const auto store_size = storage ? storage->size() : 0;
    const auto id = state_->next_object_id + 1;
    IpcReply reply;
    reply.disposition = IpcDisposition::handled;
    reply.size = 68;
    write(reply.bytes, 4, 15);
    write(reply.bytes, 16, 1);
    cmif_header(reply.bytes, 32, 0);
    write(reply.bytes, 48, id);
    write(reply.bytes, 56, static_cast<std::uint32_t>(store_size));
    write(reply.bytes, 60, static_cast<std::uint32_t>(store_size >> 32));
    state_->stores[id] = std::move(storage);
    ++state_->next_object_id;
    ++state_->object_count;
    return reply;
}


IpcReply FileSystemServiceSession::save_command(std::uint32_t command, std::size_t cmif,
                                                const IpcMessage& message,
                                                const horizon::GuestBuffers* buffers) {
    const auto path = buffers != nullptr ? save_path(*buffers, 0) : std::string{};
    const auto second = buffers != nullptr ? save_path(*buffers, 1) : std::string{};
    const std::lock_guard lock(state_->save->mutex);
    auto& entries = state_->save->entries;
    switch (command) {
    case 0: {  // CreateFile(path, option, size)
        if (path.empty()) return unsupported("Save file path is empty");
        const auto option = read(message, cmif + 16);
        const auto size = std::min<std::uint64_t>(read64(message, cmif + 24), save_file_limit);
        const auto found = entries.find(path);
        if (found != entries.end()) {
            if (found->second.directory || option == 0)
                return save_result(cmif, found->second.directory ? path_not_found : path_already_exists);
            return save_reply(cmif);
        }
        SaveNode node;
        try {
            node.bytes.assign(static_cast<std::size_t>(size), std::byte{});
        } catch (const std::bad_alloc&) {
            return unsupported("Save file allocation failed");
        }
        entries[path] = std::move(node);
        return save_reply(cmif);
    }
    case 1: {  // DeleteFile
        const auto found = entries.find(path);
        if (found == entries.end() || found->second.directory)
            return save_result(cmif, path_not_found);
        entries.erase(found);
        return save_reply(cmif);
    }
    case 2: {  // CreateDirectory
        if (path.empty()) return unsupported("Save directory path is empty");
        auto [entry, inserted] = entries.try_emplace(path);
        entry->second.directory = true;
        return save_reply(cmif);
    }
    case 3: {  // DeleteDirectory
        const auto found = entries.find(path);
        if (found == entries.end()) return save_result(cmif, path_not_found);
        entries.erase(found);
        return save_reply(cmif);
    }
    case 4: {  // DeleteDirectoryRecursively
        if (path.empty()) return unsupported("Save directory path is empty");
        for (auto it = entries.begin(); it != entries.end();) {
            if (under_path(it->first, path)) it = entries.erase(it);
            else ++it;
        }
        return save_reply(cmif);
    }
    case 5:  // RenameFile
    case 6: {  // RenameDirectory
        const auto found = entries.find(path);
        if (found == entries.end() || second.empty())
            return save_result(cmif, path_not_found);
        SaveNode node = std::move(found->second);
        entries.erase(found);
        entries[second] = std::move(node);
        return save_reply(cmif);
    }
    case 7: {  // GetEntryType: 0 directory, 1 file
        const auto found = entries.find(path);
        if (found == entries.end()) return save_result(cmif, path_not_found);
        return save_u32(cmif, found->second.directory ? 0 : 1);
    }
    case 8: {  // OpenFile
        const auto found = entries.find(path);
        if (found == entries.end() || found->second.directory)
            return save_result(cmif, path_not_found);
        const auto id = ++state_->next_object_id;
        state_->file_objects[id] = path;
        ++state_->object_count;
        return save_object(cmif, id);
    }
    case 9: {  // OpenDirectory
        const auto found = entries.find(path);
        if (!path.empty() &&
            (found == entries.end() || !found->second.directory))
            return save_result(cmif, path_not_found);
        const auto id = ++state_->next_object_id;
        state_->directory_objects[id] = path;
        ++state_->object_count;
        return save_object(cmif, id);
    }
    case 10:  // Commit
        if (const auto root = current_host_save_directory(); !root.empty() &&
            !commit_saves(*state_->save, root))
            return unsupported("Save data commit to the host directory failed");
        return save_reply(cmif);
    case 11:  // GetFreeSpaceSize
        return save_u64(cmif, save_space_free);
    case 12:  // GetTotalSpaceSize
        return save_u64(cmif, save_space_total);
    case 13: {  // CleanDirectoryRecursively
        if (path.empty()) return unsupported("Save directory path is empty");
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->first != path && under_path(it->first, path)) it = entries.erase(it);
            else ++it;
        }
        return save_reply(cmif);
    }
    case 14:  // GetFileTimeStampRaw: zero timestamps
        return save_reply(cmif, 0x20);
    case 16: {  // GetFileSystemAttribute
        IpcReply reply = save_reply(cmif, 0xc0);
        const auto base = response_offset(cmif) + 16;
        for (std::size_t index = 0; index != 13; ++index)
            reply.bytes[base + index] = std::byte{1};
        const std::uint32_t values[13] = {0x80, 0x80, 0x300, 0x300, 0x300, 0x300, 0x300,
                                          0x300, 0x300, 0x80, 0x80, 0x300, 0x300};
        for (std::size_t index = 0; index != 13; ++index)
            write(reply.bytes, base + 0x2c + index * 4, values[index]);
        return reply;
    }
    default:
        return unsupported("Save-data command is not implemented");
    }
}

IpcReply FileSystemServiceSession::file_command(std::uint32_t id, std::uint32_t command,
                                                std::size_t cmif, const IpcMessage& message,
                                                const horizon::GuestBuffers* buffers) {
    const auto object = state_->file_objects.find(id);
    if (object == state_->file_objects.end()) return unsupported("File object is not open");
    const std::lock_guard lock(state_->save->mutex);
    const auto found = state_->save->entries.find(object->second);
    if (found == state_->save->entries.end() || found->second.directory)
        return save_result(cmif, path_not_found);
    auto& bytes = found->second.bytes;
    switch (command) {
    case 0: {  // Read(option, offset, size) into the caller's buffer
        const auto offset = std::min<std::uint64_t>(read64(message, cmif + 24), bytes.size());
        const auto requested = read64(message, cmif + 32);
        if (buffers == nullptr || buffers->empty())
            return unsupported("File read needs a guest buffer");
        const auto available = bytes.size() - static_cast<std::size_t>(offset);
        const auto transferred = std::min<std::uint64_t>(
            std::min(requested, static_cast<std::uint64_t>(available)),
            static_cast<std::uint64_t>((*buffers)[0].size));
        if (!buffers->write(0, std::span<const std::byte>(
                                   bytes.data() + offset, static_cast<std::size_t>(transferred))))
            return unsupported("File read buffer is not writable");
        return save_u64(cmif, transferred);
    }
    case 1: {  // Write(option, offset, size) from the caller's buffer
        const auto offset = read64(message, cmif + 24);
        const auto requested = read64(message, cmif + 32);
        if (buffers == nullptr || buffers->empty())
            return unsupported("File write needs a guest buffer");
        const auto written = std::min<std::uint64_t>(requested, (*buffers)[0].size);
        if (offset > save_file_limit || written > save_file_limit - offset)
            return unsupported("Save file write exceeds the store limit");
        const auto end = static_cast<std::size_t>(offset + written);
        try {
            if (end > bytes.size()) bytes.resize(end);
        } catch (const std::bad_alloc&) {
            return unsupported("Save file allocation failed");
        }
        if (!buffers->read(0, std::span<std::byte>(bytes.data() + offset,
                                                   static_cast<std::size_t>(written))))
            return unsupported("File write buffer is not readable");
        return save_reply(cmif);
    }
    case 2:  // Flush
        return save_reply(cmif);
    case 3: {  // SetSize(size)
        const auto size = std::min<std::uint64_t>(read64(message, cmif + 16), save_file_limit);
        try {
            bytes.resize(static_cast<std::size_t>(size));
        } catch (const std::bad_alloc&) {
            return unsupported("Save file allocation failed");
        }
        return save_reply(cmif);
    }
    case 4:  // GetSize
        return save_u64(cmif, bytes.size());
    case 5:  // OperateRange
        return unsupported("File operate-range command is not implemented");
    default:
        return unsupported("File command is not implemented");
    }
}

IpcReply FileSystemServiceSession::directory_command(std::uint32_t id, std::uint32_t command,
                                                     std::size_t cmif, const IpcMessage& message,
                                                     const horizon::GuestBuffers* buffers) {
    (void)message;
    const auto object = state_->directory_objects.find(id);
    if (object == state_->directory_objects.end()) return unsupported("Directory object is not open");
    const auto path = object->second;
    const std::lock_guard lock(state_->save->mutex);
    std::vector<std::pair<std::string, const SaveNode*>> children;
    for (const auto& [name, node] : state_->save->entries) {
        if (name.empty() || !under_path(name, path) || name == path) continue;
        const auto remainder = path.empty() ? name : name.substr(path.size() + 1);
        if (remainder.find('/') != std::string::npos) continue;
        children.emplace_back(remainder, &node);
    }
    if (command == 1) return save_u64(cmif, children.size());
    if (command != 0) return unsupported("Directory command is not implemented");
    if (buffers == nullptr || buffers->empty())
        return unsupported("Directory read needs a guest buffer");
    const auto capacity = (*buffers)[0].size / directory_entry_size;
    const auto count = std::min<std::size_t>(capacity, children.size());
    std::vector<std::byte> raw(count * directory_entry_size);
    for (std::size_t index = 0; index != count; ++index) {
        const auto& name = children[index].first;
        const auto copy = std::min<std::size_t>(name.size(), 0x300);
        std::memcpy(raw.data() + index * directory_entry_size, name.data(), copy);
        raw[index * directory_entry_size + 0x304] =
            static_cast<std::byte>(children[index].second->directory ? 0 : 1);
        const auto size = static_cast<std::uint64_t>(children[index].second->bytes.size());
        for (unsigned byte = 0; byte != 8; ++byte)
            raw[index * directory_entry_size + 0x308 + byte] =
                static_cast<std::byte>((size >> (byte * 8)) & 255);
    }
    if (!buffers->write(0, raw))
        return unsupported("Directory read buffer is not writable");
    return save_u64(cmif, count);
}

IpcReply FileSystemServiceSession::request_with_buffers(const IpcMessage& message,
                                                       std::uint64_t sender_pid,
                                                       const horizon::GuestBuffers& buffers) {
    // Descriptor-free control and inline requests use the ordinary parser.
    if (buffers.empty()) return request(message, sender_pid);
    const std::lock_guard lock(state_->mutex);
    state_->diagnostic = nullptr;
    if (sender_pid != process_id_)
        return unsupported("File-system session belongs to a different process");
    const auto type = read(message, 0, 2);
    if ((type != 4 && type != 6) || !state_->converted)
        return unsupported("File-system buffered request shape is unsupported");
    // A one-pointer request keeps its domain header at byte16 and its CMIF
    // header at byte32; wider descriptor areas use the PID variant with the
    // header at byte32 and the CMIF header at byte48.
    std::size_t domain = 0;
    if (read(message, 32) == sfci) domain = 16;
    else if (read(message, 48) == sfci) domain = 32;
    else return unsupported("File-system buffered request shape is unsupported");
    const auto header = read(message, domain);
    const auto size = header >> 16;
    const auto id = read(message, domain + 4);
    const auto cmif = domain + 16;
    const auto command = read(message, cmif + 8);
    if ((header & 0xffff) != 1 || size < 16 || id < 1)
        return unsupported("File-system object request shape is unsupported");
    if (buffers.empty()) return unsupported("File-system object request needs a guest buffer");
    if (const auto opened = state_->stores.find(id); opened != state_->stores.end()) {
        // A data storage object: command 0 copies {offset, size} into the
        // guest buffer. The store is read on demand in bounded chunks so a large
        // image never has to live in memory.
        const auto& store = opened->second;
        if (!store || store->size() == 0)
            return unsupported("No RomFS image is configured for the file-system store");
        if (command != 0) return unsupported("File-system storage command is not implemented");
        const auto offset = read64(message, cmif + 16);
        const auto requested = read(message, cmif + 24);
        if (offset > store->size())
            return unsupported("RomFS read offset is outside the configured image");
        const auto available = static_cast<std::uint64_t>(store->size() - offset);
        const auto transferred =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(requested, available));
        if (transferred > buffers[0].size)
            return unsupported("RomFS read does not fit the guest buffer");
        constexpr std::size_t chunk_size = 1u << 20;
        std::vector<std::byte> chunk(std::min<std::size_t>(chunk_size, transferred));
        std::uint32_t done = 0;
        while (done < transferred) {
            const auto wanted = std::min<std::size_t>(chunk.size(), transferred - done);
            const auto read = store->read(offset + done, std::span(chunk.data(), wanted));
            if (!read) return unsupported("RomFS storage read failed");
            if (*read == 0) break;
            if (!buffers.write_at(0, done, std::span<const std::byte>(chunk.data(), *read)))
                return unsupported("RomFS read buffer is not writable");
            done += static_cast<std::uint32_t>(*read);
            if (*read < wanted) break;
        }
        IpcReply reply;
        reply.disposition = IpcDisposition::handled;
        reply.size = cmif + 24;
        write(reply.bytes, 4, static_cast<std::uint32_t>((reply.size - 8) / 4));
        if (cmif >= 32) write(reply.bytes, 16, 1);
        cmif_header(reply.bytes, response_offset(cmif), 0);
        write(reply.bytes, response_offset(cmif) + 16, done);
        return reply;
    }
    if (std::find(state_->save_objects.begin(), state_->save_objects.end(), id) !=
        state_->save_objects.end())
        return save_command(command, cmif, message, &buffers);
    if (state_->file_objects.find(id) != state_->file_objects.end())
        return file_command(id, command, cmif, message, &buffers);
    if (state_->directory_objects.find(id) != state_->directory_objects.end())
        return directory_command(id, command, cmif, message, &buffers);
    return unsupported("File-system object command is not implemented");
}

IpcReply FileSystemServiceSession::request(const IpcMessage& message, std::uint64_t sender_pid) {
    const std::lock_guard lock(state_->mutex);
    state_->diagnostic = nullptr;
    if (sender_pid != process_id_)
        return unsupported("File-system session belongs to a different process");
    const auto type = read(message, 0, 2);
    if (type == 5 || type == 7) {
        // CMIF control requests: pointer-buffer size and domain conversion. The
        // generated clients always issue these before their first command.
        if (read(message, 16) != sfci) return unsupported("File-system control magic is unsupported");
        const auto command = read(message, 24);
        if (command == 3) return control_reply(pointer_capacity_, 2);
        if (command == 0) {
            state_->converted = true;
            return control_reply(1, 4);
        }
        if (command == 2) {
            // Clone: hand the caller a session object on the same port that
            // shares this session's domain state. Firmware resolves domain ids
            // across clones, so the clone must not snapshot the tables.
            IpcReply reply = control_reply(0, 4);
            reply.transferred[0].reset(
                new FileSystemServiceSession(process_id_, pointer_capacity_, romfs_, addons_, state_));
            reply.transferred_count = 1;
            return reply;
        }
        return unsupported("File-system control command is unsupported");
    }
    if (type == 4 || type == 6) {
        // A domain message without a CMIF header releases the objects listed in
        // its domain header. The SDK sends it when the last reference to a
        // domain object goes away, for example right after a store has been
        // handed to RomFS initialization, and its reply is a plain success with
        // no payload.
        if (state_->converted) {
            // The domain header carries the object count in the low halfword and
            // the size of the header block in the high halfword. A zero size
            // means no CMIF request follows, which is the release shape.
            const auto header = read(message, 16);
            const auto objects = header & 0xffff;
            if (objects >= 1 && objects <= 8 && (header >> 16) == 0) {
                ++state_->released_objects;
                IpcReply reply;
                reply.disposition = IpcDisposition::handled;
                reply.size = 32;
                write(reply.bytes, 4, 6);
                cmif_header(reply.bytes, 16, 0);
                return reply;
            }
        }
        // Two domain layouts reach this port: the standard header at byte16 with
        // the CMIF header at 32, and the PID-descriptor variant with the domain
        // header at 32 and the CMIF header at 48.
        std::size_t domain = 0;
        if (state_->converted && read(message, 32) == sfci) domain = 16;
        else if (state_->converted && read(message, 48) == sfci) domain = 32;
        else return unsupported("File-system domain request shape is unsupported");
        const auto header_offset = domain;
        const auto cmif = domain + 16;
        const auto header = read(message, header_offset);
        const auto size = header >> 16;
        const auto id = read(message, header_offset + 4);
        const auto command = read(message, cmif + 8);
        if ((header & 0xffff) != 1 || size < 16 || id < 1)
            return unsupported("File-system domain header is unsupported");
        if (const auto opened = state_->stores.find(id); opened != state_->stores.end()) {
            // A data storage object. nn::fs::RomFsFileSystem::Initialize
            // reads its 80-byte superblock and then asks the same storage for its
            // size, so the transfer length is the whole configured image.
            const auto& store = opened->second;
            if (!store || store->size() == 0)
                return unsupported("No RomFS image is configured for the file-system store");
            if (command != 4)
                return unsupported("File-system storage command is not implemented");
            IpcReply reply;
            reply.disposition = IpcDisposition::handled;
            reply.size = 56;
            write(reply.bytes, 4, 12);
            cmif_header(reply.bytes, 32, 0);
            write(reply.bytes, 48, static_cast<std::uint32_t>(store->size()));
            write(reply.bytes, 52, static_cast<std::uint32_t>(store->size() >> 32));
            return reply;
        }
        if (std::find(state_->save_objects.begin(), state_->save_objects.end(), id) !=
            state_->save_objects.end())
            return save_command(command, cmif, message, nullptr);
        if (state_->file_objects.find(id) != state_->file_objects.end())
            return file_command(id, command, cmif, message, nullptr);
        if (state_->directory_objects.find(id) != state_->directory_objects.end())
            return directory_command(id, command, cmif, message, nullptr);
        if (command == 51) {
            // OpenSaveDataFileSystem(u8 space id, attribute): attribute +0 is
            // the owning program, zero or its own id for this application. FS
            // looks a save up before it checks the caller's access to it, and
            // the host keeps no other program's saves: such a save is not found.
            if (const auto program = read64(message, cmif + 24);
                program != 0 && program != state_->application_program_id)
                return save_result(cmif, target_not_found);
            // One save-data file system backed by the session's save tree. The
            // reply carries its domain object id.
            const auto save_id = ++state_->next_object_id;
            state_->save_objects.push_back(save_id);
            {
                const std::lock_guard save_lock(state_->save->mutex);
                if (!state_->save->loaded) {
                    state_->save->loaded = true;
                    if (const auto root = current_host_save_directory(); !root.empty())
                        load_saves(*state_->save, root);
                }
            }
            ++state_->object_count;
            IpcReply reply;
            reply.disposition = IpcDisposition::handled;
            reply.size = 52;
            write(reply.bytes, 4, (52-8)/4);
            write(reply.bytes, 16, 1);
            cmif_header(reply.bytes, 32, 0);
            write(reply.bytes, 48, save_id);
            return reply;
        }
        if (command == 1) {
            // SetCurrentProcess: the transport already carries the trusted PID.
            if (domain != 32 || size < 24 || read(message, 4) != 0x8000000e)
                return unsupported("File-system SetCurrentProcess needs the PID descriptor");
            IpcReply reply;
            reply.disposition = IpcDisposition::handled;
            reply.size = 48;
            write(reply.bytes, 4, 10);
            cmif_header(reply.bytes, 32, 0);
            return reply;
        }
        if (command == 200 || command == 512) {
            // Open (200) and mount (512) both create one owned file-system
            // domain object over the application's RomFS.
            if (size < 16) return unsupported("File-system open payload is unsupported");
            return open_store(romfs_);
        }
        if (command == 202) {
            // OpenDataStorageByDataId(u8 storage id, u64 data id): the RomFS of
            // installed add-on content. The SDK's add-on mount passes storage id
            // 0; the data id alone selects the item, which aoc:u listed.
            if (size < 32) return unsupported("File-system data storage payload is unsupported");
            const auto data_id = read64(message, cmif + 24);
            const auto* item = addons_ ? addons_->find(data_id) : nullptr;
            if (!item) return unsupported("Add-on content data storage is not installed");
            return open_store(item->storage);
        }
        if (command == 1005) {
            // GetGlobalAccessLogMode: one u32 output. Access logging is disabled.
            IpcReply reply;
            reply.disposition = IpcDisposition::handled;
            reply.size = 52;
            write(reply.bytes, 4, 11);
            cmif_header(reply.bytes, 32, 0);
            write(reply.bytes, 48, 0);
            return reply;
        }
        return unsupported("File-system command is not implemented");
    }
    return unsupported("File-system request shape is unsupported");
}

std::size_t FileSystemServiceSession::domain_object_count() const {
    const std::lock_guard lock(state_->mutex);
    return state_->converted ? state_->object_count + 1 : 0;
}

const std::string& FileSystemServiceSession::last_path() const {
    const std::lock_guard lock(state_->mutex);
    return state_->last_path;
}

const char* FileSystemServiceSession::diagnostic() const {
    const std::lock_guard lock(state_->mutex);
    return state_->diagnostic;
}

} // namespace armigo::services::fs
