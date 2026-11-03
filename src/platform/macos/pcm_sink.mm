// ============================================================================
// Armigo Platform - Apple Platform Integration
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Host-owned PCM16 output: a bounded single-producer / single-consumer ring of
// 48000 Hz stereo frames drained by the platform output AudioUnit, or by
// nothing at all in the explicit muted mode. The producer (the audio renderer
// thread) converts its mono, stereo or 5.1 PCM16 into stereo while copying
// into the ring; the realtime render callback drains the ring into the device
// buffer - a plain memcpy at unity gain, otherwise an exact saturating multiply
// by the current per-sink gain - and zero-fills whatever is missing. Because
// the gain is applied when frames are consumed, not when they are queued,
// frames already in the ring when the volume changes come out at the new gain,
// and the channel conversion stays a producer-side concern (the callback always
// scales fully converted stereo). Nothing in this file touches guest state,
// and the realtime callback never allocates, locks, logs or calls out.
//
// Thread ownership (annotated per field below):
//   producer = the caller of pcm_sink_write / pcm_sink_set_active /
//              pcm_sink_set_volume / pcm_sink_close, externally serialized by
//              contract (a single producer and no concurrent API use);
//   callback = the AudioUnit render thread (render_callback), a concurrent
//              consumer that is only reachable while the unit is started and
//              only reads the ring while `active` is set;
//   resume   = pcm_sink_resume_devices from any thread (after an audio session
//              interruption): restarts the units of active sinks; units_lock
//              serializes it with every unit start, stop and disposal, and it
//              touches nothing else in a sink.
//
// Lifecycle invariants:
//   1. The sink pointer passed to the AudioUnit as the render context stays
//      valid, and the ring allocation stays in place, from
//      AudioComponentInstanceNew until after AudioComponentInstanceDispose has
//      returned and the in-flight callback count is zero.
//   2. Stop/restart never resets ring indices while a callback may be inside
//      the ring: the producer first clears `active`, then stops the unit, then
//      waits for `in_flight` to drop to zero (store-active / load-in_flight
//      versus increment-in_flight / load-active, all sequentially consistent,
//      so one side always observes the other), and only then rewrites the
//      indices.
//   3. Close additionally uninitializes and disposes the unit (the AudioUnit
//      shutdown contract guarantees no further callbacks once Dispose has
//      succeeded) and waits for in_flight == 0 again before freeing. Failed
//      disposal retains storage rather than risking a late callback using
//      freed memory. The in-flight count alone is never relied on to stop new
//      callbacks.
//   4. Device failures are explicit: open returns null and set_active returns
//      -1 after an English diagnostic; an audible sink never silently degrades
//      into a muted one.
//
// Counters (unsigned 64-bit, per sink; arithmetic wraps modulo 2^64):
//   submitted_frames = frames accepted by pcm_sink_write (valid input, written
//                      into the ring or, when muted, consumed immediately);
//   consumed_frames  = frames the consumer actually took out of the ring
//                      (inserted silence is never counted; muted sinks count
//                      accepted frames here as well);
//   dropped_frames   = frames refused because the ring was full plus frames
//                      still queued when the sink was stopped;
//   underrun_frames  = stereo frames the device asked for while active that
//                      had to be zero-filled.
// The gain never touches this accounting: frames the consumer takes out of the
// ring are consumed regardless of gain (a zero gain renders silence but is not
// an underrun), and only genuinely missing frames are underruns. Stop/restart
// retains the selected gain; it starts at 1.0. Writes while stopped are
// rejected (return 0) and touch no counter.
//
// Test seam: ARMIGO_PCM_SINK_MOCK_AUDIO_UNIT replaces the seven AudioToolbox
// calls with functions a test defines (declared below), so tests can drive the
// render callback and the lifecycle with no audio device.

#include <armigo/common/diagnostics.h>
#include <armigo/platform/pcm_sink.h>

#include <AudioToolbox/AudioToolbox.h>
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#import <AVFAudio/AVFAudio.h>
#endif

#include <pthread.h>
#include <sched.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

#define PCM_LOG(...)                                          \
    do {                                                      \
        armigo::diag::error("[audio output] " __VA_ARGS__);   \
        armigo::diag::newline();                              \
    } while (0)

namespace armigo::platform {
namespace {

constexpr std::uint32_t pcm_rate = 48000;
constexpr std::uint32_t pcm_out_channels = 2;
constexpr std::uint32_t pcm_frame_bytes = pcm_out_channels * sizeof(std::int16_t);
// Ring capacity in stereo frames: a power of two so indices wrap with a mask.
// 32768 frames is ~683 ms at 48 kHz (128 KiB); the producer's own pacing, not
// this size, sets the latency.
constexpr std::uint32_t pcm_ring_frames = 32768;
constexpr std::uint32_t pcm_ring_mask = pcm_ring_frames - 1;
static_assert((pcm_ring_frames & pcm_ring_mask) == 0, "ring capacity must be a power of two");
static_assert(pcm_ring_frames < (1u << 30), "index difference arithmetic assumes capacity < 2^30");
static_assert(std::atomic<std::uint64_t>::is_always_lock_free &&
                  std::atomic<std::uint32_t>::is_always_lock_free &&
                  std::atomic<int>::is_always_lock_free,
              "callback counters must be lock-free");
static_assert(sizeof(float) == sizeof(std::uint32_t),
              "the gain is stored as its raw 32-bit object representation");

}  // namespace

struct PcmSink {
    // Immutable after open.
    bool audible = false;            // AudioUnit backed, or the explicit muted sink (no device, no ring use).
    std::int16_t* ring = nullptr;    // pcm_ring_frames interleaved stereo frames; owned by the sink.
    AudioUnit unit = nullptr;        // Null for muted sinks.
    // Producer-owned, read by the callback.
    std::atomic<std::uint32_t> tail{0};  // Next frame slot to write; advances monotonically, wraps modulo 2^32.
    std::atomic<int> active{0};          // 1 while started; the callback consumes only when it observes 1.
    // Output gain, producer-owned (pcm_sink_set_volume) and read by the
    // callback, stored as the raw 32-bit representation of the float so the
    // atomic is a plain lock-free integer.
    std::atomic<std::uint32_t> gain_bits{0};
    // Callback-owned, read by the producer.
    std::atomic<std::uint32_t> head{0};  // Next frame slot to read; the producer rewrites it only while quiescent.
    std::atomic<int> in_flight{0};       // Number of callbacks currently inside the consume path (0 or 1).
    // Counters: submitted/dropped are written by the producer only,
    // consumed/underrun by the callback only; all are atomic because
    // pcm_sink_get_counters reads them from the producer side.
    std::atomic<std::uint64_t> submitted{0}, consumed{0}, dropped{0}, underrun{0};
    // Producer-owned explicit error state: last failing AudioToolbox status, 0 when healthy.
    OSStatus last_error = 0;
    PcmSink* next_unit = nullptr;  // Audible sinks with a live unit, under units_lock.
};

namespace {

// Audible sinks with a live unit, for pcm_sink_resume_devices. The lock also
// serializes every start, stop and disposal of those units against a resume;
// the realtime callback never takes it.
pthread_mutex_t units_lock = PTHREAD_MUTEX_INITIALIZER;
PcmSink* units = nullptr;

}  // namespace

// AudioToolbox seam. Real builds call the framework; mock builds get these
// symbols from the test, which may fail any step on demand and drive the
// render callback itself.
#ifdef ARMIGO_PCM_SINK_MOCK_AUDIO_UNIT
extern "C" {
OSStatus armigo_pcm_mock_new(const AudioComponentDescription* desc, AudioUnit* out);
OSStatus armigo_pcm_mock_set_property(AudioUnit unit, AudioUnitPropertyID id, AudioUnitScope scope,
                                      AudioUnitElement element, const void* data, UInt32 size);
OSStatus armigo_pcm_mock_initialize(AudioUnit unit);
OSStatus armigo_pcm_mock_uninitialize(AudioUnit unit);
OSStatus armigo_pcm_mock_start(AudioUnit unit);
OSStatus armigo_pcm_mock_stop(AudioUnit unit);
OSStatus armigo_pcm_mock_dispose(AudioUnit unit);
}
#define AU_NEW armigo_pcm_mock_new
#define AU_SET_PROPERTY armigo_pcm_mock_set_property
#define AU_INITIALIZE armigo_pcm_mock_initialize
#define AU_UNINITIALIZE armigo_pcm_mock_uninitialize
#define AU_START armigo_pcm_mock_start
#define AU_STOP armigo_pcm_mock_stop
#define AU_DISPOSE armigo_pcm_mock_dispose
#else
namespace {
OSStatus au_real_new(const AudioComponentDescription* desc, AudioUnit* out) {
    AudioComponent component = AudioComponentFindNext(nullptr, desc);
    if (!component) {
        PCM_LOG("no default output AudioUnit component on this system");
        return kAudioUnitErr_InvalidElement;
    }
    return AudioComponentInstanceNew(component, out);
}
}  // namespace
#define AU_NEW au_real_new
#define AU_SET_PROPERTY AudioUnitSetProperty
#define AU_INITIALIZE AudioUnitInitialize
#define AU_UNINITIALIZE AudioUnitUninitialize
#define AU_START AudioOutputUnitStart
#define AU_STOP AudioOutputUnitStop
#define AU_DISPOSE AudioComponentInstanceDispose
#endif

namespace {

// Gain plumbing (see the ownership notes above). The gain is a per-sink
// scalar in [0, 128].
void store_gain(PcmSink* sink, float gain) {
    std::uint32_t bits;
    std::memcpy(&bits, &gain, sizeof bits);
    sink->gain_bits.store(bits, std::memory_order_relaxed);
}

// Scale one int16 sample by a finite gain. The product is computed in double
// and is always exact - an int16 carries at most 15 significant bits and a
// float at most 24, so the product needs at most 39 of double's 53 significand
// bits. The exact product is saturated to the int16 range and converted with a
// cast, which truncates toward zero: 32767 * 0.5 = 16383.5 -> 16383,
// -1 * 0.5 = -0.5 -> 0. After saturation the value lies in [-32768, 32767].
inline std::int16_t gain16(std::int16_t value, float gain) {
    const double scaled = static_cast<double>(value) * static_cast<double>(gain);
    if (scaled > 32767.0) return 32767;
    if (scaled < -32768.0) return -32768;
    return static_cast<std::int16_t>(scaled);
}

// Realtime consumer on the AudioUnit render thread: no allocation, locks,
// logging or calls outside this file. Fills exactly one interleaved stereo
// int16 buffer. Takes one lock-free gain snapshot per invocation and scales the
// consumed frames with it after the copy out of the ring, so every frame in one
// buffer is scaled by a single gain value and already queued frames reflect a
// new gain. Unity gain keeps the plain memcpy path and its bit-exact output.
OSStatus render_callback(void* context, AudioUnitRenderActionFlags* flags, const AudioTimeStamp*,
                         UInt32, UInt32 frame_count, AudioBufferList* buffers) {
    auto* sink = static_cast<PcmSink*>(context);
    if (!buffers || buffers->mNumberBuffers < 1 || !buffers->mBuffers[0].mData) return noErr;
    auto* destination = static_cast<std::int16_t*>(buffers->mBuffers[0].mData);
    std::uint32_t want = buffers->mBuffers[0].mDataByteSize / pcm_frame_bytes;
    if (want > frame_count) want = frame_count;
    std::uint32_t got = 0;

    sink->in_flight.fetch_add(1, std::memory_order_seq_cst);
    if (sink->active.load(std::memory_order_seq_cst)) {
        const std::uint32_t head = sink->head.load(std::memory_order_relaxed);
        const std::uint32_t tail = sink->tail.load(std::memory_order_acquire);
        const std::uint32_t gain_bits = sink->gain_bits.load(std::memory_order_relaxed);
        float gain;
        std::memcpy(&gain, &gain_bits, sizeof gain);
        const std::uint32_t available = tail - head;
        got = available < want ? available : want;
        if (got) {
            const std::uint32_t position = head & pcm_ring_mask;
            std::uint32_t first = pcm_ring_frames - position;
            if (first > got) first = got;
            std::memcpy(destination, sink->ring + std::size_t{position} * pcm_out_channels,
                        std::size_t{first} * pcm_frame_bytes);
            if (got > first)
                std::memcpy(destination + std::size_t{first} * pcm_out_channels, sink->ring,
                            std::size_t{got - first} * pcm_frame_bytes);
            if (gain != 1.0f) {
                // Non-unity gain: scale the copied stereo frames in place. The
                // gain never changes consumption: `got` frames are consumed.
                for (std::uint32_t index = 0; index < got * pcm_out_channels; ++index)
                    destination[index] = gain16(destination[index], gain);
            }
            sink->head.store(head + got, std::memory_order_release);
            sink->consumed.fetch_add(got, std::memory_order_relaxed);
        }
        if (want > got) sink->underrun.fetch_add(want - got, std::memory_order_relaxed);
    }
    sink->in_flight.fetch_sub(1, std::memory_order_seq_cst);

    // Zero-fill the whole remainder of the buffer as handed over, not just `want` frames.
    const std::size_t filled = std::size_t{got} * pcm_frame_bytes;
    if (buffers->mBuffers[0].mDataByteSize > filled)
        std::memset(reinterpret_cast<std::uint8_t*>(destination) + filled, 0,
                    buffers->mBuffers[0].mDataByteSize - filled);
    if (got == 0 && flags) *flags |= kAudioUnitRenderAction_OutputIsSilence;
    return noErr;
}

// Producer side: wait until no callback is inside the consume path. Only
// meaningful after `active` has been cleared (invariant 2); bounded by one
// callback duration.
void quiesce(PcmSink* sink) {
    while (sink->in_flight.load(std::memory_order_seq_cst) != 0) sched_yield();
}

// Source channel mapping into the ring (producer side).
//   1 channel : dual mono, L = R = sample.
//   2 channels: copied unchanged.
//   6 channels: front L, front R, centre, LFE, rear L, rear R. ITU-R BS.775
//               Lo/Ro in Q15: L = 0.707*FL + 0.5*C + 0.5*RL, R symmetric, the
//               standard (FL + 0.707*C + 0.707*RL) scaled by 0.707 for 3 dB of
//               headroom. LFE is excluded (a plain Lo/Ro downmix does not
//               carry it). Results beyond int16 are saturated.
inline std::int16_t saturate16(std::int32_t value) {
    return static_cast<std::int16_t>(value > 32767 ? 32767 : value < -32768 ? -32768 : value);
}

void store_frames(std::int16_t* ring, std::uint32_t tail, const std::int16_t* source,
                  std::uint32_t count, unsigned channels) {
    if (channels == pcm_out_channels) {
        const std::uint32_t position = tail & pcm_ring_mask;
        std::uint32_t first = pcm_ring_frames - position;
        if (first > count) first = count;
        std::memcpy(ring + std::size_t{position} * pcm_out_channels, source,
                    std::size_t{first} * pcm_frame_bytes);
        if (count > first)
            std::memcpy(ring, source + std::size_t{first} * pcm_out_channels,
                        std::size_t{count - first} * pcm_frame_bytes);
        return;
    }
    for (std::uint32_t index = 0; index < count; ++index, ++tail, source += channels) {
        std::int16_t* out = ring + std::size_t{tail & pcm_ring_mask} * pcm_out_channels;
        if (channels == 1) {
            out[0] = out[1] = source[0];
            continue;
        }
        // 23170 = 0.707 in Q15, 16384 = 0.5 in Q15. Worst case |sum| = 55938 * 32768 < 2^31.
        const std::int32_t left = (23170 * source[0] + 16384 * source[2] + 16384 * source[4]) >> 15;
        const std::int32_t right = (23170 * source[1] + 16384 * source[2] + 16384 * source[5]) >> 15;
        out[0] = saturate16(left);
        out[1] = saturate16(right);
    }
}

#if TARGET_OS_IPHONE
bool activate_audio_session() {
    @autoreleasepool {
        NSError* error = nil;
        if (![AVAudioSession.sharedInstance setActive:YES error:&error]) {
            PCM_LOG("cannot activate audio session: %s", error.localizedDescription.UTF8String);
            return false;
        }
        return true;
    }
}
bool prepare_audio_session() {
    @autoreleasepool {
        AVAudioSession* session = AVAudioSession.sharedInstance;
        NSError* error = nil;
        if (![session setCategory:AVAudioSessionCategoryPlayback error:&error] ||
            ![session setPreferredSampleRate:pcm_rate error:&error]) {
            PCM_LOG("cannot configure audio session: %s", error.localizedDescription.UTF8String);
            return false;
        }
        static std::once_flag installed;
        std::call_once(installed, [session] {
            static id observer = [NSNotificationCenter.defaultCenter
                addObserverForName:AVAudioSessionInterruptionNotification object:session queue:nil
                usingBlock:^(NSNotification* notification) {
                    if ([notification.userInfo[AVAudioSessionInterruptionTypeKey] unsignedIntegerValue] ==
                        AVAudioSessionInterruptionTypeEnded) pcm_sink_resume_devices();
                }];
            (void)observer;
        });
        return activate_audio_session();
    }
}
#endif

int open_unit(PcmSink* sink) {
    AudioComponentDescription description{};
    description.componentType = kAudioUnitType_Output;
#if TARGET_OS_IPHONE
    description.componentSubType = kAudioUnitSubType_RemoteIO;  // No DefaultOutput unit; input element 1 is never enabled.
#else
    description.componentSubType = kAudioUnitSubType_DefaultOutput;  // Output only; input element 1 is never enabled.
#endif
    description.componentManufacturer = kAudioUnitManufacturer_Apple;
    OSStatus status = AU_NEW(&description, &sink->unit);
    if (status || !sink->unit) {
        sink->unit = nullptr;
        sink->last_error = status ? status : -1;
        PCM_LOG("cannot create the default output AudioUnit (status %d)", static_cast<int>(sink->last_error));
        return -1;
    }

    AudioStreamBasicDescription format{};
    format.mSampleRate = pcm_rate;
    format.mFormatID = kAudioFormatLinearPCM;
    format.mFormatFlags = kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked;
    format.mBytesPerPacket = pcm_frame_bytes;
    format.mFramesPerPacket = 1;
    format.mBytesPerFrame = pcm_frame_bytes;
    format.mChannelsPerFrame = pcm_out_channels;
    format.mBitsPerChannel = 16;
    AURenderCallbackStruct callback{render_callback, sink};
    const char* step = "set stream format";
    status = AU_SET_PROPERTY(sink->unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0,
                             &format, sizeof format);
    if (!status) {
        step = "set render callback";
        status = AU_SET_PROPERTY(sink->unit, kAudioUnitProperty_SetRenderCallback,
                                 kAudioUnitScope_Input, 0, &callback, sizeof callback);
    }
    if (!status) {
        step = "initialize";
        status = AU_INITIALIZE(sink->unit);
    }
    if (status) {
        sink->last_error = status;
        PCM_LOG("cannot %s on the default output AudioUnit (status %d); no output device?", step,
                static_cast<int>(status));
        const OSStatus disposed = AU_DISPOSE(sink->unit);
        if (disposed) {
            sink->last_error = disposed;
            PCM_LOG("AudioComponentInstanceDispose failed during open cleanup (status %d); retaining callback storage",
                    static_cast<int>(disposed));
            return -2;  // The live unit may still invoke its callback: the caller must retain the sink and ring.
        }
        sink->unit = nullptr;
        quiesce(sink);
        return -1;
    }
    return 0;
}

}  // namespace

PcmSink* pcm_sink_open(bool audible) {
#if TARGET_OS_IPHONE
    if (audible && !prepare_audio_session()) return nullptr;
#endif
    auto* sink = new (std::nothrow) PcmSink;
    if (!sink) {
        PCM_LOG("cannot allocate the sink");
        return nullptr;
    }
    sink->audible = audible;
    store_gain(sink, 1.0f);  // The documented initial gain is unity.
    if (sink->audible) {
        sink->ring = static_cast<std::int16_t*>(std::calloc(pcm_ring_frames, pcm_frame_bytes));
        if (!sink->ring) {
            PCM_LOG("cannot allocate the %u-frame ring", static_cast<unsigned>(pcm_ring_frames));
            delete sink;
            return nullptr;
        }
        const int opened = open_unit(sink);
        if (opened) {
            if (opened != -2) {
                std::free(sink->ring);
                delete sink;
            }
            return nullptr;
        }
        pthread_mutex_lock(&units_lock);
        sink->next_unit = units;
        units = sink;
        pthread_mutex_unlock(&units_lock);
    }
    return sink;
}

int pcm_sink_set_active(PcmSink* sink, bool active) {
    if (!sink) return -1;
    const int was = sink->active.load(std::memory_order_relaxed);
    if (active) {
        if (was) return 0;
        // The ring is empty here: open starts empty and stop flushes. Start the
        // device first so a start failure leaves the sink exactly as it was
        // (stopped, no callback ever consuming).
        if (sink->audible) {
            pthread_mutex_lock(&units_lock);
            const OSStatus status = AU_START(sink->unit);
            if (!status) sink->active.store(1, std::memory_order_seq_cst);
            pthread_mutex_unlock(&units_lock);
            if (status) {
                sink->last_error = status;
                PCM_LOG("cannot start the output AudioUnit (status %d); sink stays stopped",
                        static_cast<int>(status));
                return -1;
            }
            return 0;
        }
        sink->active.store(1, std::memory_order_seq_cst);
        return 0;
    }
    if (!was) return 0;
    int result = 0;
    if (sink->audible) {
        pthread_mutex_lock(&units_lock);
        sink->active.store(0, std::memory_order_seq_cst);
        const OSStatus status = AU_STOP(sink->unit);
        pthread_mutex_unlock(&units_lock);
        if (status) {
            sink->last_error = status;
            PCM_LOG("cannot stop the output AudioUnit (status %d); queued PCM is discarded anyway",
                    static_cast<int>(status));
            result = -1;
        }
    } else {
        sink->active.store(0, std::memory_order_seq_cst);
    }
    quiesce(sink);
    // Consumer quiescent and `active` clear: flush by moving the read index up to the write index.
    const std::uint32_t head = sink->head.load(std::memory_order_relaxed);
    const std::uint32_t tail = sink->tail.load(std::memory_order_relaxed);
    if (tail != head) {
        sink->dropped.fetch_add(tail - head, std::memory_order_relaxed);
        sink->head.store(tail, std::memory_order_seq_cst);
    }
    return result;
}

std::size_t pcm_sink_write(PcmSink* sink, const std::int16_t* samples, std::size_t frames,
                           unsigned channels) {
    if (!sink || !samples || frames == 0) return 0;
    if (channels != 1 && channels != 2 && channels != 6) return 0;
    if (frames > std::numeric_limits<std::size_t>::max() / sizeof(*samples) / channels)
        return 0;  // No valid source span can represent this request.
    if (!sink->active.load(std::memory_order_relaxed)) return 0;  // Stopped: rejected, uncounted.
    if (!sink->audible) {
        // Muted: accepted frames are consumed on the spot; no ring, no conversion, no device.
        sink->submitted.fetch_add(frames, std::memory_order_relaxed);
        sink->consumed.fetch_add(frames, std::memory_order_relaxed);
        return frames;
    }
    const std::uint32_t tail = sink->tail.load(std::memory_order_relaxed);
    const std::uint32_t head = sink->head.load(std::memory_order_acquire);
    const std::uint32_t space = pcm_ring_frames - (tail - head);
    // Bounded: at most `space` <= pcm_ring_frames frames are read from `samples`.
    const std::uint32_t count = frames < space ? static_cast<std::uint32_t>(frames) : space;
    if (count) {
        store_frames(sink->ring, tail, samples, count, channels);
        sink->tail.store(tail + count, std::memory_order_release);
        sink->submitted.fetch_add(count, std::memory_order_relaxed);
    }
    if (frames > count) sink->dropped.fetch_add(frames - count, std::memory_order_relaxed);
    return count;
}

int pcm_sink_set_volume(PcmSink* sink, float volume) {
    if (!sink) return -1;
    if (!std::isfinite(volume)) return -1;  // NaN and both infinities: rejected, the stored gain is untouched.
    // Finite out-of-range values clamp into [0, 128] (the device volume
    // range); negative values clamp to silence. Also valid for muted sinks:
    // the gain is stored, but the muted path still makes no AudioToolbox calls.
    // Stop/restart retains the stored value; nothing resets it but open.
    const float gain = volume < 0.0f ? 0.0f : (volume > 128.0f ? 128.0f : volume);
    store_gain(sink, gain);
    return 0;
}

void pcm_sink_get_counters(const PcmSink* sink, PcmSinkCounters* out) {
    if (!out) return;
    if (!sink) {
        *out = {};
        return;
    }
    out->submitted_frames = sink->submitted.load(std::memory_order_relaxed);
    out->consumed_frames = sink->consumed.load(std::memory_order_relaxed);
    out->dropped_frames = sink->dropped.load(std::memory_order_relaxed);
    out->underrun_frames = sink->underrun.load(std::memory_order_relaxed);
}

void pcm_sink_resume_devices() {
#if TARGET_OS_IPHONE
    pthread_mutex_lock(&units_lock);
    const bool audible = units != nullptr;
    pthread_mutex_unlock(&units_lock);
    if (!audible || !activate_audio_session()) return;
#endif
    pthread_mutex_lock(&units_lock);
    for (PcmSink* sink = units; sink; sink = sink->next_unit) {
        if (!sink->active.load(std::memory_order_relaxed)) continue;
        const OSStatus status = AU_START(sink->unit);  // last_error is producer-owned: report only
        if (status)
            PCM_LOG("cannot restart the output AudioUnit (status %d); the sink stays silent until it is restarted",
                    static_cast<int>(status));
    }
    pthread_mutex_unlock(&units_lock);
}

void pcm_sink_close(PcmSink* sink) {
    if (!sink) return;
    pcm_sink_set_active(sink, false);  // Clears `active`, stops the unit, waits for the consumer, flushes.
    if (sink->unit) {
        pthread_mutex_lock(&units_lock);  // A resume never sees a unit being disposed.
        for (PcmSink** link = &units; *link; link = &(*link)->next_unit) {
            if (*link == sink) {
                *link = sink->next_unit;
                break;
            }
        }
        pthread_mutex_unlock(&units_lock);
        // Only successful disposal excludes future callbacks. The quiesce
        // below covers a callback that was already past the active check; it
        // cannot substitute for disposal.
        OSStatus status = AU_UNINITIALIZE(sink->unit);
        if (status)
            PCM_LOG("AudioUnitUninitialize failed (status %d); disposing anyway", static_cast<int>(status));
        status = AU_DISPOSE(sink->unit);
        if (status) {
            sink->last_error = status;
            PCM_LOG("AudioComponentInstanceDispose failed (status %d); retaining inactive callback storage",
                    static_cast<int>(status));
            return;  // Fail-safe retention: a failed dispose does not exclude future callbacks.
        }
        sink->unit = nullptr;
    }
    quiesce(sink);
    std::free(sink->ring);
    delete sink;
}

namespace {

// The portable output interface over one sink. The renderer thread is the
// single producer; set_volume may come from the audio device service, so the
// adapter serializes API use as the sink's contract requires.
class SinkOutput final : public audio::PcmOutput {
public:
    explicit SinkOutput(PcmSink* sink) : sink_(sink) {
        // ARMIGO_AUDIO_DUMP=FILE.wav (diagnostic): also writes every accepted
        // frame, as 48 kHz stereo PCM16, to a WAV file.
        if (const char* path = std::getenv("ARMIGO_AUDIO_DUMP"); path && *path) {
            dump_ = std::fopen(path, "wb");
            if (dump_) {
                const std::array<std::byte, 44> header{};
                std::fwrite(header.data(), 1, header.size(), dump_);
            }
        }
    }
    ~SinkOutput() override {
        report(true);
        pcm_sink_close(sink_);
        if (dump_) {
            finish_dump();
            std::fclose(dump_);
        }
    }

    bool set_active(bool active) override {
        const std::lock_guard lock(mutex_);
        return pcm_sink_set_active(sink_, active) == 0;
    }
    std::size_t write(const std::int16_t* samples, std::size_t frames, unsigned channels) override {
        const std::lock_guard lock(mutex_);
        const auto accepted = pcm_sink_write(sink_, samples, frames, channels);
        if (dump_ && accepted && (channels == 1 || channels == 2 || channels == 6)) {
            for (std::size_t frame = 0; frame < accepted; ++frame) {
                const std::int16_t left = samples[frame * channels];
                const std::int16_t right = samples[frame * channels + (channels == 1 ? 0 : 1)];
                std::fwrite(&left, sizeof left, 1, dump_);
                std::fwrite(&right, sizeof right, 1, dump_);
            }
            dumped_frames_ += accepted;
        }
        report(false);
        return accepted;
    }
    bool set_volume(float volume) override {
        const std::lock_guard lock(mutex_);
        return pcm_sink_set_volume(sink_, volume) == 0;
    }
    audio::PcmCounters counters() const override {
        PcmSinkCounters counters;
        pcm_sink_get_counters(sink_, &counters);
        return {counters.submitted_frames, counters.consumed_frames, counters.dropped_frames,
                counters.underrun_frames};
    }

private:
    // ARMIGO_AUDIO_TRACE=1: the sink's counters every ten seconds and at close.
    void report(bool final) {
        static const bool trace = [] {
            const char* value = std::getenv("ARMIGO_AUDIO_TRACE");
            return value && *value == '1';
        }();
        if (!trace) return;
        const auto now = std::chrono::steady_clock::now();
        if (!final && now - reported_ < std::chrono::seconds(10)) return;
        reported_ = now;
        PcmSinkCounters counters;
        pcm_sink_get_counters(sink_, &counters);
        armigo::diag::opt_in("[audio output] submitted=%llu consumed=%llu dropped=%llu underrun=%llu%s\n",
                     static_cast<unsigned long long>(counters.submitted_frames),
                     static_cast<unsigned long long>(counters.consumed_frames),
                     static_cast<unsigned long long>(counters.dropped_frames),
                     static_cast<unsigned long long>(counters.underrun_frames), final ? " (closed)" : "");
    }
    void finish_dump() {
        const auto put32 = [&](std::uint32_t value) { std::fwrite(&value, 4, 1, dump_); };
        const auto put16 = [&](std::uint16_t value) { std::fwrite(&value, 2, 1, dump_); };
        const auto bytes = static_cast<std::uint32_t>(dumped_frames_ * 4);
        std::fseek(dump_, 0, SEEK_SET);
        std::fwrite("RIFF", 1, 4, dump_); put32(36 + bytes);
        std::fwrite("WAVEfmt ", 1, 8, dump_); put32(16); put16(1); put16(2);
        put32(48000); put32(48000 * 4); put16(4); put16(16);
        std::fwrite("data", 1, 4, dump_); put32(bytes);
    }

    PcmSink* sink_;
    std::mutex mutex_;
    std::FILE* dump_ = nullptr;
    std::uint64_t dumped_frames_ = 0;
    std::chrono::steady_clock::time_point reported_ = std::chrono::steady_clock::now();
};

}  // namespace

std::shared_ptr<audio::PcmOutput> make_pcm_sink_output(bool audible) {
    PcmSink* sink = pcm_sink_open(audible);
    if (!sink) return nullptr;
    try {
        return std::make_shared<SinkOutput>(sink);
    } catch (...) {
        pcm_sink_close(sink);
        return nullptr;
    }
}

}  // namespace armigo::platform
