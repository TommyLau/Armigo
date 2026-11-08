// ============================================================================
// Armigo Tests - Native Runtime Verification
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

// Independently authored synthetic process entries. The tests place this code
// in an NSO and run only its sealed offline-prepared copy.
    .text
    .p2align 2
    .globl _armigo_process_fixture_begin
    .globl _armigo_process_fixture_end
    .globl _armigo_process_fixture_main
    .globl _armigo_process_fixture_main_end
    .globl _armigo_process_fixture_worker
    .globl _armigo_process_fixture_worker_end
    .globl _armigo_process_fixture_block
    .globl _armigo_process_fixture_block_end
    .globl _armigo_process_fixture_grant_stop
    .globl _armigo_process_fixture_grant_stop_end
    .globl _armigo_process_fixture_grant_heap
    .globl _armigo_process_fixture_grant_worker_stack
    .globl _armigo_process_fixture_worker_stack
    .globl _armigo_process_fixture_original
    .globl _armigo_process_fixture_original_end
    .globl _armigo_process_fixture_original_slot
    .globl _armigo_process_fixture_original_target
    .globl _armigo_process_fixture_original_relocate_slot
    .globl _armigo_process_fixture_original_worker
    .globl _armigo_process_fixture_original_worker_end
    .globl _armigo_process_fixture_original_worker_stack

_armigo_process_fixture_begin:
_armigo_process_fixture_original:
    cbnz x0, 90f
    cbz x1, 90f
    cbnz x16, 90f
    cbnz x29, 90f
    cbnz x30, 90f
    tbz x18, #0, 90f
    mrs x9, fpcr
    cbnz x9, 90f
    mrs x9, fpsr
    cbnz x9, 90f
    mov x19, x1
    mov x20, sp
    and x9, x20, #15
    cbnz x9, 90f
    mrs x9, TPIDR_EL0
    cbnz x9, 90f
    mrs x21, TPIDRRO_EL0
    cbz x21, 90f

    // Check every kernel-provided TLR byte before any IPC or SDK writes: all
    // clear except the thread's own handle at 0x110.
    mov x9, #0
1:
    ldr x10, [x21, x9]
    cmp x9, #0x110
    b.ne 3f
    mov w11, w1
    cmp x10, x11
    b.ne 90f
    b 4f
3:
    cbnz x10, 90f
4:
    add x9, x9, #8
    cmp x9, #0x200
    b.ne 1b

    // The guest bootstrap performs its own authored relative relocation.
_armigo_process_fixture_original_slot:
    adr x9, .
    ldr x10, [x9]
    cbnz x10, 90f
_armigo_process_fixture_original_relocate_slot:
    adr x9, .
_armigo_process_fixture_original_target:
    adr x10, .
    str x10, [x9]
    ldr x11, [x9]
    sub x11, x11, x10
    cbnz x11, 90f

    // Query the last byte below entry SP. The NPDM supplies 256 KiB here.
    sub sp, sp, #64
    mov x0, sp
    add x1, sp, #48
    sub x2, x20, #1
    svc #0x06
    cbnz x0, 90f
    ldr x9, [sp]
    ldr x10, [sp, #8]
    mov x11, #0x40000
    cmp x10, x11
    b.ne 90f
    add x11, x9, x10
    cmp x11, x20
    b.ne 90f
    ldr w11, [sp, #16]
    cmp w11, #11
    b.ne 90f
    ldr w11, [sp, #24]
    cmp w11, #3
    b.ne 90f
    // Both stack endpoints are usable, including the requested bottom page.
    ldr x10, [x9]
    cbnz x10, 90f
    str x19, [x9]
    ldr x10, [x9]
    cmp x10, x19
    b.ne 90f
    str xzr, [x9]

    // The initial x1 must be a real thread handle, not a pointer or pseudo handle.
    mov x1, x19
    svc #0x0c
    cbnz x0, 90f
    cmp w1, #44
    b.ne 90f

    // Query the TLR mapping and exercise the guest-owned writable TLS register.
    mov x0, sp
    add x1, sp, #48
    mov x2, x21
    svc #0x06
    cbnz x0, 90f
    ldr w9, [sp, #16]
    cmp w9, #12
    b.ne 90f
    add x9, x21, #0x100
    msr TPIDR_EL0, x9
    mrs x10, TPIDR_EL0
    cmp x9, x10
    b.ne 90f
    mrs x10, TPIDRRO_EL0
    cmp x10, x21
    b.ne 90f
    add sp, sp, #64

    // CreateThread snapshots FPCR/FPSR. Changing the creator after creation
    // must not affect the worker that StartThread later makes runnable.
    movz x9, #0x02c0, lsl #16
    msr fpcr, x9
    movz x9, #0x0800, lsl #16
    movk x9, #3
    msr fpsr, x9
    adr x1, _armigo_process_fixture_original_worker
    add x2, x21, #0x100
_armigo_process_fixture_original_worker_stack:
    adr x3, .
    add x3, x3, #16, lsl #12
    mov w4, #44
    mov w5, #0
    svc #0x08
    cbnz x0, 90f
    movz x9, #0x0140, lsl #16
    msr fpcr, x9
    mov x9, #0x14
    msr fpsr, x9
    mov x0, x1
    svc #0x09
    cbnz x0, 90f
    add x0, x21, #0x100
    mov x1, #2
    mov x2, #0
    mov x3, #-1
    svc #0x34
    cbz x0, 2f
    mov w9, #0xfa01
    cmp w0, w9
    b.ne 90f
2:
    ldr w9, [x21, #0x100]
    cmp w9, #1
    b.ne 90f
    svc #0x07
    b .
90:
    mov x0, #0x51
    mov x1, #0
    mov x2, #0
    svc #0x26
    b .
_armigo_process_fixture_original_end:

_armigo_process_fixture_original_worker:
    cbz x0, 91f
    cbnz x1, 91f
    cbnz x16, 91f
    cbnz x29, 91f
    cbnz x30, 91f
    tbz x18, #0, 91f
    mrs x9, fpcr
    movz x10, #0x02c0, lsl #16
    cmp x9, x10
    b.ne 91f
    mrs x9, fpsr
    movz x10, #0x0800, lsl #16
    movk x10, #3
    cmp x9, x10
    b.ne 91f
    mrs x9, TPIDR_EL0
    cbnz x9, 91f
    mrs x10, TPIDRRO_EL0
    cbz x10, 91f
    sub x11, x0, #0x100
    cmp x10, x11
    b.eq 91f
    mov x9, #0
1:
    ldr x11, [x10, x9]
    cmp x9, #0x110
    b.ne 3f
    // The worker's own handle: a 32-bit handle, not a pointer.
    cbz w11, 91f
    lsr x11, x11, #32
3:
    cbnz x11, 91f
    add x9, x9, #8
    cmp x9, #0x200
    b.ne 1b
    mov x1, #1
    mov x2, #0
    mov x3, #1
    svc #0x35
    cbnz x0, 91f
    svc #0x0a
    b .
91:
    mov x0, #0x52
    mov x1, #0
    mov x2, #0
    svc #0x26
    b .
_armigo_process_fixture_original_worker_end:

_armigo_process_fixture_main:
    cbnz x0, 90f
    cbnz x1, 90f
    mov x19, sp
    and x9, x19, #15
    cbnz x9, 90f
    mrs x20, TPIDR_EL0
    mrs x21, TPIDRRO_EL0
    cbz x20, 90f
    cmp x20, x21
    b.ne 90f
    ldr x9, [x20]
    cbnz x9, 90f

    // ConnectToNamedPort("sm:").
    adr x1, 80f
    svc #0x1f
    cbnz x0, 90f
    mov x22, x1

    // RegisterClient with a PID descriptor. The transport supplies the trusted
    // process identity; the zero payload PID is intentionally not authoritative.
    bl 70f
    mov x9, #0x10
    movk x9, #0x8000, lsl #48
    str x9, [x20]
    mov w9, #1
    str w9, [x20, #8]
    mov x0, x22
    svc #0x21
    cbnz x0, 90f

    // Ask sm: for fsp-srv.
    bl 70f
    mov x9, #0x11
    movk x9, #2, lsl #32
    str x9, [x20]
    mov x9, #0x7366
    movk x9, #0x2d70, lsl #16
    movk x9, #0x7273, lsl #32
    movk x9, #0x76, lsl #48
    str x9, [x20, #8]
    mov x0, x22
    svc #0x21
    cbnz x0, 90f
    ldr w23, [x20, #12]
    cbz w23, 90f

    // Convert the fsp-srv session to a domain.
    bl 70f
    mov w9, #5
    str w9, [x20]
    mov w9, #8
    str w9, [x20, #4]
    mov w9, #0x4653
    movk w9, #0x4943, lsl #16
    str w9, [x20, #16]
    mov x0, x23
    svc #0x21
    cbnz x0, 90f
    ldr w9, [x20, #32]
    cmp w9, #1
    b.ne 90f

    // Open the configured RomFS storage as domain object 2.
    bl 70f
    mov w9, #6
    str w9, [x20]
    mov w9, #12
    str w9, [x20, #4]
    mov w9, #1
    movk w9, #0x10, lsl #16
    str w9, [x20, #16]
    mov w9, #1
    str w9, [x20, #20]
    mov w9, #0x4653
    movk w9, #0x4943, lsl #16
    str w9, [x20, #32]
    mov w9, #1
    str w9, [x20, #36]
    mov w9, #200
    str w9, [x20, #40]
    mov x0, x23
    svc #0x21
    cbnz x0, 90f
    ldr w9, [x20, #48]
    cmp w9, #2
    b.ne 90f

    // Read the public RomFS header through one complete HIPC receive descriptor.
    bl 70f
    mov w9, #6
    movk w9, #0x100, lsl #16
    str w9, [x20]
    mov w9, #16
    str w9, [x20, #4]
    mov w9, #0x50
    str w9, [x20, #8]
    add x10, x20, #0x100
    str w10, [x20, #12]
    lsr x11, x10, #36
    and x11, x11, #7
    lsl x11, x11, #2
    lsr x12, x10, #32
    and x12, x12, #15
    lsl x12, x12, #28
    orr x11, x11, x12
    str w11, [x20, #16]
    mov w9, #1
    movk w9, #0x20, lsl #16
    str w9, [x20, #32]
    mov w9, #2
    str w9, [x20, #36]
    mov w9, #0x4653
    movk w9, #0x4943, lsl #16
    str w9, [x20, #48]
    mov w9, #1
    str w9, [x20, #52]
    mov w9, #0x50
    str w9, [x20, #72]
    mov x0, x23
    svc #0x21
    cbnz x0, 90f
    ldr x9, [x20, #0x100]
    cmp x9, #0x50
    b.ne 90f
    ldr x24, [x20, #0x148]

    // Read and verify the independently authored file at the RomFS data offset.
    // This single-file fixture has no preceding file-data entries.
    bl 70f
    mov w9, #6
    movk w9, #0x100, lsl #16
    str w9, [x20]
    mov w9, #16
    str w9, [x20, #4]
    mov w9, #9
    str w9, [x20, #8]
    add x10, x20, #0x180
    str w10, [x20, #12]
    lsr x11, x10, #36
    and x11, x11, #7
    lsl x11, x11, #2
    lsr x12, x10, #32
    and x12, x12, #15
    lsl x12, x12, #28
    orr x11, x11, x12
    str w11, [x20, #16]
    mov w9, #1
    movk w9, #0x20, lsl #16
    str w9, [x20, #32]
    mov w9, #2
    str w9, [x20, #36]
    mov w9, #0x4653
    movk w9, #0x4943, lsl #16
    str w9, [x20, #48]
    mov w9, #1
    str w9, [x20, #52]
    str x24, [x20, #64]
    mov w9, #9
    str w9, [x20, #72]
    mov x0, x23
    svc #0x21
    cbnz x0, 90f
    ldr x9, [x20, #0x180]
    mov x10, #0x7973
    movk x10, #0x746e, lsl #16
    movk x10, #0x6568, lsl #32
    movk x10, #0x6974, lsl #48
    cmp x9, x10
    b.ne 90f
    ldrb w9, [x20, #0x188]
    cmp w9, #0x63
    b.ne 90f

    // CreateThread(entry, arg, exact stack top, priority, core).
    adr x1, _armigo_process_fixture_worker
    add x2, x20, #0x300
_armigo_process_fixture_worker_stack:
    adr x3, _armigo_process_fixture_worker_stack
    add x3, x3, #16, lsl #12
    mov w4, #0x2c
    mov w5, #0
    svc #0x08
    cbnz x0, 90f
    cbz x1, 90f
    mov x0, x1
    svc #0x09
    cbnz x0, 90f

    // The offline classifier cannot yet prove that CreateThread replaces W1
    // with a handle on only its successful edge. Keep the real thread handle
    // in registers for StartThread. Exercise the zero-object 0x18 timeout path,
    // then use the firmware address wait for the worker completion handshake.
    mov x1, #0
    mov x2, #0
    mov x3, #0
    svc #0x18
    cbz x0, 90f
    add x0, x20, #0x300
    mov x1, #2
    mov x2, #0
    mov x3, #-1
    svc #0x34
    cbz x0, 60f
    mov w9, #0xfa01
    cmp w0, w9
    b.ne 90f
60:
    ldr x9, [x20, #0x300]
    cmp x9, #1
    b.ne 90f

    svc #0x07
    b .

70:
    mov x9, #32
    mov x10, x20
71:
    stp xzr, xzr, [x10], #16
    subs x9, x9, #1
    b.ne 71b
    ret

80:
    .asciz "sm:"
    .p2align 2
90:
    svc #0x42
    b .
_armigo_process_fixture_main_end:

_armigo_process_fixture_worker:
    cbnz x1, 91f
    mrs x9, TPIDR_EL0
    mrs x10, TPIDRRO_EL0
    cbz x9, 91f
    cmp x9, x10
    b.ne 91f
    ldr x11, [x9]
    cbnz x11, 91f
    sub x11, x0, #0x300
    cmp x9, x11
    b.eq 91f
    mov x1, #1
    mov x2, #0
    mov x3, #1
    svc #0x35
    cbnz x0, 91f
    svc #0x0a
    b .
91:
    svc #0x42
    b .
_armigo_process_fixture_worker_end:

_armigo_process_fixture_block:
    cbnz x0, 92f
    cbnz x1, 92f
    mrs x9, TPIDR_EL0
    mrs x10, TPIDRRO_EL0
    cbz x9, 92f
    cmp x9, x10
    b.ne 92f
    mov x1, #0
    mov x2, #0
    mov x3, #-1
    svc #0x18
92:
    svc #0x07
    b .
_armigo_process_fixture_block_end:

_armigo_process_fixture_grant_stop:
    cbnz x0, 95f
    cbnz x1, 95f
_armigo_process_fixture_grant_heap:
    movz x19, #0
    movk x19, #0, lsl #16
    movk x19, #0, lsl #32
    movk x19, #0, lsl #48
    mov x1, #0x200000
    svc #0x01
    cbnz x0, 96f
    cmp x1, x19
    b.ne 97f

    // Exercise the conservative ownership observation: the mutex word does
    // not name x0, so ArbitrateLock succeeds without a kernel-side grant. The
    // runtime still retains tag 0x80000001 until the word changes. Bit 30 is
    // the waiter marker; bit 31 remains part of the owner tag and catches an
    // unsafe mask that would discard both high bits.
    mov w9, #1
    movk w9, #0x8000, lsl #16
    stlr w9, [x19]
    mov x0, #2
    mov x1, x19
    mov w2, w9
    svc #0x1a
    cbnz x0, 98f
    mov w9, #1
    add x20, x19, #4
    stlr w9, [x20]
    add x20, x19, #8
93:
    ldar w9, [x20]
    cbz w9, 93b
    adr x1, _armigo_process_fixture_worker
    add x2, x19, #16
_armigo_process_fixture_grant_worker_stack:
    adr x3, _armigo_process_fixture_worker_stack
    add x3, x3, #16, lsl #12
    mov w4, #0x2c
    mov w5, #0
    svc #0x08
    svc #0x42
    b .
95:
    svc #0x40
    b .
96:
    svc #0x41
    b .
97:
    svc #0x42
    b .
98:
    svc #0x43
    b .
_armigo_process_fixture_grant_stop_end:

    .globl _armigo_process_fixture_mutex_stop
    .globl _armigo_process_fixture_mutex_stop_end
    .globl _armigo_process_fixture_mutex_heap
    .globl _armigo_process_fixture_mutex_waiter_stack
    .globl _armigo_process_fixture_mutex_owner
    .globl _armigo_process_fixture_mutex_owner_end
    .globl _armigo_process_fixture_mutex_waiter
    .globl _armigo_process_fixture_mutex_waiter_end

// Heap words: mutex, owner handle, waiter handle, owner ready, contention ready,
// handoff received, run query, query resumed, run unlock, unlock resumed, escape.
_armigo_process_fixture_mutex_stop:
_armigo_process_fixture_mutex_heap:
    movz x19, #0
    movk x19, #0, lsl #16
    movk x19, #0, lsl #32
    movk x19, #0, lsl #48
    mov x1, #0x200000
    svc #0x01
    cbnz x0, 109f
    cmp x1, x19
    b.ne 109f
    adr x1, _armigo_process_fixture_mutex_owner
    mov x2, x19
    add x3, x19, #0x200, lsl #12
    mov w4, #44
    mov w5, #0
    svc #0x08
    cbnz x0, 109f
    // Reconstruct the returned handle through flags: preparation conservatively
    // retains the entry-address provenance in X1 across CreateThread.
    mov w12, w1
    mov w13, #32
    mov w14, #0
101:
    lsr w14, w14, #1
    ands w9, w12, #1
    cset w15, ne
    orr w14, w14, w15, lsl #31
    lsr w12, w12, #1
    subs w13, w13, #1
    b.ne 101b
    str w14, [x19, #4]
    mov x0, x1
    svc #0x09
    cbnz x0, 109f
    // Bootstrap the first real grant through a condition-variable handoff.
    // The fixture never writes an owner tag into the mutex word itself.
    add x21, x19, #44
100:
    ldar w9, [x21]
    cbz w9, 100b
    mov x0, x21
    mov x1, #1
    svc #0x1d
    cbnz x0, 109f
    adr x1, _armigo_process_fixture_mutex_waiter
    mov x2, x19
_armigo_process_fixture_mutex_waiter_stack:
    adr x3, _armigo_process_fixture_mutex_waiter_stack
    add x3, x3, #16, lsl #12
    mov w4, #44
    mov w5, #0
    svc #0x08
    cbnz x0, 109f
    mov w12, w1
    mov w13, #32
    mov w14, #0
102:
    lsr w14, w14, #1
    ands w9, w12, #1
    cset w15, ne
    orr w14, w14, w15, lsl #31
    lsr w12, w12, #1
    subs w13, w13, #1
    b.ne 102b
    str w14, [x19, #8]
    mov x0, x1
    svc #0x09
    cbnz x0, 109f
    mov x1, #0
    mov x2, #0
    mov x3, #-1
    svc #0x18
109:
    svc #0x42
    b .
_armigo_process_fixture_mutex_stop_end:

_armigo_process_fixture_mutex_owner:
    mov x19, x0
    ldr w20, [x19, #4]
    // Obtain an actual kernel grant before admitting the lock contender.
    mov x0, x19
    add x1, x19, #44
    mov w2, w20
    mov x3, #-1
    svc #0x1c
    cbnz x0, 119f
    ldar w9, [x19]
    cmp w9, w20
    b.ne 119f
    mov w9, #1
    add x21, x19, #12
    stlr w9, [x21]
    add x21, x19, #16
111:
    ldar w9, [x21]
    cbz w9, 111b
    // The host observes actual kernel enrollment before permitting this unlock.
    add x21, x19, #48
112:
    ldar w9, [x21]
    cbz w9, 112b
    mov x0, x19
    svc #0x1b
    cbnz x0, 119f
    svc #0x0a
119:
    svc #0x42
    b .
_armigo_process_fixture_mutex_owner_end:

_armigo_process_fixture_mutex_waiter:
    mov x19, x0
    add x21, x19, #12
120:
    ldar w9, [x21]
    cbz w9, 120b
    ldr w20, [x19, #4]
    ldr w22, [x19, #8]
    // Mark actual contention without replacing the current owner's tag.
    // The owner waits for our contention-ready flag before releasing, so
    // this marker update has no concurrent mutex writer.
    ldar w9, [x19]
    cmp w9, w20
    b.ne 129f
    orr w9, w9, #0x40000000
    stlr w9, [x19]
    mov w9, #1
    add x21, x19, #16
    stlr w9, [x21]
    mov w0, w20
    mov x1, x19
    mov w2, w22
    svc #0x1a
    cbnz x0, 129f
    // Only ArbitrateUnlock can write our tag: a raced no-grant return fails.
    ldar w9, [x19]
    cmp w9, w22
    b.ne 129f
    mov w9, #1
    add x21, x19, #20
    stlr w9, [x21]
    add x21, x19, #24
122:
    ldar w9, [x21]
    cbz w9, 122b
    // Pending cancellation must still allow this resume while the grant is held.
    mov x0, #0
    svc #0x0b
    mov w9, #1
    add x21, x19, #28
    stlr w9, [x21]
    add x21, x19, #32
123:
    ldar w9, [x21]
    cbz w9, 123b
    mov x0, x19
    svc #0x1b
    // With pending stop, executing even this first instruction is a regression.
    mov w9, #1
    add x21, x19, #36
    stlr w9, [x21]
    add x21, x19, #40
124:
    ldar w9, [x21]
    cbz w9, 124b
    svc #0x0a
129:
    svc #0x42
    b .
_armigo_process_fixture_mutex_waiter_end:
    .globl _armigo_process_fixture_nested_stop
    .globl _armigo_process_fixture_nested_stop_end
    .globl _armigo_process_fixture_nested_heap
    .globl _armigo_process_fixture_nested_waiter
    .globl _armigo_process_fixture_nested_waiter_end

// A worker owns a kernel mutex grant, then parks in a nested wait the main
// thread cannot release. Heap words: 0 grant mutex, 4 handle of the thread named
// as the nested mutex's owner, 8 nested mutex, 12 grant held, 16 go, 20 resumed
// after the wait, 24 wait kind (0 lock, 1 sleep, 2 condition wait, 3 user-mode poll, 4 user-mode spin), 28 about to
// block, 44 grant condition key, 52 nested condition key, 60 poll key, 64 escape.
_armigo_process_fixture_nested_stop:
_armigo_process_fixture_nested_heap:
    movz x19, #0
    movk x19, #0, lsl #16
    movk x19, #0, lsl #32
    movk x19, #0, lsl #48
    mov x1, #0x200000
    svc #0x01
    cbnz x0, 149f
    cmp x1, x19
    b.ne 149f
    adr x1, _armigo_process_fixture_nested_waiter
    mov x2, x19
    add x3, x19, #0x200, lsl #12
    mov w4, #44
    mov w5, #0
    svc #0x08
    cbnz x0, 149f
    mov x0, x1
    svc #0x09
    cbnz x0, 149f
    // Wait until the waiter has armed the condition key, then hand it the
    // grant mutex through a signal; the fixture never writes an owner tag.
    add x21, x19, #44
140:
    ldar w9, [x21]
    cbz w9, 140b
    mov x0, x21
    mov x1, #1
    svc #0x1d
    cbnz x0, 149f
    // Park until the process is terminated.
    mov x1, #0
    mov x2, #0
    mov x3, #-1
    svc #0x18
149:
    svc #0x42
    b .
_armigo_process_fixture_nested_stop_end:

_armigo_process_fixture_nested_waiter:
    mov x19, x0
    // Obtain an actual kernel grant of the grant mutex.
    mov x0, x19
    add x1, x19, #44
    mov w2, #0x11
    mov x3, #-1
    svc #0x1c
    cbnz x0, 159f
    ldar w9, [x19]
    cmp w9, #0x11
    b.ne 159f
    mov w9, #1
    add x21, x19, #12
    stlr w9, [x21]
    add x21, x19, #16
150:
    ldar w9, [x21]
    cbz w9, 150b
    ldr w10, [x19, #24]
    mov w9, #1
    add x21, x19, #28
    stlr w9, [x21]
    cbz w10, 151f
    cmp w10, #1
    b.eq 152f
    cmp w10, #3
    b.eq 153f
    cmp w10, #4
    b.eq 155f
    // Kind 2: condition-variable wait on the nested mutex, no timeout.
    add x0, x19, #8
    add x1, x19, #52
    mov w2, #0x11
    mov x3, #-1
    svc #0x1c
    b 158f
151:
    // Kind 0: ArbitrateLock on a mutex whose owner never releases it.
    ldr w0, [x19, #4]
    add x1, x19, #8
    mov w2, #0x11
    svc #0x1a
    b 158f
152:
    // Kind 1: a sleep far beyond the test's lifetime.
    mov x0, #1
    lsl x0, x0, #50
    svc #0x0b
    b 158f
153:
    // Kind 3: a user-mode wait for a release that never comes, polling with a
    // call that cannot block. Heap word 64 is the test's escape.
    add x21, x19, #64
154:
    add x0, x19, #60
    mov x1, #0
    svc #0x1d
    ldar w9, [x21]
    cbz w9, 154b
    svc #0x0a
155:
    // Kind 4: the same wait with no call at all, so no gate is ever reached.
    add x21, x19, #64
156:
    ldar w9, [x21]
    cbz w9, 156b
    svc #0x0a
158:
    // Reached only when the wait returned: termination must not resume it.
    // Spin in guest code, so a resumed wait leaves the run without quiescence
    // (the test fails on that) rather than exiting like a terminated thread.
    mov w9, #1
    add x21, x19, #20
    stlr w9, [x21]
    b .
159:
    svc #0x0a
    b .
_armigo_process_fixture_nested_waiter_end:
_armigo_process_fixture_end:
