// ============================================================================
// Armigo Platform - Android Guest Memory Mapping
// Copyright (c) 2026 Tommy Lau.
// SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-Armigo-Proprietary
//
// This file is licensed under GPL-3.0-only or, by separate written
// agreement, an Armigo proprietary license.
// See LICENSE, COMMERCIAL-LICENSE.md, and DISCLAIMER.md.
// ============================================================================

#include <asm/unistd.h>

.syntax unified
.arm
.text
.global armigo_3ds_map_memory
.type armigo_3ds_map_memory,%function
armigo_3ds_map_memory:
    push {r4-r7}
    mov r3,#0x100000
    orr r3,r3,#0x22
    mvn r4,#0
    mov r5,#0
    mov r7,#__NR_mmap2
    svc #0
    pop {r4-r7}
    bx lr
.size armigo_3ds_map_memory,.-armigo_3ds_map_memory
.section .note.GNU-stack,"",%progbits
