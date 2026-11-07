# Headless fluctuating-rate validation

The harness is opt-in. With neither `--pacing-schedule` nor `--frame-log`, the
runner constructs the original presentation clock and uses the original native
present pacer. These flags are rejected in windowed mode. They do not select a
title patch or change a game's nominal/reference rate.

```
armigo-headless run ... --refresh-hz 120 --pacing-schedule schedule.txt \
  --pacing-max-hz 120 --frame-log frames.tsv
```

`--pacing-max-hz` defaults to `--refresh-hz`. Choose the ceiling explicitly for
the mode under test: 70/140 capped at 60 correctly becomes constant 60; it is
not evidence of fluctuating guest time. Use 20/120 or 30/120 capped at 60 when
fluctuation is required in a 60 mode. A positive swap interval still counts
that many opportunities; interval zero remains immediate. Existing present-link
policy still determines the native present's effective interval.

The schedule uses milliseconds and Hz. Rate rows concatenate on an elapsed
monotonic-time axis starting at shell clock creation, before guest loading.
Blank lines and `#` comments are accepted. Rates are 1–1000 Hz; the schedule is
bounded to 24 hours and the generated grid to ten million opportunities.

```
fixed 10000 30
alternate 10000 70 140 500
alternate 10000 30 120 1000
alternate 10000 20 120 1000
alternate-frame 10000 70 140
ramp 10000 20 120
fixed 1000 29
fixed 1000 30
fixed 1000 31
fixed 1000 119
fixed 1000 120
fixed 1000 121
hitch 15000 50
hitch 25000 100
hitch 35000 250
```

`alternate D A B H` holds each rate for H ms. `alternate-frame D A B` alternates
one period of A and one period of B on the wall-time axis, using capped periods.
Late or missing frames do not advance that phase. `ramp D A B` interpolates Hz
linearly over D ms. After the final segment its second (or only) rate persists.
Each opportunity's next period is sampled at its start; a segment transition
therefore takes effect at the first opportunity at/after the boundary, with no
retroactive deadline changes. Hitches suppress opportunities in `[AT, AT+D)`;
the first suppressed opportunity is moved to AT+D. They do not block threads
with uncancellable sleeps. Overlapping hitches are rejected.

VI events and native/command-stream presents use the same immutable opportunity
grid under a schedule. Late frames skip obsolete opportunities without catch-up
bursts. The guest's 19.2 MHz counter, service clock and timeouts still measure
unscaled host monotonic time; the harness does not synthesize game dt. VI events
start when the guest acquires them and actual signal delivery can be late. This
is headless display policy, not a firmware or physical variable-refresh claim.
The run's time limit and timestamped input remain wall-time based. Inputs must
be aligned to scene readiness separately; startup time is part of the schedule.

`--frame-log` works with or without a schedule. Its TSV has one row after each
successful host consumer completion, before the VI callback returns:

- `elapsed_ns`: monotonic time since shell clock creation; `present_ns`: absolute
  monotonic timestamp of that observation (not GPU submission or physical scanout).
- `raw_period_ns`: difference between observations; first row is zero.
  `capped_period_ns` is `min(raw_period_ns, 50000000)`, an analysis proxy, not proof
  of the game's update count or published simulation step.
- `requested_interval`: the guest's swap interval; `requested_period_ns`: that
  interval times the instantaneous configured opportunity period. During hitches
  or a rate boundary this is not the realized wait duration.
- `refresh_index`: most recent grid opportunity at observation;
  `missed_opportunities`: positive excess over the requested interval since the
  preceding observed present. Native present-link overrides can change the
  effective interval. Log-only runs retain the original independent native cadence,
  so indices are grid-relative observations, not claimed native present targets.
- `vi_ns`, `vi_target_ns`, `vi_index`: most recent actual VI signal observation,
  its scheduled deadline and index. Zero means no observed VI signal yet. The
  timestamp is sampled after all events for that refresh have been signaled,
  under a short snapshot lock that is never held during watch copies or log I/O.
  Its difference from the target includes host wake/signaling/observation latency;
  it is a signal-completion observation, not an exact signal time or guest wake time.

Optionally set `ARMIGO_FRAME_WATCH=0xADDRESS:WIDTH,DECIMAL_ADDRESS:WIDTH` with widths
1, 2, 4 or 8 bytes (at most 64 entries). Each range is checked and copied exactly
once per logged frame while holding the guest mapping lock, without publishing
a pin that could reject a concurrent guest mapping change. Values are decoded
as unsigned little-endian integers; inaccessible
memory is `NA`. Floating values can be decoded by private analysis. Samples are
observational and may race guest writers; they are not atomic multi-field
snapshots. Watch lists are ignored unless frame logging is enabled. Keep all
addresses, logs, semantic landmarks, game-specific scripts and captures private.
Concurrent observers serialize timestamp sampling and row append together.
Watch copies and log flushes hold only that observation lock; VI signaling and
grid planning never acquire it. A slow log can delay presentation observations,
so instrumented present periods include that overhead. Logging errors latch the
first structured failure, which runtime polling sees even if the native
presentation callback has already returned.

For native pacing checks, `ARMIGO_NVN_TIMELINE=FILE` together with
`ARMIGO_NVN_TIMELINE_PRESENT_ONLY=1` retains the clock header and only the
`pres_*` and `present_*` events. These include queued presents, GPU-fence
readiness, native slot waits, entry into the host consumer, and successful
consumer completion (`pres_complete`). Combine this with
`ARMIGO_FRAME_STATS=0` to avoid the full GPU timeline and per-frame
diagnostic output. It still has measurement overhead; compare with an untraced
control. The filtered trace cannot supply GPU segment timings or a CPU/GPU
breakdown, and neither its consumer entry nor completion proves physical
scanout. Omitting the filter, or setting it to `0`, preserves the full timeline.

Timeline callers append binary records to a preallocated ring. A background
writer formats the existing TSV records, and normal process exit drains them.
`ARMIGO_NVN_TIMELINE_CAPACITY` selects 2–16777216 records (default 1048576).
A full ring drops new records without waiting. `timeline_overflow`,
`timeline_incomplete`, and `timeline_symbols_dropped` footer records report
loss; their key is the dropped count. `timeline_unquiesced` reports the number
of admitted attempts still active when shutdown closed admission; those
attempts may publish or report a loss after the footer snapshot. Reject traces containing these records
for pacing admission or complete gap attribution. Event names use fixed symbol
storage: at most 512 distinct names, each up to 63 bytes. Output errors are
reported on stderr. Forced termination can lose the undrained tail.

Use present-only tracing and `ARMIGO_FRAME_STATS=0` for pacing admission;
full traces are attribution runs and must be labeled as traced. Compare traced
and untraced controls on the same build and workload before attributing a
performance difference to the graphics pipeline.

For admission, record input/build/configuration/cache identity, scratch saves,
unchanged 30 A/A controls and separate fixed 30/60/120 controls. Compare semantic
landmarks in elapsed wall time and integrated capped simulation time; a present
is not necessarily a game update. Per-family endpoint tolerances, input repeats,
movie/audio clocks, pause/resume and long gameplay remain independent validation
requirements. The harness cannot infer their success from frame cadence alone.

## Relocatable watches

Absolute entries keep their syntax and column names. A loaded module name can
replace the absolute base: `main+0x100:8>+0x50:4` reads eight bytes at loaded
`main` plus 0x100 as an unsigned pointer, adds 0x50, then reads four bytes there.
Names match the actual loaded module names exactly; no title selection or fixed
module order is used. Bases are bound after the complete module set loads and
before guest execution. Unknown or ambiguous names, null pointers, arithmetic
overflow and unreadable steps produce `NA` for that frame. Each step uses the
same bounded mapping-locked copy, without publishing a retained pin.

A list supports **64 entries**, each with up to **16 pointer dereferences**
(17 reads including the final value). Offsets are nonnegative decimal or `0x`
hexadecimal; widths are 1, 2, 4 or 8 bytes. A chain's current unsigned value is
the base of its next `>+OFFSET:WIDTH` step. Pointer steps are separate samples,
not an atomic chain snapshot. Modules are not retained by the observer.

Column names are stable across relocation: `watch_main_256_8_p80_4` for the
example above, versus unchanged `watch_4096_4` for absolute `4096:4`. Numbers in
column names are decimal; `_pOFFSET_WIDTH` describes each pointer step. This
lets private analysis compare the same watch across launches despite ASLR.

## Scripted input delivery

With `--input-script`, `--frame-log frames.tsv` also writes
`frames.tsv.inputs.tsv`. `--input-log FILE` selects a different destination or
turns on delivery logging without frame logging. It is headless-only, requires
an input script, and must have a different path from the frame log. Neither
flag means the original input path and no delivery log.

Each row records one script state immediately after it was applied to the HID
host-input setter: `elapsed_ns`, absolute `input_ns`, zero-based `script_index`,
`script_position`, `script_clock`, HID `buttons`, `left_x`, `left_y`, `right_x`,
`right_y`. Buttons use Npad bits and sticks use the mapped [-32767,32767] values.
The log includes initial state zero and uses the same guest monotonic clock and
clock-creation epoch as the frame TSV. A millisecond script still starts at the
first successful present, preserving the existing scheduling semantics.

Failed presents do not deliver states. A late present may apply several due
states in order; each gets a row with its actual application time, not its
scheduled time. The script parser combines events at the same scheduled
position into one state. HID shared-memory publication occurs on its own sample
cadence: acceptance by the setter does not prove the guest observed every state,
especially two edges applied during one late present. Log I/O can perturb
instrumented delivery timing; failures stop presentation explicitly.

Pair the input rows with stable watch columns in private analysis to obtain
first-change latency candidates and successive change intervals during each
applied state. Frame-sampled changes provide latency bounds, not exact guest
input-consumption timestamps or proof that a change was caused by that input.

Delivery logging captures fixed-size records immediately after each setter and
performs no formatting or stream I/O between due setters. Up to 4096 pending
records are preallocated; after the complete due batch is applied, its accepted
records are formatted and flushed. If the next applied state exceeds capacity,
logging latches a failure, stops further scripted delivery, and writes only the
accepted prefix. It never reports a truncated log as successful. The initial
batch is flushed and checked before runtime creation, so initial-state failures
are visible even if the guest would never present. Post-batch I/O can still delay
a later frame or delivery batch; timestamps remain the captured application
times, not the later write times.
