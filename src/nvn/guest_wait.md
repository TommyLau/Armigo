# Blocking from a direct entry

Use `GuestWaitScope` and `wait_guest_event` for fence, vsync, or acquire waits.
The scope is bound on the guest's host thread and retains references to its
actual kernel `ThreadRecord`, handle table, and a cancellation policy callback.
The caller keeps those objects alive until the scope is destroyed. Nested scopes
restore the previous binding; unbound calls report unsupported behavior.

The native runtime runs guests on separate host threads. Virtual-core bindings
are metadata, not an exclusive execution permit. The helper therefore leaves
bindings and thread lifecycle unchanged and uses the caller-aware kernel event
wait. That wait registers guest cancellation, pins the event, handles timeout
and host stop, and unregisters before returning. A plain condition-variable
wait would omit this kernel state.

The policy callback must return the runtime stop token only when interruption
is allowed. The process runtime defers host cancellation while an observed
mutex grant remains owned; its `ThreadBinding::owns_grant()` policy must also
apply here. Binding a raw stop token unconditionally is insufficient.

Integration point: the process service binding owns the scope for the entire
guest execution lifetime, after constructing its thread/cancellation state.
Its private current invocation pointer is set only around SVC dispatch and
cannot identify a direct NVN caller. Stage a never blocks and does not install
this scope. The runtime owner must add that binding when enabling asynchronous
subsystems; the private harness can bind the scope directly today.

Completion, timeout, guest cancellation, host cancellation, and unsupported
behavior remain distinct results. Publish resource readiness before signaling,
hold no lock needed by the producer during the wait, and avoid reset races.
Do not reclassify a completed wait merely because stop became requested later.
Never throw or jump across guest frames. In particular, a cancelled void
`nvnQueueFinish` cannot return as though completion occurred: its real backend
needs a defined safe shutdown continuation before enabling cancellable waits.
