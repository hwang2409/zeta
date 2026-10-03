# Incident triage note

The cache implementation is a red herring for this incident. Preserve its
ordering behavior and follow the timestamp-unit evidence to find the readiness
failure.

`created_at`, `ttl`, and an injected `now` callback all use integer monotonic
milliseconds. A job is ready when `now >= created_at + ttl`. When no callback is
supplied, the scheduler must obtain the current value with
`time.monotonic_ns() // 1_000_000`; wall-clock time is not compatible with
persisted timestamps.
