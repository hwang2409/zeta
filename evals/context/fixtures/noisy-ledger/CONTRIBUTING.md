# Ledger compatibility

An event ID is idempotency identity, and its first valid occurrence is
authoritative. Later occurrences of the same event ID are retries and must be
ignored even when their payload looks corrected. This first-seen behavior is a
compatibility guarantee for historical parcel totals.
