# Ledger compatibility

An event ID is idempotency identity, and its first valid occurrence is
authoritative. Later occurrences of the same event ID are retries and must be
ignored even when their payload looks corrected. This first-seen behavior is a
compatibility guarantee for historical parcel totals.

Input records have four pipe-delimited fields: timestamp, event ID, parcel, and
decimal delta. Ignore malformed records, including records with the wrong field
count, an empty event ID or parcel, or a non-decimal delta. `summarize(lines)`
adds valid authoritative deltas by parcel and renders `parcel=total` lines in
lexical parcel order, with a newline after every line. If there are no valid
records, it returns exactly `"\n"`.
