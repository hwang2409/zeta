# multiple accounts and subscription usage

## status

phase 1 account storage, selection, mutable display labels, and offline identity display are implemented. phase 2 verified explicit usage refresh is implemented. exhaustion recovery remains later work.

## problem and v1 outcome

henry has several claude and chatgpt/codex subscriptions. zeta currently has one
subscription oauth file per provider. when one subscription is exhausted, he must
leave zeta, inspect another client, log in again, and return. the goal is a safe,
visible account picker that lets him switch deliberately without restarting zeta.

v1 has three vertical phases:

1. **safe accounts and selection:** store named accounts, migrate the existing
   login without loss, support sequential login, a global default, and a
   per-session selection. all backend construction uses one session account context.
2. **usage and the workflow:** after terms review, refresh all stale accounts with
   one explicit `r`, show partial failures, switch the current session with
   `enter`, and ship the smallest `zeta serve` list/switch mapping with a real
   render-path test in the same change.
3. **exhaustion recovery:** recognize only verified provider signals, save attached
   usage, and offer a one-key switch or the picker. never replay the failed request
   and never switch automatically.

api-key auth and ollama are not account types. anthropic api-key auth remains an
explicit mode, and ollama has no oauth store ([`factory.py:41-73`](../src/zeta/providers/factory.py#L41-L73)).

## ground truth

`zeta login` currently builds a provider-specific browser pkce flow and stores one
credential path per provider ([`providers/login.py:50-73`](../src/zeta/providers/login.py#L50-L73)).
the oauth record has `version`, `access_token`, `refresh_token`, `expires_at`,
and optional offline `email` or `workspace` display metadata; account metadata
and usage must never contain token fields. the shared oauth store
refreshes and atomically replaces its file ([`oauth.py:499-605`](../src/zeta/oauth.py#L499-L605)).
The account store must use the stricter session-file pattern: pinned parent,
`O_NOFOLLOW | O_NONBLOCK`, regular-file and link-count checks, same-directory
exclusive temporary file, `fsync`, `os.replace`, and `0600`
([`session_files.py:105-185`](../src/zeta/core/session_files.py#L105-L185)).

A normal backend is constructed with a provider credential
([`factory.py:76-118`](../src/zeta/providers/factory.py#L76-L118)). a child inherits
its parent's backend unless it chooses another model, in which case it constructs
a backend at child start ([`runner.py:313-352`](../src/zeta/agent/runner.py#L313-L352)).
The session account context must replace these direct paths. automatic memory reconciliation
still exists; eviction summaries were removed in #529 and are not a backend in
this design. memory reconciliation is a session-owned background call and uses
the session's pinned account through the same context. explicit
`ZETA_ALLOW_API_KEY=1` selects api-key auth, not a named subscription account.

`/model` changes the session provider by constructing and applying a backend. the
tui implementation is [`slash_handlers/__init__.py:148-166`](../src/zeta/tui/slash_handlers/__init__.py#L148-L166);
the serve implementation is [`model_selection.py:37-72`](../src/zeta/server/model_selection.py#L37-L72).

anthropic's current bootstrap behavior is preserved: when the zeta OAuth file is
absent, `AnthropicCredentialStore.access_token()` calls `bootstrap()` and reads
Claude Code `claude_credentials`/`.credentials.json` or `credentials.json`, and
on macOS may read the `Claude Code-credentials` keychain item, before saving a
valid result into its zeta OAuth file ([`providers/anthropic.py:140-188,230-250`](../src/zeta/providers/anthropic.py#L140-L188),
[`oauth.py:569-581`](../src/zeta/oauth.py#L569-L581), `origin/main` at
`8d050961` or later). in the account model, the first such resolution is a
lifecycle-locked, atomic materialization of that credential as
`anthropic:default`, including its registry row and global default; it then uses
the normal account resolver. bootstrap failure remains a safe login-required
error, and no bootstrap token is copied into metadata or usage state.

## identity and storage

an `AccountId` is the immutable `(provider, id)` stored by sessions and
automations. provider is `anthropic` or `codex`. the id and mutable display label
use lower-case, case-insensitive slugs of 1–32 ascii letters, digits, `_`, or
`-`; `.` and `..` are invalid. rows display `provider:label`. labels are unique
per provider across both ids and labels, so a reference is never ambiguous.
provider-returned email, workspace, or account id is display metadata only. one
account owns one oauth token set, lifecycle lock, usage state, immutable id, and
mutable label.

```text
~/.zeta/
  accounts.json
  accounts/
    anthropic/<id>/oauth.json
    anthropic/<id>/usage.json
    codex/<id>/oauth.json
    codex/<id>/usage.json
```

`accounts.json` contains the schema and one global default per provider. it is
the sole owner of the account catalog and defaults. provider email or workspace
is retained only when it is already present in the OAuth response and is shown
as optional display metadata; it is never an account identifier. each
session's `meta.json` is the sole owner of which account that session uses and
of its per-provider native-context boundary; it also stores the session's
monotonic switch sequence. `oauth.json` contains the token fields plus optional
`email` and `workspace` display fields. `usage.json` contains
the normalized snapshot and refresh-attempt state. files are `0600`; directories
are `0700`. secrets, auth headers, authorization codes, provider bodies, and
exception text never enter logs, notices, model input, transcripts, tool results,
or serve responses. serve login already uses fixed safe error messages
([`server/login.py:72-97`](../src/zeta/server/login.py#L72-L97)).

## labels

`zeta login <provider>` asks for a display label; the first account for that
provider offers `default`. an explicit label is available with `--account`, and
replacing an existing label requires `--replace`. the immutable id is derived
from the label when the account is created.

`zeta accounts rename <provider>:<id-or-label> <new-label>` and `n` in
`/accounts` change only the registry label. credential and lock directories,
global defaults, saved session pins, and approved automation revisions keep the
same immutable `AccountId`. command arguments and `--account` accept either the
id or label.

## crash-safe legacy migration

migration runs under a durable migration lock. it is never a move of the only
copy. for each legacy `anthropic-oauth.json` or `codex-oauth.json`:

1. lock the registry and provider account lifecycle in the documented order:
   migration lock, registry lock, then provider lifecycle lock. read and validate
   the legacy file with the hardened reader.
2. create `accounts/`, then `accounts/<provider>/`, then
   `accounts/<provider>/default/`, using pinned no-follow descriptors. after each
   new directory entry, `fsync` its parent directory (including `accounts/` and
   `accounts/<provider>/`) before continuing. create the destination
   `oauth.json`, copy the validated token data with an exclusive temporary file,
   `fsync` the file, `fsync` its directory, and verify it is readable with the
   expected token schema.
3. atomically write the registry row and global default, `fsync` the registry file,
   then `fsync` the registry directory. this committed row is the authority.
4. only after every directory entry, the file, the registry, and the readability
   check are durable, unlink the legacy file, then `fsync` its parent directory.
   a successful migration has one writable owner.

at every crash boundary recovery takes the migration lock, reads the committed
registry, and chooses exactly one authority. a crash while creating directories
or before the destination file is durable leaves the legacy file authoritative;
partial empty directories are harmless and may be removed. a durable destination
without a registry row is an uncommitted duplicate, so the intact legacy file
remains authoritative and recovery either completes the registry commit after
revalidating both copies or removes only the duplicate. a crash after the
registry commit but before unlink leaves both copies and recovery keeps the
registry destination. a crash after unlink but before its parent `fsync` leaves
the durable registry destination authoritative; recovery fails closed if neither
complete authority is provable. it never deletes the only valid token.

if two zeta processes start migration, one owns the migration lock and the other
waits, rereads the committed registry, and does not copy or retire a second time.
login does not silently overwrite a named account.

## locking and mutation rules

one registry lock protects every registry read-modify-write: account creation,
replacement metadata, logout, and global default. one per-session metadata
transaction owns session selection and its switch sequence; it is not a registry
mutation. one per-account lifecycle lock covers login/replace, logout, token
refresh, and migration of that account. one per-account usage lock covers
snapshots and refresh-attempt state. lock ordering is registry lock before
lifecycle lock before usage lock; no operation takes them in reverse order.
lifecycle and usage lock files live in `~/.zeta/account-locks/<provider>/<name>/`,
not inside the deletable account directory, and are durable `0600` files using
the same process-safe exclusive primitive.

an explicit usage request acquires the stable lifecycle lock first, then the
usage lock, checks and records the cooldown, resolves or refreshes the token,
keeps both locks through the provider request and final `usage.json` write, and
releases them in reverse order. replace or logout waits for that request, then
invalidates the old account generation before changing or deleting credentials.
a result is attributed only to the generation that issued it; a replaced or
logged-out generation's in-flight result is discarded and cannot populate the
replacement. after logout no token from the old generation may be reused. a
process-death recovery treats an abandoned in-flight attempt as failed, keeps
the prior snapshot, and permits the next cooldown-governed request.

refresh-attempt state persists `last_started_at`, `last_finished_at`, outcome,
and safe error class. the 60-second cooldown is checked and written under the
usage lock before the request, so failed 429 and network attempts also consume
the cooldown across processes. a request that loses a race returns the existing
in-flight or cooldown result. tests must cover logout-vs-refresh,
concurrent-default-update, failed-refresh-cooldown, and login-replace-vs-request.

## login, logout, and commands

v1 exposes only these offline commands:

```text
zeta login anthropic --account work
zeta login codex --account personal
zeta accounts
zeta accounts --json
zeta accounts default anthropic:work
zeta accounts refresh
zeta accounts logout anthropic:work
```

`default` changes the global default, which a session uses only when it has no
saved selection at load (see the session-load rule above). `switch` is reserved for
an attached tui or serve session and always targets that session. there is no
legacy `--provider` alias. one login flow per provider is allowed at a time;
adding another account is sequential. an existing account requires `--replace`.
logout is local only in v1; remote revocation is later work.

## usage sources and verification gate

v1 claims usage at a glance only after terms review and verified endpoint
adapters. the official claude code and codex clients' subscription endpoints are
called with the same subscription credential. the exact candidate requests are
`GET https://api.anthropic.com/api/oauth/usage` (Anthropic),
`GET https://chatgpt.com/backend-api/wham/usage` (Codex ChatGPT backend), and
`GET https://chatgpt.com/api/codex/usage` (Codex alternate backend), as pinned in
[codex-usage-client] and [codex-usage-url]. zeta calls them only from explicit
user action, never background polling, and applies the persisted cooldown above.
if terms do not permit an endpoint, endpoint-backed v1 is not shippable; passive
or partial display is a preview and must not be presented as the requested v1.

before implementation, record redacted fixtures from real responses for every
accepted provider source. each fixture records the exact field name, accepted
scale, reset units, missing/null behavior, and exact exhaustion value. unsupported
variants remain `unknown`; no guessed conversion is allowed.

The 2026-10-22 release-gate fixtures in `tests/fixtures/usage/` verified these
sources:

- Anthropic `GET /api/oauth/usage` returned HTTP 200. `five_hour` and
  `seven_day` each contain `utilization` on a 0–100 percent scale and
  `resets_at` as an ISO 8601 timestamp with an offset. Both windows were present
  in the observed response; nullable optional fields are ignored, while a
  missing/null window or value is unknown and is not displayed. No exhausted
  response was observed, so no exhaustion value is verified.
- Codex `GET /backend-api/wham/usage` returned HTTP 200. Under `rate_limit`,
  `primary_window` and nullable `secondary_window` contain `used_percent` on a
  0–100 scale, `limit_window_seconds` and `reset_after_seconds` in seconds, and
  `reset_at` as Unix epoch seconds. The observed `secondary_window` was null and
  is omitted. The response also contained `allowed`, `limit_reached`, and
  nullable `rate_limit_reached_type`; the observed non-exhausted values were
  `true`, `false`, and null, so no exhausted value is verified.
- Codex `GET /api/codex/usage` returned HTTP 404. Its shape remains unknown and
  it is not a usage source.

The verified adapters therefore show Anthropic five-hour/seven-day windows and
the Codex primary window. Codex secondary remains unknown because the fixture's
`secondary_window` was null. A missing, null, malformed, non-finite, or
out-of-range value makes that window unknown.
Refresh attempts have a persisted 60-second per-account cooldown, including
failed attempts.

the codex client source verifies
`x-codex-primary-used-percent`, `x-codex-secondary-used-percent`, their window
and reset fields, the two usage paths, and
`x-codex-rate-limit-reached-type` ([codex-rate-limit-parser],
[codex-usage-client], [codex-usage-url], and [codex-usage-mapping]). the
Anthropic public reports [anthropic-usage-endpoint] and
[anthropic-usage-concurrency-report] establish the endpoint, OAuth beta header,
and observed `five_hour`/`seven_day` fields and 429 risk; the local probe
reported the same shape. the codex mappings and all normalized exhaustion
values remain unverified until phase-2 redacted fixtures, even where the pinned
client source shows the field names. the anthropic endpoint probe established
only that
`five_hour` and `seven_day` objects can contain numeric `utilization` and
`resets_at`; their scale and exhaustion fields are unverified until a redacted
fixture is recorded. anthropic response-header mappings are also unverified and
are not a v1 source.

normalize only verified data into this shape:

```json
{
  "version": 1,
  "captured_at": "2026-10-21T12:34:56Z",
  "source": "usage_endpoint",
  "windows": [{
    "kind": "five_hour | seven_day | primary | secondary | provider_name",
    "used_percent": 42.0,
    "window_minutes": 300,
    "resets_at": "2026-10-21T15:00:00Z"
  }]
}
```

`used_percent` is 0–100 only after the fixture verifies its scale. preserve the
provider window identifier; reject non-finite, out-of-range, missing, or unknown
values. snapshots show `now` through 15 minutes, then `stale`, or `usage unknown`.
A refresh preserves the old snapshot on auth, 429, malformed, or changed data and
shows a safe per-account failure.

## `/accounts` and selection

`/accounts` is a compact modal with every account listed, current/default markers,
all known windows, reset times, freshness, and safe failure state. `r` refreshes
**all stale accounts sequentially**, bounded to one request per account, and
renders successes and partial failures. it does not refresh only the highlighted
row. `enter` switches the current session account. `l` starts a named login, `x`
performs confirmed local logout, and `d` changes the global default. text and
screen-reader output include percent values and absolute local reset times; color
is not required.

there are two selections: `accounts.json` owns the global default per provider,
and that session's `meta.json` owns its selected account per provider. for every
session without a saved selection, new or resumed, session load resolves the
provider's current global default, saves it as the session's selection, and uses
that selection from then on. changing the global default never changes a loaded
session; only `/accounts` switch in that session does. if the default account no
longer exists at load, fail with a clear error that names `/accounts`; do not
silently choose another account. a resumed pre-rollout session therefore pins
the default at its first load after rollout. children and session-owned
background memory calls pin the session selection at creation; later
parent/default changes never retarget them. automations pin the approved
account and require re-approval to change it. serve methods require `session_id`
for a switch; list is account-scoped and never changes selection.

switching persists selection, the native-context boundary, and the per-session
switch sequence atomically in session metadata before replacing an idle backend.
the boundary is a monotonic sequence per provider. payload filtering rejects
native artifacts older than the boundary:
Anthropic thinking signatures/redacted thinking and Codex encrypted reasoning are
never made eligible again by switching back. on `a → b → a`, the third selection
gets a new boundary. if metadata persistence or backend construction fails,
restore the prior backend and selection; do not report success. test resume and
`a → b → a`.

all backend creation asks one session account context with explicit auth mode: session account
for interactive, child, automation, and memory calls; the global-default fallback
follows the session-load rule above; API-key mode only when explicitly enabled.
add attribution tests for primary, child, automation, and memory calls, including
resume of pre-rollout metadata without a saved selection.

## exhaustion recovery

classify exhaustion only from verified provider-specific fixture fields and
accepted values. attach the validated usage snapshot to the account. never treat
an unknown 429, missing field, or changed schema as exhaustion; retain the
existing bounded retry behavior. after a subscription-limit failure, offer one
key to switch when exactly one alternative has fresh, verified usage and is not
known exhausted. otherwise open `/accounts`. switching never retries or replays
the failed request.

## delivery order and cuts

phase 1 delivers storage, migration, locks, selection, account context, and native
boundaries. phase 2 delivers verified explicit refresh, `/accounts`, local logout
if needed for correction, and the smallest serve list/session-switch mapping with
a real render-path test in the same change. phase 3 delivers verified
exhaustion recovery. this ordering gives surface parity in the phase that adds
the surface.

later: passive header capture, concurrent login flows, the legacy cli alias,
remote revocation, rich gamma accounts ui or login protocol, automatic failover,
renaming, and any eviction-summary account path (eviction summaries were removed
in #529).

## open questions for henry

none are required to implement this v1. the selected product behavior is current
session plus an explicit global default that any session without a saved
selection pins at load, new or resumed; live children and other sessions are
never retargeted. terms approval and redacted provider
fixtures are release gates, not product questions.

[codex-rate-limit-parser]: https://github.com/openai/codex/blob/430a16799d90e296fb27b1e94f206023902b7e56/codex-rs/codex-api/src/rate_limits.rs#L15-L87
[codex-usage-client]: https://github.com/openai/codex/blob/430a16799d90e296fb27b1e94f206023902b7e56/codex-rs/backend-client/src/client/rate_limit_resets.rs#L15-L70
[codex-usage-url]: https://github.com/openai/codex/blob/430a16799d90e296fb27b1e94f206023902b7e56/codex-rs/backend-client/src/client/rate_limit_resets.rs#L100-L113
[codex-usage-mapping]: https://github.com/openai/codex/blob/430a16799d90e296fb27b1e94f206023902b7e56/codex-rs/backend-client/src/client.rs#L615-L750
[anthropic-usage-endpoint]: https://github.com/anthropics/claude-code/issues/31021
[anthropic-usage-concurrency-report]: https://github.com/anthropics/claude-code/issues/31319
