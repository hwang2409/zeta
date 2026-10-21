# multiple accounts and subscription usage

## status

design only. no implementation is part of this document.

## problem and v1 outcome

henry has several claude and chatgpt/codex subscriptions. zeta currently has one
subscription oauth file per provider. when one subscription is exhausted, he must
leave zeta, inspect another client, log in again, and return. the goal is a safe,
visible account picker that lets him switch deliberately without restarting zeta.

v1 has three vertical phases:

1. **safe accounts and selection:** store named accounts, migrate the existing
   login without loss, support sequential login, a global default, and a
   per-session selection. all backend construction uses one account resolver.
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
the oauth record has `version`, `access_token`, `refresh_token`, and `expires_at`;
account metadata and usage must never contain token fields. the shared oauth store
refreshes and atomically replaces its file ([`oauth.py:499-605`](../src/zeta/oauth.py#L499-L605)).
The account store must use the stricter session-file pattern: pinned parent,
`O_NOFOLLOW | O_NONBLOCK`, regular-file and link-count checks, same-directory
exclusive temporary file, `fsync`, `os.replace`, and `0600`
([`session_files.py:105-185`](../src/zeta/core/session_files.py#L105-L185)).

A normal backend is constructed with a provider credential
([`factory.py:76-118`](../src/zeta/providers/factory.py#L76-L118)). a child inherits
its parent's backend unless it chooses another model, in which case it constructs
a backend at child start ([`runner.py:313-352`](../src/zeta/agent/runner.py#L313-L352)).
The resolver must replace these direct paths. automatic memory reconciliation
still exists; eviction summaries were removed in #529 and are not a backend in
this design. memory reconciliation is a session-owned background call and uses
the session's pinned account through the same resolver. explicit
`ZETA_ALLOW_API_KEY=1` selects api-key auth, not a named subscription account.

`/model` changes the session provider by constructing and applying a backend. the
tui implementation is [`slash_handlers/__init__.py:148-166`](../src/zeta/tui/slash_handlers/__init__.py#L148-L166);
the serve implementation is [`model_selection.py:37-72`](../src/zeta/server/model_selection.py#L37-L72).

## identity and storage

an `AccountId` is `(provider, name)`, displayed as `anthropic:work` or
`codex:personal`. provider is `anthropic` or `codex`. name is a lower-case,
case-insensitive slug of 1–32 ascii letters, digits, `_`, or `-`; `.` and `..`
are invalid. provider-returned email, workspace, or account id is display
metadata only. one account owns one oauth token set, lifecycle lock, usage state,
and local name.

```text
~/.zeta/
  accounts.json
  accounts/
    anthropic/<name>/oauth.json
    anthropic/<name>/usage.json
    codex/<name>/oauth.json
    codex/<name>/usage.json
```

`accounts.json` contains the schema, display metadata, per-provider global
 defaults, per-account selection metadata, and a monotonic switch sequence.
`oauth.json` contains only the existing four token fields. `usage.json` contains
the normalized snapshot and refresh-attempt state. files are `0600`; directories
are `0700`. secrets, auth headers, authorization codes, provider bodies, and
exception text never enter logs, notices, model input, transcripts, tool results,
or serve responses. serve login already uses fixed safe error messages
([`server/login.py:72-97`](../src/zeta/server/login.py#L72-L97)).

## crash-safe legacy migration

migration runs under a durable migration lock. it is never a move of the only
copy. for each legacy `anthropic-oauth.json` or `codex-oauth.json`:

1. lock the registry and provider account lifecycle in the documented order:
   migration lock, registry lock, then provider lifecycle lock. read and validate
   the legacy file with the hardened reader.
2. create the destination account directory and copy the validated token data to
   `accounts/<provider>/default/oauth.json` using an exclusive temporary file.
   `fsync` the file, `fsync` the destination directory, and verify that the
   destination is readable and has the expected token schema.
3. atomically write the registry row and global default, `fsync` the registry file,
   then `fsync` the registry directory. this committed row is the authority.
4. unlink the legacy file only after the registry commit, then `fsync` the legacy
   parent directory. a successful migration has one writable owner.

at every crash boundary recovery takes the migration lock, reads the committed
registry, and chooses exactly one authority: a committed row points to the
validated destination; without that row the validated destination is an
uncommitted duplicate and the intact legacy file remains authoritative. recovery
completes the registry commit from the duplicate, or removes only the duplicate
after verifying the legacy source. it never deletes the only valid token. if the
legacy source disappeared but the destination and registry are both durable,
the committed registry remains authoritative. if they are not both durable,
recovery fails closed with a repair message rather than guessing.

if two zeta processes start migration, one owns the migration lock and the other
waits, rereads the committed registry, and does not copy or retire a second time.
login does not silently overwrite a named account.

## locking and mutation rules

one registry lock protects every registry read-modify-write: account creation,
replacement metadata, logout, global default, and session-selection metadata.
one per-account lifecycle lock covers login/replace, logout, token refresh, and
migration of that account. one per-account usage lock covers snapshots and
refresh-attempt state. lock ordering is registry lock before lifecycle lock before
usage lock; no operation takes them in reverse order. lock files are durable
`0600` files and use the same process-safe exclusive locking primitive.

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

`default` changes the global default for new sessions. `switch` is reserved for
an attached tui or serve session and always targets that session. there is no
legacy `--provider` alias. one login flow per provider is allowed at a time;
adding another account is sequential. an existing account requires confirmation
in a terminal and `--replace` in non-interactive use. logout is local only in
v1; remote revocation and rename are later work.

## usage sources and verification gate

v1 claims usage at a glance only after terms review and verified endpoint
adapters. the official claude code and codex clients' subscription endpoints are
called with the same subscription credential. zeta calls them only from explicit
user action, never background polling, and applies the persisted cooldown above.
if terms do not permit an endpoint, endpoint-backed v1 is not shippable; passive
or partial display is a preview and must not be presented as the requested v1.

before implementation, record redacted fixtures from real responses for every
accepted provider source. each fixture records the exact field name, accepted
scale, reset units, missing/null behavior, and exact exhaustion value. unsupported
variants remain `unknown`; no guessed conversion is allowed. the codex client
source verifies `x-codex-primary-used-percent`, `x-codex-secondary-used-percent`,
the usage paths, and `x-codex-rate-limit-reached-type` (pinned sources in the
original proposal). the anthropic endpoint probe established only that
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

there are two selections: a global default per provider for new sessions, and a
per-session selected account per provider. a resumed session keeps its selection.
children and session-owned background memory calls pin the session selection at
creation; later parent/default changes never retarget them. automations pin the
approved account and require re-approval to change it. serve methods require
`session_id` for a switch; list is account-scoped and never changes selection.

switching persists selection and the native-context boundary atomically in session
metadata before replacing an idle backend. the boundary is a monotonic sequence
per provider. payload filtering rejects native artifacts older than the boundary:
Anthropic thinking signatures/redacted thinking and Codex encrypted reasoning are
never made eligible again by switching back. on `a → b → a`, the third selection
gets a new boundary. if metadata persistence or backend construction fails,
restore the prior backend and selection; do not report success. test resume and
`a → b → a`.

all backend creation uses one resolver with explicit auth mode: session account
for interactive, child, automation, and memory calls; global default only at new
session creation; API-key mode only when explicitly enabled. add attribution tests
for primary, child, automation, and memory calls.

## exhaustion recovery

classify exhaustion only from verified provider-specific fixture fields and
accepted values. attach the validated usage snapshot to the account. never treat
an unknown 429, missing field, or changed schema as exhaustion; retain the
existing bounded retry behavior. after a subscription-limit failure, offer one
key to switch when exactly one alternative has fresh, verified usage and is not
known exhausted. otherwise open `/accounts`. switching never retries or replays
the failed request.

## delivery order and cuts

phase 1 delivers storage, migration, locks, selection, resolver, and native
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
session plus an explicit global default for future sessions; live children and
other sessions are never retargeted. terms approval and redacted provider
fixtures are release gates, not product questions.
