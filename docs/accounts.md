# Multiple accounts and subscription usage

## Status

Design only. No implementation is part of this document.

## Problem and goal

A user can have several Claude and ChatGPT/Codex subscriptions, but zeta has one
OAuth file per provider. When one subscription reaches its limit, the user must
leave zeta, inspect accounts in another client, log in again, and then return.
The goal is to make account availability visible and make a deliberate account
switch take effect without restarting zeta.

The v1 outcome is:

- store several named subscription accounts for each provider;
- show current and stale usage data in `/accounts`;
- log in, log out, refresh, and switch from `/accounts`;
- provide the same operations through `zeta accounts` and the serve protocol;
- identify a subscription-limit failure and offer a one-key switch; and
- never switch accounts automatically.

API-key auth and Ollama are not account types in v1. Anthropic API-key auth is a
separate, explicit mode today, while Ollama has no OAuth credential store
([`factory.py:34-73`](../src/zeta/providers/factory.py#L34-L73),
[`factory.py:119-138`](../src/zeta/providers/factory.py#L119-L138)).

## Ground truth

### Current zeta authentication

`zeta login` currently accepts `--provider anthropic|codex`, defaults to
Anthropic, and runs a browser PKCE flow
([`cli/main.py:262-269`](../src/zeta/cli/main.py#L262-L269),
[`cli/main.py:377-380`](../src/zeta/cli/main.py#L377-L380)). The provider wiring
uses one fixed path per provider, `~/.zeta/anthropic-oauth.json` or
`~/.zeta/codex-oauth.json`; Anthropic stores no account handle, while Codex
extracts an account ID from the access token
([`providers/login.py:50-71`](../src/zeta/providers/login.py#L50-L71)). A
key-name-only inspection of Henry's two current files on 2026-10-21 found the
same four keys that zeta serializes: `version`, `access_token`, `refresh_token`,
and `expires_at`; no token value was read into this document
([`oauth.py:458-464`](../src/zeta/oauth.py#L458-L464)).

The shared store serializes refreshes with process and asyncio locks. It reads a
valid access token, refreshes an expired token, and atomically replaces the file
with mode `0600`
([`oauth.py:499-560`](../src/zeta/oauth.py#L499-L560),
[`oauth.py:569-605`](../src/zeta/oauth.py#L569-L605)). Anthropic and Codex each
implement provider-specific refresh requests
([`anthropic.py:252-277`](../src/zeta/providers/anthropic.py#L252-L277),
[`codex.py:119-166`](../src/zeta/providers/codex.py#L119-L166)). Anthropic can
bootstrap from Claude Code files or the macOS keychain only when the normal home
rules permit it
([`anthropic.py:140-188`](../src/zeta/providers/anthropic.py#L140-L188),
[`anthropic.py:230-250`](../src/zeta/providers/anthropic.py#L230-L250)).

A backend is built with that provider's single store. Thus an ordinary session
uses the credential selected when its backend is constructed
([`factory.py:76-118`](../src/zeta/providers/factory.py#L76-L118)). A child with
no model reuses its parent's backend; a child with an explicit model constructs
a new backend for that model's provider
([`agent/runner.py:313-352`](../src/zeta/agent/runner.py#L313-L352)). An
automation creates a new backend from the provider and model saved in its
session metadata
([`runtime/unattended.py:17-33`](../src/zeta/runtime/unattended.py#L17-L33)).
None of these paths has an account selector today.

`ZETA_ANTHROPIC_OAUTH_COMPAT=1` changes Anthropic's subscription-OAuth request
compatibility behavior only when the credential is an `AnthropicCredentialStore`
([`anthropic.py:125-129`](../src/zeta/providers/anthropic.py#L125-L129)). It does
not select an account. `/model` can change the session provider by constructing
and applying a backend for the selected model
([`core/slash.py:1030-1068`](../src/zeta/core/slash.py#L1030-L1068),
[`server/model_selection.py:37-72`](../src/zeta/server/model_selection.py#L37-L72)).

The serve protocol is bounded JSON-RPC 2.0. Protocol 1.1 negotiates optional
features in `hello`
([`server/protocol.py:1-26`](../src/zeta/server/protocol.py#L1-L26),
[`serve-protocol.md:14-31`](serve-protocol.md#L14-L31)). Its current login
surface has `login_providers`, `login_start`, `login_status`, and `login_cancel`.
The server allows only one pending login task per provider and reports only a
provider-level `credentials_present` flag
([`server/login.py:15-70`](../src/zeta/server/login.py#L15-L70)).

### Usage sources

#### Claude subscription OAuth

Anthropic's help center confirms that paid Claude and Claude Code usage is
shared, that usage has a five-hour session window, and that other limits can
apply over longer periods.[^anthropic-limits] Claude Code's public changelog
records `/usage` as the command for plan-limit information.[^claude-changelog]

A single read-only probe against
`GET https://api.anthropic.com/api/oauth/usage` on 2026-10-21, using Henry's
existing zeta OAuth credential and the `anthropic-beta: oauth-2025-04-20`
header, returned HTTP 200. The response had `five_hour` and `seven_day` objects;
each had numeric `utilization` and an ISO-8601 string `resets_at`. The probe
printed only status, content type, key names, and value types. It did not print
or persist a token or usage value. This matches public reports of the same
endpoint and fields, including reports that excessive calls can return 429 and
break `/usage`.[^anthropic-usage-endpoint]

Normal Claude responses have also been reported to carry
`anthropic-ratelimit-unified-5h-utilization`,
`anthropic-ratelimit-unified-5h-reset`,
`anthropic-ratelimit-unified-7d-utilization`,
`anthropic-ratelimit-unified-7d-reset`, and status/fallback headers.[^anthropic-headers]
Anthropic does not publish these subscription headers as a stable API contract,
and this design did not spend a model request to probe them. Their exact value
scale, availability on every response, model-specific variants, and long-term
support are **unverified**. Therefore they are an opportunistic passive source,
not the only source.

#### ChatGPT/Codex OAuth

The open-source Codex CLI parses response headers named
`x-codex-primary-used-percent`, `x-codex-primary-window-minutes`, and
`x-codex-primary-reset-at`, plus equivalent `secondary` headers. It maps them
to windows with percent used, duration in minutes, and a Unix reset timestamp.
It also parses named limit families and optional credit headers
([Codex `rate_limits.rs:15-87`][codex-rate-limit-parser]). This is a passive
source on normal responses.

Codex also makes an authenticated GET to `/wham/usage` for a ChatGPT backend
(or `/api/codex/usage` for the other path) and maps primary and secondary
windows from the response
([Codex `rate_limit_resets.rs:15-70`][codex-usage-client],
[Codex `rate_limit_resets.rs:100-113`][codex-usage-url],
[Codex `backend-client/client.rs:615-750`][codex-usage-mapping]). Its status
module turns those snapshots into `/status` rows with percent used, reset text,
and progress bars, and marks data stale after 15 minutes
([Codex `tui/status/rate_limits.rs:1-60`][codex-status-stale]). These files are
primary evidence for the client behavior, but the ChatGPT backend endpoint and
header family are not a published stable public API.

#### Polling policy and terms risk

Passive capture has no extra request cost: zeta reads usage metadata from the
response headers of a request it already made. Explicit refresh makes one
additional authenticated GET for the selected account. It must happen only on
user action in v1, with a per-account 60-second local cooldown and request
coalescing across processes. No timer polls in the background. Public reports
show that the Anthropic usage endpoint itself can return 429 under concurrent
use.[^anthropic-usage-endpoint]

Both usage endpoints and both subscription header families are undocumented or
not guaranteed as public interfaces. Their availability, schema, and permitted
client behavior can change. Before shipping endpoint refresh, maintainers must
confirm that current provider terms permit this use. If that review is not
complete, ship passive capture and display first, and label refresh
`unavailable`; do not imitate private-client fingerprints or bypass a 401/403/429.

## Proposed model

### Identity

An `AccountId` is `(provider, name)`, displayed as `anthropic:work` or
`codex:personal`.

- `provider` is `anthropic` or `codex`.
- `name` is a user-chosen, case-insensitive slug: 1-32 ASCII letters, digits,
  `_`, or `-`; it cannot be `.` or `..`.
- The canonical form is lower-case. Duplicate canonical names are rejected.
- Provider-returned email, workspace, or account ID is optional display
  metadata, not the local identity. Renaming an account changes only its local
  name.

One account owns one OAuth token set, one refresh lock, and one usage snapshot.
Refresh and logout operate on that account only.

### Storage

Use one account registry and one directory per account:

```text
~/.zeta/
  accounts.json
  accounts/
    anthropic/
      work/
        oauth.json
        usage.json
    codex/
      personal/
        oauth.json
        usage.json
```

`accounts.json` contains schema version, non-secret display metadata, and the
default account for each provider. `oauth.json` contains only the existing four
token fields. `usage.json` contains only normalized usage snapshots. All three
file types and lock files use mode `0600`; directories use `0700`.

The account store must use zeta's hardened file pattern: pin the parent
directory, open with `O_NOFOLLOW | O_NONBLOCK`, require a regular file with one
link, write a same-directory exclusive temporary file, `fsync`, and publish
with `os.replace`. Zeta already uses this pattern for session files
([`core/session_files.py:110-185`](../src/zeta/core/session_files.py#L110-L185)).
The current OAuth store already uses a same-directory temporary file,
`fsync`, `os.replace`, and `0600`, but its reads do not yet use the full
`O_NOFOLLOW` validation
([`oauth.py:523-562`](../src/zeta/oauth.py#L523-L562)); the account store must
not copy that gap.

Account metadata and usage never contain access or refresh tokens. Tokens,
authorization codes, provider response bodies from auth failures, and auth
headers must never enter logs, notices, model input, tool results, transcripts,
or serve responses. The current serve login path already replaces provider
exception text with fixed safe messages because responses can contain tokens or
callback codes
([`server/login.py:72-97`](../src/zeta/server/login.py#L72-L97)). The account
implementation must keep that boundary.

### Implementation seam

Put validation, migration, locking, refresh, selection, and snapshot persistence
behind one account-registry module. TUI, CLI, backends, automations, and serve
call its typed interface; none constructs account paths or parses account files.
Anthropic and Codex usage adapters plug into one internal normalization seam,
because two real provider formats vary. This keeps provider field changes out
of every caller and makes the registry interface the test surface.

### One-time migration

On the first account-store open, under one migration lock:

1. If `accounts.json` exists, do nothing.
2. For each legacy OAuth file that exists, validate it with the hardened reader.
3. Move it with `os.replace` to `accounts/<provider>/default/oauth.json`.
4. Write `accounts.json` with that account as the provider default.
5. Recover safely after a crash by treating a valid destination without a
   registry row as an incomplete migration and adding the row.

The old files are moved, not copied and not retained as fallback paths. Keeping
two writable token locations would create split-brain refreshes and make logout
ambiguous. A successful migration has one credential owner. If migration fails,
leave the legacy source intact and fail with a recovery message; never truncate
or delete it. External Claude Code credential bootstrap can still seed an
explicit account when no Anthropic account exists, but it must not silently
overwrite a named account.

## Login and logout

### CLI

The intended commands are:

```text
zeta login anthropic --account work
zeta login codex --account personal
zeta accounts
zeta accounts --json
zeta accounts switch anthropic:work
zeta accounts refresh [anthropic:work]
zeta accounts logout anthropic:work
```

`zeta login --provider anthropic` remains accepted only through the migration
release and resolves to `zeta login anthropic --account default`; remove the old
shape in the next breaking CLI release. This is command-line compatibility,
not a second storage path.

A login writes only the target account. If it already exists, ask before
replacing it in an interactive terminal; require `--replace` in non-interactive
use. A successful new login becomes that provider's default only when it is the
first account, or when `--make-default` is supplied.

### TUI

`/accounts login` opens a provider and account-name form, then starts the browser
flow. Login flows are keyed by `AccountId`, not only provider. Today each provider
has a fixed callback URI and the login flow validates its port and path
([`providers/login.py:43-70`](../src/zeta/providers/login.py#L43-L70)). The new
coordinator uses one shared callback listener per provider and routes callbacks
by PKCE `state`, so several accounts can be pending at once without binding that
port more than once. Each flow has independent verifier, state, deadline,
cancellation, result, and target store. Disconnect cancels only flows owned by
that client.

`/accounts logout` asks for confirmation, revokes remotely only if the provider
has a documented supported revocation operation, then removes the local token
file. Remote revocation is **unverified for both subscription flows** and is not
required for v1. Logging out an active account does not interrupt an in-flight
request. It blocks new requests from that account, and the next request asks the
user to select or log into another account.

## Usage snapshot and capture

Persist this provider-neutral shape per account:

```json
{
  "version": 1,
  "captured_at": "2026-10-21T12:34:56Z",
  "source": "response_headers | usage_endpoint",
  "windows": [
    {
      "kind": "five_hour | seven_day | primary | secondary | provider_name",
      "used_percent": 42.0,
      "window_minutes": 300,
      "resets_at": "2026-10-21T15:00:00Z"
    }
  ]
}
```

Unknown fields stay unknown; do not infer window names from duration alone.
Normalize percentages to 0-100, retain the provider's raw window identifier,
and reject non-finite or out-of-range values. `captured_at` is zeta's receipt
time. `resets_at` is UTC internally and local time only at rendering.

After response headers arrive, the backend parses recognized usage fields and
atomically replaces that account's snapshot. Missing headers do not erase a
newer endpoint snapshot. A snapshot update is internal application state: it
must not create transcript messages, model context, or noisy TUI events.

An explicit refresh obtains a current access token through that account's
normal refresh lock, makes one provider usage GET, validates the bounded JSON
response, and saves it with `source: usage_endpoint`. Authentication failure
marks the account `login required`; 429 preserves the old snapshot and reports
that refresh was rate-limited. A malformed or changed response also preserves
the old snapshot and reports `usage unavailable`; it never logs the body.

Display freshness as:

- `now` through 15 minutes: no label;
- over 15 minutes: `stale · updated <relative time>`;
- no snapshot: `usage unknown`.

The 15-minute boundary aligns with the current Codex UI behavior, not with a
provider guarantee ([Codex `tui/status/rate_limits.rs:48-60`][codex-status-stale]).
It is a display rule only. V1 does not poll automatically.

## `/accounts` user experience

`/accounts` opens a compact modal instead of putting secrets or a large table in
chat history:

```text
accounts                                      r refresh   l login

  claude
  ● anthropic:work       5h  [██████░░░░]  62%  resets 2h 11m
                         7d  [████░░░░░░]  38%  resets fri 09:00
    anthropic:personal   5h  [█░░░░░░░░░]   8%  resets 4h 42m
                         stale · updated 27m ago

  codex
  ● codex:personal       5h  [████████░░]  81%  resets 14:20
                         7d  [█████░░░░░]  47%  resets mon 10:00

  enter switch   l login   x log out   r refresh   esc close
```

The marker means active for this session, not global default. Use provider names
users recognize (`claude`, `codex`) as headings and canonical IDs in detail and
CLI output. Percent text accompanies every bar. Reset labels include an
absolute local time when a relative label could cross a day boundary. Keyboard
and screen-reader text do not rely on color.

Actions:

- `enter`: switch the highlighted account for this session;
- `l`: log in a new named account;
- `x`: confirm and log out;
- `r`: explicitly refresh highlighted usage; and
- `d`: make highlighted account the provider default for new sessions.

`zeta accounts` renders the same normalized data as text. `--json` returns
versioned data with canonical IDs, active/default flags, snapshot timestamps,
sources, and windows, but never token fields or token-file paths.

## Selection and switching semantics

There are two selections:

1. a global default per provider, used only when a new interactive session is
   created; and
2. a per-session selected account per provider, persisted in session metadata.

A resumed session keeps its own selection. Changing a global default does not
retarget existing sessions. `/model` uses the session's selected account for
the model's provider. If the session has no selection for that provider, it
copies the current global default into session metadata before constructing the
backend.

Switching is serialized at the provider-request boundary. It records the new
session selection and replaces the idle backend's credential binding. An
already-started HTTP request and its retries remain on the old account. The
next new provider request uses the new account. The notice says exactly which
turn boundary applies; it never claims an in-flight request moved.

A child inherits and pins the parent's selected account for the child's
provider at child start. A child started with a model from another provider
pins that session's selected account for the other provider. Existing children
do not follow later parent or global switches. This extends today's behavior,
where an inherited child shares the backend and an explicit-model child builds
one at start ([`agent/runner.py:313-352`](../src/zeta/agent/runner.py#L313-L352)).

An automation revision stores an explicit account ID for its provider at
approval time and each run pins it for the full run. It does not follow the
global default. If that account is missing or logged out, the run fails before
creating a model turn with a safe actionable receipt. This extends today's
persisted provider/model selection
([`automations/models.py:19-25`](../src/zeta/automations/models.py#L19-L25),
[`runtime/unattended.py:24-33`](../src/zeta/runtime/unattended.py#L24-L33)).
Editing the account requires re-approval because it changes the paid identity.

### Cache and provider-native continuity

Changing credentials can make provider-side prompt caches cold. Zeta must not
promise cache reuse across accounts, and the usage/status display must not
attribute a post-switch request to the old account.

Zeta replays signed Anthropic thinking blocks when they have signatures
([`anthropic_payload.py:94-123`](../src/zeta/providers/anthropic_payload.py#L94-L123))
and replays Codex reasoning `encrypted_content` from the same internal signature
field
([`codex_payload.py:49-82`](../src/zeta/providers/codex_payload.py#L49-L82)).
OpenAI issue reports show encrypted reasoning can fail across an organization or
provider boundary.[^codex-encrypted] Anthropic documents that thinking blocks
must be passed back unmodified, but does not document whether signatures are
portable across subscription accounts.[^anthropic-thinking] No primary source
found by this design establishes same-provider, cross-account portability for
either artifact. It is **unverified**.

V1 therefore creates a native-context boundary on account switch: earlier
provider-native signatures, redacted thinking data, and encrypted reasoning are
not sent after the boundary. Visible assistant text, tool calls/results, user
messages, and Codex reasoning summaries remain. The first request on the new
account is expected to have a cold prompt cache. If a provider later documents
cross-account portability, this conservative boundary can be relaxed with a
focused test; v1 must not learn by sending opaque artifacts and retrying after a
400.

## Serve protocol for Gamma

Add optional protocol-1.1 feature `accounts`. A client must request it in
`hello`; old clients see no new methods or fields. This follows the current
feature-negotiation contract
([`server/protocol.py:9-25`](../src/zeta/server/protocol.py#L9-L25),
[`serve-protocol.md:23-31`](serve-protocol.md#L23-L31)).

Methods:

- `accounts_list`: safe metadata, active/default flags, and normalized usage;
- `accounts_switch {account_id, session_id?}`: session selection only;
- `accounts_refresh {account_id}`: explicit, cooldown-bound refresh;
- `accounts_logout {account_id}`: confirmed local logout contract;
- `account_login_start {provider, account}`;
- `account_login_status {login_id}`; and
- `account_login_cancel {login_id}`.

Login start returns an opaque `login_id` and authorization URL. Status and
cancel use `login_id`, which allows concurrent flows. No response contains an
access token, refresh token, authorization code, auth header, raw provider
error/body, or local secret path. Mutations reject an active turn except
`accounts_switch`, which may be accepted during a turn but is queued for the
next provider request and reports `effective_after: current_request`.

Gamma should render the server data; it must not read `~/.zeta` directly or
call provider endpoints itself.

## Out-of-usage handling

Today, provider retry policy treats every HTTP 429 as retryable, with at most
five attempts and a shared 60-second retry budget
([`retry_policy.py:15-33`](../src/zeta/providers/retry_policy.py#L15-L33),
[`retry_policy.py:62-137`](../src/zeta/providers/retry_policy.py#L62-L137),
[`retry_policy.py:188-223`](../src/zeta/providers/retry_policy.py#L188-L223)).
That does not distinguish a short service throttle from an exhausted
subscription window.

Add a provider-specific classifier before generic retry handling. It may use a
stable provider error code, a `rate-limit-reached` header, and a known future
reset. It must not classify from free-form English alone. The result is one of:

- `transient_rate_limit`: keep existing bounded retry behavior;
- `subscription_usage_exhausted`: do not retry the same account; or
- `unknown_429`: keep the existing bounded retry behavior.

On `subscription_usage_exhausted`, save any usage metadata on the response and
show:

```text
anthropic:work has reached its 5h limit (resets in 1h 18m).
anthropic:personal has 74% remaining. press s to switch for the next request.
```

Only accounts with a fresh snapshot and every known blocking window below 100%
are described as having remaining usage. Stale or unknown accounts are listed
as `usage unknown`, never promised as available. `s` immediately switches to
the named recommended alternative for the next request. If zeta cannot
recommend exactly one account, it offers the ordinary picker instead and does
not call that a one-key switch. The failed request is not silently replayed
because tools or remote side effects may already have occurred. The user
explicitly resubmits or continues.

Automatic failover is later work, not v1. It risks moving private prompts to a
different employer/workspace, breaking native reasoning continuity, losing
prompt-cache reuse, double-running a partially accepted request, and making
cost attribution surprising. Henry must opt into that policy explicitly.

## Delivery phases

Each phase is a separate, independently reviewable PR.

1. **Account store and migration.** Add account IDs, hardened storage, one-time
   migration, account-aware refresh locks, and CLI list/login/logout/switch.
   Named tests: `test_legacy_oauth_files_move_once`,
   `test_migration_recovers_destination_without_registry_row`,
   `test_account_store_rejects_symlink_and_hardlink`,
   `test_refresh_is_serialized_per_account`,
   `test_cli_never_serializes_token_fields`.
2. **Backend and session selection.** Bind account IDs to backends, persist
   per-session selections, define next-request switching, pin children and
   automations, and create the native-context boundary. Named tests:
   `test_switch_applies_after_inflight_request`,
   `test_resumed_session_keeps_account`,
   `test_child_pins_account_at_start`,
   `test_automation_revision_pins_account`,
   `test_account_switch_omits_preboundary_native_reasoning`.
3. **Passive usage capture.** Parse bounded provider headers, normalize and
   persist snapshots, and mark staleness. Named tests:
   `test_anthropic_usage_headers_update_account`,
   `test_codex_primary_secondary_headers_update_account`,
   `test_missing_headers_do_not_erase_endpoint_snapshot`,
   `test_usage_snapshot_rejects_nonfinite_percent`,
   `test_usage_update_never_enters_transcript`.
4. **Accounts UX and explicit refresh.** Add `/accounts`, `zeta accounts`, the
   user-triggered endpoints with cooldown/coalescing, login concurrency, and
   accessible rendering. Gate endpoint calls on the terms review. Named tests:
   `test_accounts_modal_actions`,
   `test_accounts_text_and_json_redact_secrets`,
   `test_same_provider_logins_route_by_state`,
   `test_refresh_cooldown_coalesces_processes`,
   `test_refresh_429_preserves_snapshot`.
5. **Serve/Gamma surface.** Negotiate `accounts`, add bounded methods and safe
   errors, and keep legacy clients unchanged. Named tests:
   `test_accounts_feature_is_negotiated`,
   `test_accounts_methods_hidden_from_legacy_client`,
   `test_account_login_flows_are_keyed_by_login_id`,
   `test_accounts_wire_never_contains_secrets`,
   `test_serve_switch_reports_effective_boundary`.
6. **Exhaustion recovery.** Distinguish confirmed subscription exhaustion,
   show alternatives, and provide the one-key switch path without replay.
   Named tests: `test_transient_429_keeps_retry_policy`,
   `test_subscription_exhaustion_does_not_retry`,
   `test_exhaustion_offer_requires_fresh_available_usage`,
   `test_switch_does_not_replay_failed_request`,
   `test_unknown_429_does_not_claim_exhaustion`.

## Open questions for Henry

1. **Selection scope:** should account switches be per session plus a separate
   default, or global for all live sessions? **Recommendation:** per session;
   global switching can silently move running children and unrelated work.
2. **Automation identity:** should an automation pin an account in its approved
   revision? **Recommendation:** yes; changing the paid identity must require
   re-approval and must not follow a mutable default.
3. **Legacy CLI compatibility:** keep `zeta login --provider ...` for one release
   or remove it immediately? **Recommendation:** keep only the command alias for
   one release, but migrate to one storage layout immediately.
4. **Provider usage endpoints:** ship explicit refresh after a terms review, or
   ship passive-only first? **Recommendation:** passive-first if the review is
   not complete; never background-poll undocumented subscription endpoints.
5. **Logout meaning:** local credential removal only, or require remote
   revocation? **Recommendation:** local removal in v1, clearly labeled, until a
   documented revocation operation is verified.
6. **Automatic failover:** should zeta ever choose another account without a
   confirmation? **Recommendation:** no in v1. Revisit only as an explicit
   per-session opt-in after provider-native continuity and partial-request
   behavior are verified.
7. **Account labels:** user-defined local names only, or automatically expose
   provider email/workspace names? **Recommendation:** local names are primary;
   show provider metadata only when it is safely available and let the user
   hide it.

## Verified and unverified source summary

Verified from zeta source: the one-file-per-provider layout, token key schema,
refresh locking, backend construction, child/automation credential selection,
`/model` provider switching, OAuth compatibility flag, serve feature
negotiation, login task limit, native reasoning replay, and generic 429 retry
behavior (see the linked file-and-line citations above).

Verified from a read-only Anthropic probe plus public reports: the OAuth usage
GET accepts the existing subscription credential and returns `five_hour` and
`seven_day` utilization/reset objects. Verified from open-source Codex source:
normal-response header names and parsing, the authenticated usage GET paths,
window fields, and the client's 15-minute stale display rule.

Unverified: Anthropic subscription headers as a stable contract; the exact
scale and universal presence of those headers; provider permission for
third-party periodic use of either private usage endpoint; remote logout
revocation; and cross-account portability of Anthropic thinking signatures or
Codex encrypted reasoning. The design treats all of these conservatively.

[^anthropic-limits]:
    Anthropic Help Center,
    ["Use Claude Code with your Pro or Max plan"](https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan)
    and
    ["What is a limit reset?"](https://support.claude.com/en/articles/17007452-what-is-a-limit-reset).
[^claude-changelog]:
    Anthropic, [Claude Code changelog](https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md)
    (search for `/usage`; the repository publishes the product changelog but not
    the client source).
[^anthropic-usage-endpoint]:
    Anthropic Claude Code issue
    [#31021](https://github.com/anthropics/claude-code/issues/31021) documents the
    endpoint, required OAuth beta header, expected five-hour/seven-day fields,
    and observed 429 behavior. Issue
    [#31319](https://github.com/anthropics/claude-code/issues/31319) reports
    concurrent-session exhaustion of that endpoint. These are public field
    reports, not an official API contract.
[^anthropic-headers]:
    Anthropic Claude Code issue
    [#29300](https://github.com/anthropics/claude-code/issues/29300) lists the
    unified headers and states that Claude Code parses them. This is a public
    report in the provider's repository, not published API documentation.
[^codex-encrypted]:
    OpenAI Codex issue
    [#13724](https://github.com/openai/codex/issues/13724) reports
    `invalid_encrypted_content` with an `organization_id` mismatch; issue
    [#17541](https://github.com/openai/codex/issues/17541) reports decryption
    failure after a model-provider switch. Neither establishes same-provider
    cross-account portability.
[^anthropic-thinking]:
    Anthropic docs,
    [Extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking),
    require complete, unmodified thinking blocks when passed back. They do not
    state that signatures are portable across accounts.

[codex-rate-limit-parser]:
    https://github.com/openai/codex/blob/430a16799d90e296fb27b1e94f206023902b7e56/codex-rs/codex-api/src/rate_limits.rs#L15-L87
[codex-usage-client]:
    https://github.com/openai/codex/blob/430a16799d90e296fb27b1e94f206023902b7e56/codex-rs/backend-client/src/client/rate_limit_resets.rs#L15-L70
[codex-usage-url]:
    https://github.com/openai/codex/blob/430a16799d90e296fb27b1e94f206023902b7e56/codex-rs/backend-client/src/client/rate_limit_resets.rs#L100-L113
[codex-usage-mapping]:
    https://github.com/openai/codex/blob/430a16799d90e296fb27b1e94f206023902b7e56/codex-rs/backend-client/src/client.rs#L615-L750
[codex-status-stale]:
    https://github.com/openai/codex/blob/430a16799d90e296fb27b1e94f206023902b7e56/codex-rs/tui/src/status/rate_limits.rs#L1-L60
