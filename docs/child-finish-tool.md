# explicit completion for background subagents

status: proposed

date: 2026-10-20

## summary

background `worker` and `reviewer` children should remain live until they call a
new `finish` tool. a turn that ends without `finish` is not completion. zeta
nudges the child once, then records it as idle and waits without using model
tokens.

this behavior starts behind a global setting and profile opt-in. phoebe will use
it before it becomes the default. foreground children and other profiles keep
the current one-shot behavior during the rollout.

## motivation

current completion is inferred from provider turn shape, not agent intent. this
has caused real failures:

- an eval runner replied `WAIT` while waiting for its parent's approval. zeta
  treated the no-tool turn as completion, closed the child, and killed its
  background run after 39 of 96 calls.
- a reviewer finished seconds after it reported. the parent's follow-up then
  failed with `no live run/child ... canceled, finished, or never started`. the
  parent had to start a new reviewer and lost the original context.
- a provider `server_error` currently exits outside the normal finish path and
  kills child-owned tasks. pr #521 adds retry for that error class. this design
  separately defines the terminal result after retries are exhausted.

an agent should say when its contract is complete. silence at a turn boundary
should mean “available for more work,” not “destroy this run.”

## current behavior

- `consume_child` saves an assistant message as final when `TURN_END` reports
  zero tool calls (`src/zeta/agent/runner.py:160-165`). it then returns a
  completed result from that text (`src/zeta/agent/runner.py:235-251`).
- `consume_run` delivers queued `agent_send` prompts and notification turns,
  then finishes when neither exists (`src/zeta/agent/runner.py:254-273`,
  `src/zeta/agent/runner.py:344-384`).
- a follow-up loop exists only when a profile accepts follow-ups and the child
  is background (`src/zeta/agent/conversation_channel.py:24-27`). packaged
  `general` and `run` accept them; `explore` and `plan` do not
  (`src/zeta/agent/presets.py:43-100`). the local profiles also opt in
  (`~/.zeta/agents/worker.md:6`, `~/.zeta/agents/reviewer.md:6`).
- eligible children receive non-blocking `ask_parent`; it durably publishes a
  question and says that the answer will arrive as a follow-up
  (`src/zeta/tools/ask_parent/__init__.py:22-100`). `agent_send` appends the
  prompt durably and wakes the child (`src/zeta/agent/conversation_channel.py:82-141`,
  `src/zeta/tools/agent_send/__init__.py:15-97`).
- completion is persisted as a notification, the live child marker is removed,
  and the parent is woken (`src/zeta/agent/conversation_channel.py:143-185`).
  terminal receipts are bounded to 10,000 bytes
  (`src/zeta/agent/receipt.py:21-27`).
- if an inferred final turn still owns children or tasks, the finish gate asks
  the model to wait, cancel, or call `agent_handoff`
  (`src/zeta/agent/background.py:33-72`, `src/zeta/agent/finish_gate.py:195-229`).
  its defaults are 8 model turns, 600 seconds for the model bound, and 3,600
  seconds for a running task (`src/zeta/agent/background.py:91-122`). pr #512
  lets a running task outlive the short bound but retains the one-hour safety
  bound.
- child closure aborts the registry and waits for it to kill background tasks;
  the resulting killed task ids are included in the completion path
  (`src/zeta/agent/background.py:880-931`).
- live markers and lifecycle files drive `agent_status` and tui run controls
  (`src/zeta/tools/agent/__init__.py:740-772`,
  `src/zeta/tui/cards/agent.py:726-843`). restart recovery cancels unfinished
  descendants instead of recreating their loops
  (`src/zeta/agent/background.py:479-580`).
- a live child holds an asyncio run and its backend, store, registry, and tool
  state. it holds at least one session-directory fd in `ConversationStore` and
  one cwd fd in its cloned `ToolRegistry` (`src/zeta/core/store/_store.py:120-156`,
  `src/zeta/tools/registry.py:187-264`, `src/zeta/tools/registry.py:658-672`).
  pr #498 fixed those registry fds leaking after children ended; an idle child
  still holds them intentionally until it ends.

## scope

this design distinguishes a completed contract from a quiet turn, preserves
context for follow-ups, waits without model calls, keeps terminal data bounded,
and replaces the opted-in finish gate. it does not change foreground children,
resume processes after zeta exits, add a workflow framework, or change provider
retry policy; pr #521 owns retries.

## proposed design

### eligibility and setting

add this global-only setting:

```toml
[agents]
explicit_finish = false
```

add profile frontmatter `explicit_finish: true`. the tool is available only
when all of these are true:

1. `[agents].explicit_finish` is true in global settings;
2. the selected profile has `explicit_finish: true`;
3. the child is background and `accepts_follow_ups: true`.

project settings cannot enable it during the experiment. foreground children
never receive the tool. the first opted-in profiles are the user's `worker` and
`reviewer`; packaged presets remain unchanged until the phoebe trial succeeds.
the effective value is captured in the agent-catalog and child-marker snapshots
so resume and inspection do not reinterpret an old run under new settings.

### the `finish` tool

register one session tool named `finish` for eligible children:

```json
{
  "type": "object",
  "properties": {
    "status": {"enum": ["done", "blocked", "failed"]},
    "summary": {"type": "string", "minLength": 1, "maxLength": 8000},
    "head_sha": {"type": "string", "pattern": "^[0-9a-f]{7,64}$"},
    "verdict": {"enum": ["pass", "fail", "inconclusive"]}
  },
  "required": ["status", "summary"],
  "additionalProperties": false
}
```

`head_sha` is for implementation work. `verdict` is for reviews. both are
optional because neither applies to every contract. the tool requires no
approval and is not parallel-safe.

`finish` attempts to record terminal intent through the serialized terminal
transition described below. its successful result tells the model that no more
work or prose is needed. zeta durably records the accepted arguments and tool
result, enters `finishing`, and lets the current provider turn close so the
provider message remains valid. later assistant prose is ignored as the report.
a second `finish`, and every other tool call after a successful `finish` in the
same turn, is rejected as `child is finishing`.

report status and runtime state are separate:

| `finish.status` | lifecycle state | receipt error | meaning |
| --- | --- | --- | --- |
| `done` | `completed` | false | contract completed |
| `blocked` | `completed` | false | contract cannot proceed; parent action is needed |
| `failed` | `failed` | true | contract ended unsuccessfully |

add `report_status`, `summary`, and supplied optional fields to structured
completion content. notification text starts with `done:`, `blocked:`, or
`failed:` and then the summary. `MAX_AGENT_NOTIFICATION_TEXT` is a 10,000
**character** validation check (`src/zeta/core/store/_validation.py:11,32-49`);
the terminal receipt limit is separately byte-based
(`src/zeta/agent/receipt.py:21-27`).

before building text or structured content, compute one canonical summary by
trimming a utf-8-safe prefix until the complete persisted receipt envelope fits
the byte limit with that same summary in both representations. never trim
`report_status`, `head_sha`, or `verdict`. extend `_compact_structured_content`
(`src/zeta/agent/receipt.py:273-344`) to preserve those three bounded scalars
before optional metadata and preserve the already-bounded summary when it fits.
do not persist free-form final prose too. exact-limit tests must use multibyte
text and measure the complete serialized envelope, not python character count.

### no-tool turn and idle state

on the first normal turn that ends without a successful `finish`, send one
harness nudge in a new model turn:

> you have not called `finish`. if the contract is complete, call it now. if
> you need a parent answer, call `ask_parent`. otherwise end this turn and zeta
> will keep you idle for follow-ups without using tokens.

this nudge occurs at most once for the child's lifetime. if the nudge turn also
ends without `finish`, or any later woken turn ends without it, set lifecycle
state to `idle`, set `current_step` to `idle, waiting for follow-up`, and wait.
`idle` is non-terminal: keep the parent marker, pending-prompt queue, loop,
registry, and store open.

on each transition into `idle`, publish exactly one bounded, nonterminal
`child_idle` notification to the direct parent and wake it without polling. the
notification includes the child id, a bounded excerpt of the last assistant
text, and the id of any open `ask_parent` question. repeated status reads do not
republish it. this makes even the original bare `WAIT` response visible to a
parent that is waiting for terminal completion; the parent can send approval or
other direction with `agent_send`.

these events wake an idle child and produce one ordinary model turn:

- a durable `agent_send` follow-up, including an answer to `ask_parent`;
- exit of one of the child's own background tasks;
- completion, failure, or cancellation of one of its direct child agents.

coalesce simultaneous events into that turn. task and child events use the
existing notification input, so the model sees what changed before it acts.

these events do not wake it:

- `agent_status`, `agent_output`, tui viewing, or other reads;
- unrelated parent or sibling notifications;
- progress output from a still-running task;
- elapsed time before a terminal timeout;
- repeated status polling.

### idle timeout and capacity

for the phoebe trial, use a 60-minute idle timeout **pending user
confirmation**. every listed wake event resets the clock. time while the child
owns a running process task or any live descendant agent does not count. after
the final owned descendant becomes terminal, deliver its notification; if the
following turn returns to idle, start a new 60-minute clock. auto-finish never
cancels or adopts owned work.

the synthesized completion has lifecycle state `completed`, report status
`blocked`, reason `idle_timeout`, and summary:

> auto-finished after 60 minutes idle without an explicit finish report.

append a bounded excerpt of the child's last assistant text when one exists.
include normal turns, elapsed time, and tool-call statistics. do not claim that
the contract succeeded.

for the phoebe trial, cap idle explicit-finish children at 16 per root agent
tree **pending user confirmation**. active model turns and children that own a
running task or live descendant agent do not count. reject admission of the
newest excess child to idle by synthesizing `blocked` with reason
`idle_capacity`; do not evict an older child, cancel work, or hand ownership to
its parent. apply capacity only after owned descendants clear and their
notifications are read. the parent can cancel any live child with
`agent_cancel`.

there is no separate “finish now” parent tool in the first version. the parent
can send “finish now” with `agent_send`, which gives the child a chance to write
an accurate report, or cancel it when no report is needed.

### running work and replacement of the finish gate

`finish` refuses while the child owns a running process task or live child
agent. the error lists bounded ids and tells the child to:

- wait for task or child notifications;
- cancel work with `task_kill` or `agent_cancel`; or
- hand off child agents with `agent_handoff`.

register the existing `agent_handoff` beside `finish` for eligible children.
it transfers live descendant agents to the parent. process tasks remain tied to
the child's registry and cannot be handed off; they must exit or be killed.
only after ownership is clear can `finish` succeed.

for opted-in children, delete the 8-turn and 600-second finish-gate model loop.
remove its inferred-final-message entry condition. retain one safety rule from
pr #512: a process task that is still running 3,600 seconds after the child
first becomes idle is killed, and its exit wakes the child to report. this is a
resource safety deadline, not a finish gate and not a model-turn deadline. if
the child does not call `finish` after that wake, the normal idle clock applies.

this leaves one normal path: explicit `finish`; and three harness-owned terminal
paths: idle timeout/capacity, cancellation, and unrecoverable error. non-opted-in
profiles retain the old finish gate unchanged.

### serialized terminal-transition fence

use one per-child serialized fence for accepted `finish`, idle timeout, capacity
auto-finish, cancellation, and restart recovery. this extends today's prompt
and notification draining in `src/zeta/agent/runner.py:344-384`; do not build a
separate check for each terminal path.

under the fence, inspect prompt delivery claims, relevant task/direct-child
notification claims, owned work, and the durable pending queues. close the
pending-prompt queue only with the existing atomic “close if empty” operation.
a prompt committed before closure wins and must be read. a sender racing after
closure gets a delivery error and cannot believe an unread follow-up was
accepted. a task or direct-child terminal notification that is pending or
claimed also wins and must be read.

explicit `finish` returns a tool error telling the child to read the pending
input when any winning prompt or relevant notification exists. timeout and
capacity defer, wake the child with that input, and retry only after it returns
to idle. cancellation and restart recovery still terminate by policy, but use
the same fence. they first give already-committed prompts an explicit canceled
delivery outcome, then use “close if empty” and record cancellation atomically;
they never force-close a nonempty queue or masquerade as explicit or automatic
completion. the fence accepts `finish` only when queues are empty, no relevant
delivery is claimed, and no owned work is live. it then durably records the
arguments and successful tool result before entering `finishing`.

### state transitions

| state | event under the fence | next state and action |
| --- | --- | --- |
| `running` | healthy turn without accepted `finish` | nudge once, then `idle` and publish one `child_idle` |
| `running` or `idle` | prompt or relevant notification wins | `running`; deliver it before any terminal transition |
| `running` or `idle` | `finish`, but input is pending/claimed or work is owned | unchanged; reject with a tool error that identifies what to read or clear |
| `running` or `idle` | valid `finish` wins | `finishing`; close admission and durably store report plus successful result |
| `finishing` | second `finish` or any other tool call | `finishing`; reject as `child is finishing` |
| `finishing` | turn end or later stream/provider failure | terminal from the explicit report; publish it once |
| `idle` | timeout/capacity, but input or owned descendants exist | `running` for input, or remain `idle` with clock paused; do not auto-finish |
| `idle` | timeout/capacity wins with no input or descendants | `completed` with synthesized `blocked` receipt |
| any live state | cancellation or restart recovery wins | `canceled`; atomically close admission and clean up owned work |
| `running` or `idle` | provider failure before accepted `finish` | `failed` after retries; clean up owned work |
| any terminal state | any event | unchanged; no second terminal receipt |

### errors, cancellation, exit, and restart

- **provider error:** apply configured retries, including pr #521 when merged.
  before the fence has durably accepted `finish` and its successful tool result,
  an exhausted provider error wins: emit a failed receipt containing the
  sanitized provider error and killed-task metadata, close the child, and clean
  up owned work. after that durable point, the explicit report wins; a later
  stream failure cannot replace it and zeta publishes that report once.
- **cancellation:** `agent_cancel` remains immediate after it acquires the shared
  fence. mark `canceled`, close pending questions, cancel descendants, kill
  process tasks, and publish the current bounded canceled receipt. a `finish`
  that was not durably accepted before cancellation acquired the fence does not
  outrank cancellation.
- **parent exit:** graceful parent shutdown cancels live children as it does
  now. a crash is reconciled as canceled on recovery. child conversations stay
  available for inspection, but `agent_send` cannot revive them.
- **restart:** do not rehydrate idle loops in the first version. durable idle
  state makes the reason visible during recovery, but recovery converts it to a
  canceled terminal result. this avoids pretending process tasks, in-memory
  backends, approvals, and tool leases survived.
- **foreground children:** unchanged. they still complete from their final
  no-tool assistant turn and do not receive `finish`.

### parent visibility

`agent_status` must accept and return lifecycle state `idle`, plus
`report_status` and `idle_since` when present. an idle child remains in live
results and still accepts `agent_send`. its tui card shows `idle · waiting for
follow-up` and idle duration. it is not styled as completed.

completion uses the existing notification/wake path. the parent sees the
bounded canonical report and structured status exactly once. the original
background agent tool result remains `running`; the later notification remains
the terminal receipt, as today.

extend agent notification validation with bounded `report_status`, `head_sha`,
and `verdict`, preserving their enums and sha pattern. keep child-generated text
json-serialized; do not interpolate it into unquoted prompt text. extend
`_notification_receipt` (`src/zeta/context_eviction/__init__.py:1050-1080`) to
retain those exact scalars after context eviction. retain summary text only when
it fits the existing eviction text budget; otherwise store its digest and rely
on the existing `recall_history` source link for exact recovery.

### resource and token cost

an idle child makes no provider request, so it uses no model tokens. it retains:

- one asyncio waiter and child/watcher state;
- its backend and conversation objects in memory;
- at least two open directory fds: the child session store and registry cwd;
- tool registry state and any approval/store leases.

exact memory depends on provider and transcript size, so the design does not
promise a byte figure. a confirmed 16-child idle cap would bound the known
baseline to at least 32 retained directory fds per root tree, plus provider and
tool-specific resources; excluded children with owned descendants remain
additional live resources and must be counted separately. metrics must record
the observed fd count and rss delta at idle entry and exit during the phoebe
trial. pr #498's close path remains mandatory on all terminal paths.

## alternatives considered

### infer completion from final prose

rejected. `WAIT`, “done,” or “ready” are ambiguous and caused the reported
failures.

### keep the current finish gate and add `finish`

rejected. two completion authorities can disagree and preserve the old
8-turn, 600-second machinery.

### finish immediately on any `finish` call, even with running work

rejected. closing the registry would kill work that the report implies is safe.
refusal makes ownership explicit and testable.

### make `summary` the only argument

rejected. parents need to distinguish success, a blocker, and a failed contract
without parsing prose. `head_sha` and `verdict` cover the two first profiles
without creating arbitrary metadata.

### block inside `ask_parent`

rejected. the durable non-blocking question path works; idle adds the missing
wait state.

### resume idle children after restart

rejected for the first release. rebuilding a valid model loop, tool registry,
approval state, and task ownership is a larger durability protocol. persisted
conversation remains available even when the live lease ends.

### no idle timeout or cap

rejected. idle children use no tokens but retain fds, memory, leases, and a live
address for follow-ups. pr #498 showed that descriptor lifetime is operationally
important.

## rollout and measurement

1. add tests first; confirm each semantic test fails on the pre-implementation
   revision for the expected reason.
2. implement `[agents].explicit_finish = false`, profile parsing/snapshotting,
   and the tool without changing any default behavior.
3. add `explicit_finish: true` only to the user's `worker` and `reviewer`
   profiles. enable the global setting only in phoebe's environment.
4. delete the current finish gate only from the opted-in path. keep it unchanged
   for legacy profiles during the experiment.
5. run phoebe with the setting for at least 100 eligible children and seven
   days, including implementation workers and reviewers.
6. compare with the preceding 100 eligible children. record:
   - `agent_send` failures because the target already finished;
   - no-tool turns followed by later parent messages;
   - auto-finishes by `idle_timeout` and `idle_capacity`;
   - tasks killed on normal completion, safety deadline, cancellation, and
     provider error;
   - finish refusals due to owned work;
   - idle duration, concurrent idle peak, fd count, and rss delta;
   - extra nudge turns and tokens.
7. success requires zero known premature finishes, zero running tasks killed by
   successful `finish`, and no increase in unreconciled child markers. review
   timeout/cap and token data before changing defaults.
8. after review, opt packaged background `run` into the protocol. remove the
   setting and legacy finish gate only in a later migration. do not silently
   change foreground `general`, `explore`, or `plan` behavior.

## test plan

all node ids are proposed and must be created test-first. each semantic test
must fail on the pre-implementation revision for the expected reason.

- `tests/test_agent.py::test_explicit_finish_done_publishes_canonical_report`
  covers schema, lifecycle mapping, structured fields, and one notification.
- `tests/test_agent.py::test_explicit_finish_blocked_and_failed_map_to_receipts`
  proves `blocked` has `isError: false` while `failed` has `isError: true`.
- `tests/test_agent.py::test_explicit_finish_nudges_once_then_idles` covers one
  nudge, durable idle state, and no provider call while waiting.
- `tests/test_agent.py::test_wait_with_running_task_wakes_parent_then_accepts_approval`
  reproduces the original bare `WAIT`, keeps the task alive, wakes the parent,
  delivers later approval, and lets the child report.
- `tests/test_agent.py::test_idle_notification_wakes_waiting_parent_without_polling`
  checks one bounded `child_idle` notice with last text and open question id.
- `tests/test_agent.py::test_idle_child_wakes_for_task_and_child_completion_only`
  covers relevant wakes and excludes reads, progress, and unrelated notices.
- `tests/test_agent.py::test_reviewer_follow_up_committed_at_finish_boundary_is_read`
  uses a barrier to prove a committed prompt wins over `finish`; a send after
  closure is rejected.
- `tests/test_agent.py::test_pending_task_or_child_notification_rejects_finish`
  barrier-tests both pending and claimed notification delivery.
- `tests/test_agent.py::test_terminal_fence_follow_up_wins_timeout_capacity_cancel_and_recovery_races`
  barrier-tests queue closure and deterministic delivery outcomes for every
  harness-owned terminal transition.
- `tests/test_agent.py::test_explicit_finish_refuses_owned_work_until_clear`
  covers task/child refusal, cancellation, and descendant handoff.
- `tests/test_agent.py::test_idle_timeout_pauses_for_owned_task_and_child_agent`
  covers both ownership types, no cancellation or implicit handoff, and timeout
  only after ownership clears and its notification is read.
- `tests/test_agent.py::test_idle_capacity_excludes_owned_task_and_child_agent`
  covers both ownership types and newest-child admission only after they clear.
- `tests/test_agent.py::test_explicit_finish_twice_rejects_second_call` and
  `tests/test_agent.py::test_tool_call_after_successful_finish_is_rejected`
  close the same-turn protocol after one accepted report.
- `tests/test_agent.py::test_provider_error_after_successful_finish_keeps_report`
  proves the durable explicit report outranks a later stream failure; the
  existing pre-acceptance provider-error test proves the opposite boundary.
- `tests/test_agent_receipt.py::test_explicit_finish_multibyte_summary_fits_exact_envelope_limit`
  checks utf-8-safe trimming and both summary representations at the persisted
  byte limit.
- `tests/test_context_eviction.py::test_notification_receipt_preserves_finish_head_sha_and_verdict`
  verifies exact scalar recovery after eviction and summary digest/recall when
  text does not fit.
- `tests/test_background.py::test_explicit_finish_task_safety_deadline_kills_and_wakes`
  covers the single 3,600-second task deadline and killed-task metadata.
- `tests/test_agent.py::test_explicit_finish_cancel_wins_before_acceptance` and
  `tests/test_agent.py::test_idle_child_restart_recovers_as_canceled` cover the
  shared fence, cleanup, and v1 no-rehydration rule.
- `tests/test_agents.py::test_explicit_finish_requires_global_and_profile_opt_in`,
  `tests/test_settings.py::test_project_cannot_enable_explicit_finish`, and
  `tests/test_tui.py::test_idle_agent_card_and_status_remain_live` cover rollout
  eligibility, configuration, and inspection.
- existing finish-gate tests remain unchanged for legacy profiles; an opted-in
  regression proves those runs never enter `run_finish_gate`.

## open questions

1. **pending user confirmation:** use a 60-minute idle timeout for the phoebe
   trial to favor follow-up continuity?
2. **pending user confirmation:** cap idle children at 16 per root tree, reject
   the newest excess admission, and exclude children with owned descendants?
