# explicit completion for background subagents

status: proposed

date: 2026-10-20

## summary

background `worker` and `reviewer` children should remain live until they call a
new `finish` tool. a turn that ends without `finish` is not completion. zeta
records the child as idle and waits without using model tokens. after 10
eligible idle minutes, it can wake the child with a bounded reminder.

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
  state. it holds at least three fds: the session-directory fd in
  `ConversationStore`, the separate `runtime.lease` file fd held by its
  `SessionRuntimeLease`, and one cwd fd in its cloned `ToolRegistry`
  (`src/zeta/core/store/_store.py:159-178`,
  `src/zeta/core/store/runtime_lease.py:38-51`,
  `src/zeta/tools/registry.py:650-672`). pr #498 fixed those registry fds
  leaking after children ended; an idle child still holds them intentionally
  until it ends.

## scope

this design distinguishes a completed contract from a quiet turn, preserves
context for follow-ups, waits without model calls except for bounded idle
nudges, keeps terminal data bounded, and replaces the opted-in finish gate. it
does not change foreground children, resume processes after zeta exits, add a
workflow framework, or change provider
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

when a normal turn ends without a successful `finish`, set lifecycle state to
`idle`, set `current_step` to `idle, waiting for follow-up`, and wait without an
immediate extra model turn. `idle` is non-terminal: keep the parent marker,
pending-prompt queue, loop, registry, and store open.

on each transition into `idle`, publish exactly one bounded, nonterminal
`child_idle` notification to the direct parent and wake it without polling. the
notification has the dedicated kind `child_idle`, and includes a bounded
`child_instance_id`, `child_session_path`, `text` excerpt of the last assistant
text, and optional `ask_parent_question_id`. its status is `child_idle`; it is a
durable relevant notification, not a terminal completion. give this kind its own
bounded validator and append path: accept `kind = child_idle` and
`status = child_idle` with only those fields and the existing bounded
text/description envelope, while terminal statuses continue to require their
terminal fields (`src/zeta/core/store/_validation.py:32-49` currently accepts
only `completed`, `error`, and `canceled`). keep it in the normal pending-
notification queue, but exclude it from terminal completion lookup, recovery,
and exactly-once terminal deduplication. repeated status reads do not republish
it. this makes even the original bare `WAIT` response visible to a parent that
is waiting for terminal completion; the parent can send approval or other
direction with `agent_send`.

these events wake an idle child and produce one ordinary model turn:

- a durable `agent_send` follow-up, including an answer to `ask_parent`;
- exit of one of the child's own background tasks;
- completion, failure, cancellation, or `child_idle` notification from one of
  its direct child agents. `child_idle` wakes an already-idle explicit-finish
  parent even when that parent owns an idle child or grandchild; the parent
  receives the notification turn before deciding whether to call `finish`.

coalesce simultaneous events into that turn. task and child events use the
existing notification input, so the model sees what changed before it acts.

these events do not wake it:

- `agent_status`, `agent_output`, tui viewing, or other reads;
- unrelated parent or sibling notifications;
- progress output from a still-running task;
- elapsed time before the 10-minute idle-nudge boundary;
- repeated status polling.

### immediate reminder, idle nudge, and resource warning

when an explicit-finish child ends a turn with visible text but without calling
`finish`, zeta immediately gives it one reminder turn:

> you ended your turn without calling `finish`. call `finish` now with your report, or continue the work.

this reminder is delivered only when the child owns no running process task or
live child agent and no follow-up or relevant notification is pending or
claimed. the same serialized fence used by idle nudges decides admission, so
real input wins and zeta never delivers both. one reminder is allowed per idle
period; if the reminder also ends without `finish`, the child becomes idle and
is not reminded again until external input starts a new period. a child with
owned work or a live child waits for that work or its notification instead.
the reminder counts toward the lifetime limit of three idle nudges.

an idle period becomes eligible for one automatic nudge after 10 continuous
minutes only when the child is still idle, no model turn is active, it owns no
running process task or live child agent, and no follow-up or relevant
notification is pending or claimed. recheck all conditions under the serialized
fence before waking it with this ordinary model message:

> you have been idle for 10 minutes. call `finish` if the contract is complete,
> call `ask_parent` if you are waiting on a decision, or continue the work.

the nudge is a wake event, not a terminal path. it resets the idle clock and
produces one ordinary turn. if that turn ends without `finish`, the child enters
a new idle period and can receive another nudge after the same 10-minute rule.
allow at most three automatic idle nudges over the child's lifetime. after the
third, leave it idle until an external wake, cancellation, provider-error exit,
or restart recovery; never spend another model turn merely because time passed.
this absolute bound prevents a child that repeatedly returns no tool calls from
creating a nudge loop.

any wake event resets the 10-minute clock. time while a task or child agent is
live, while a model turn is active, or while input is pending or claimed does
not count. the task/child completion or queued input is delivered first; a new
idle clock starts only if the following turn returns to idle.

there is no hard idle-child cap in v1 **pending user confirmation**. instead,
`agent_status` and the tui report the idle-child count and the known held
idle-child fd floor: at least three per idle child—the session-directory fd,
its `runtime.lease` file fd, and the cloned registry cwd fd. warn above 16 idle
children in one root tree, which means a known floor above 48 held fds, but do
not reject, finish, cancel, or hand off any child.

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
resource safety deadline, not a finish gate, idle-child completion, or a
model-turn deadline. if the child does not call `finish` after that wake, it
returns to idle under the normal nudge rule.

an opted-in idle child never finishes because of elapsed time or tree capacity.
only accepted `finish`, parent/user cancellation, an unrecoverable provider
error before accepted `finish`, or restart recovery ends it. non-opted-in
profiles retain the old finish gate unchanged.

### serialized terminal-transition fence

use one per-child serialized fence for accepted `finish`, the idle-nudge wake,
cancellation, and restart recovery. this extends today's prompt and notification
draining in `src/zeta/agent/runner.py:344-384`; do not build a separate check for
each path.

under the fence, inspect prompt delivery claims, relevant task/direct-child
notification claims, owned work, and the durable pending queues. close the
pending-prompt queue only with the existing atomic “close if empty” operation.
a prompt committed before closure wins and must be read. a sender racing after
closure gets a delivery error and cannot believe an unread follow-up was
accepted. a task or direct-child terminal notification that is pending or
claimed also wins and must be read.

explicit `finish` returns a tool error telling the child to read the pending
input when any winning prompt or relevant notification exists. an idle-nudge
attempt loses to that input, delivers it instead, and resets the clock.
cancellation and restart recovery still terminate by policy, but use the same
fence. they first give already-committed prompts an explicit canceled delivery
outcome, then use “close if empty” and record cancellation atomically; they
never force-close a nonempty queue or masquerade as explicit completion. the
fence accepts `finish` only when queues are empty, no relevant delivery is
claimed, and no owned work is live. it then durably records the
arguments and successful tool result before entering `finishing`.

### state transitions

| state | event under the fence | next state and action |
| --- | --- | --- |
| `running` | explicit-finish turn with text but without accepted `finish` | `running`; if eligible, deliver one immediate reminder, otherwise continue to idle |
| `running` | healthy turn without accepted `finish` after the reminder | `idle`; publish one `child_idle` and start the eligible clock |
| `idle` | external prompt or relevant notification wins | `running`; deliver it and reset the clock (including a durable `child_idle` notification) |
| `idle` | 10 eligible minutes, fewer than three lifetime nudges | `running`; deliver one idle nudge and reset the clock |
| `idle` | nudge boundary while turn/work/input exists | stay live; deliver real input first or keep the clock paused |
| `idle` | 10 eligible minutes after three lifetime nudges | remain `idle`; do not wake or finish |
| `running` or `idle` | `finish`, but input is pending/claimed or work is owned | unchanged; reject with a tool error that identifies what to read or clear |
| `running` or `idle` | valid `finish` wins | `finishing`; close admission and durably store report plus successful result |
| `finishing` | second `finish` or any other tool call | `finishing`; reject as `child is finishing` |
| `finishing` | turn end or later stream/provider failure | terminal from the explicit report; publish it once |
| `finishing` | cancellation, graceful shutdown, or restart recovery | finalize the accepted report exactly once; map its status to the terminal lifecycle state, publish or recover its terminal receipt, close questions and markers, and release resources |
| any live state except `finishing` | cancellation or restart recovery wins | `canceled`; atomically close admission and clean up owned work |
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
  fence. before accepted `finish`, mark `canceled`, close pending questions,
  cancel descendants, kill process tasks, and publish the current bounded
  canceled receipt. after accepted `finish`, cancellation is a no-op: admission
  is already closed and the explicit report remains authoritative; publish no
  second receipt.
- **parent exit:** graceful parent shutdown cancels live children as it does
  now, except an accepted `finish` is finalized exactly once from its stored
  report as described for `finishing`. a crash is reconciled the same way: map
  the stored report to its terminal lifecycle state, publish or recover its
  terminal receipt, close questions and markers, and release resources. child
  conversations stay available for inspection, but `agent_send` cannot revive
  them.
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

while waiting, an idle child makes no provider request and uses no model tokens.
each eligible idle nudge makes one ordinary provider turn, subject to the
three-nudge lifetime bound. the child retains:

- one asyncio waiter and child/watcher state;
- its backend and conversation objects in memory;
- at least three open fds: the child session-directory fd, its regular
  `runtime.lease` file fd, and the registry cwd fd;
- tool registry state and any approval/store leases.

exact memory depends on provider and transcript size, so the design does not
promise a byte figure. each idle child has a known floor of at least three
retained fds; 16 idle children therefore have a floor of at least 48, before
provider and tool-specific resources. `agent_status`, the tui warning, and
trial metrics report both idle-child count and this known fd floor. metrics also record
observed process fd count and rss delta at idle entry and exit. pr #498's close
path remains mandatory on every terminal path.

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

### auto-finish idle children

rejected. elapsed time and tree size do not show that a contract ended. idle
children retain resources, so v1 exposes their count and known fd floor and
warns above 16 instead of inventing a second completion authority.

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
   - idle-nudge count, nudge responses, and nudge tokens;
   - parent/user cancellations of idle children;
   - tasks killed on normal completion, safety deadline, cancellation, and
     provider error;
   - finish refusals due to owned work;
   - idle duration, concurrent idle peak, above-16 warnings, known idle fd
     floor (at least three per idle child), observed process fd count, and rss
     delta.
7. success requires zero known premature finishes, zero idle children finished
   because of time or capacity, zero running tasks killed by successful
   `finish`, and no increase in unreconciled child markers. review nudge, idle
   resource, and token data before changing defaults.
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
- `tests/test_agent.py::test_explicit_finish_no_tool_turn_enters_idle_without_request`
  covers durable idle state and no immediate extra provider call.
- `tests/test_agent.py::test_wait_with_running_task_wakes_parent_then_accepts_approval`
  reproduces the original bare `WAIT`, keeps the task alive, wakes the parent,
  delivers later approval, and lets the child report.
- `tests/test_agent.py::test_idle_notification_wakes_waiting_parent_without_polling`
  checks one bounded `child_idle` notice with last text and open question id.
- `tests/test_agent.py::test_child_idle_then_terminal_receipt_is_appended_and_delivered_once`
  appends an idle notice first, then proves terminal completion is still appended
  and delivered once rather than being suppressed by completion deduplication.
- `tests/test_agent.py::test_idle_child_wakes_for_task_and_child_completion_only`
  covers relevant wakes and excludes reads, progress, and unrelated notices.
- `tests/test_agent.py::test_child_idle_wakes_idle_explicit_finish_parent_tree`
  uses a parent, child, and grandchild that all use `finish`, proving a
  grandchild's durable `child_idle` notification wakes the already-idle parent
  through the child notification turn.
- `tests/test_agent.py::test_reviewer_follow_up_committed_at_finish_boundary_is_read`
  uses a barrier to prove a committed prompt wins over `finish`; a send after
  closure is rejected.
- `tests/test_agent.py::test_pending_task_or_child_notification_rejects_finish`
  barrier-tests both pending and claimed notification delivery.
- `tests/test_agent.py::test_terminal_fence_serializes_finish_idle_nudge_cancel_and_recovery`
  barrier-tests queue closure, nudge eligibility, and deterministic delivery
  outcomes for every fenced path.
- `tests/test_agent.py::test_explicit_finish_refuses_owned_work_until_clear`
  covers task/child refusal, cancellation, and descendant handoff.
- `tests/test_agent.py::test_idle_nudge_after_ten_minutes_without_running_work`
  verifies one ordinary nudge turn after the exact eligible boundary.
- `tests/test_agent.py::test_idle_nudge_waits_for_task_child_active_turn_and_input`
  covers both ownership types, a model turn, and pending or claimed input.
- `tests/test_agent.py::test_idle_wake_resets_nudge_clock` verifies follow-up,
  task, child, and nudge wakes restart a full 10-minute period.
- `tests/test_agent.py::test_idle_nudge_stops_after_three_lifetime_nudges`
  proves repeated no-tool replies cannot create an unbounded nudge loop.
- `tests/test_agent.py::test_idle_child_never_auto_finishes_for_time_or_count`
  advances time and creates more than 16 idle children without terminal output.
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
  `tests/test_agent.py::test_explicit_finish_cancel_after_acceptance_keeps_report`
  cover both sides of the shared fence: cancellation wins before acceptance,
  while cancellation after accepted `finish` is unchanged and emits no second
  receipt.
- `tests/test_agent.py::test_parent_exit_cancels_idle_child_subtree_and_cleans_receipts`
  covers graceful parent exit canceling an idle child and idle grandchild,
  including subtree marker, pending-prompt queue, task, and receipt cleanup.
- `tests/test_agent.py::test_events_after_terminal_state_emit_no_second_receipt`
  proves late prompts, notifications, cancellation, and recovery are ignored
  after terminal publication.
- `tests/test_agent.py::test_idle_child_restart_recovers_as_canceled` covers the
  shared fence cleanup and v1 no-rehydration rule.
- `tests/test_agent.py::test_finish_recovery_after_durable_acceptance_publishes_report_once`
  interrupts after durable acceptance and before normal terminal publication,
  then proves recovery finalizes the stored report and emits one terminal receipt.
- `tests/test_agents.py::test_explicit_finish_requires_global_and_profile_opt_in`,
  `tests/test_settings.py::test_project_cannot_enable_explicit_finish`, and
  `tests/test_tui.py::test_idle_agent_card_reports_count_fd_floor_and_warning`
  cover rollout eligibility, configuration, and resource visibility, including
  the at-least-three-fds-per-idle-child floor and 48-fd warning basis.
- existing finish-gate tests remain unchanged for legacy profiles; an opted-in
  regression proves those runs never enter `run_finish_gate`.

## open questions

1. **decided:** no hard idle-child cap in v1. report the idle count and known
   floor of at least three fds per idle child in `agent_status` and the tui, and
   warn above 16 idle children per root tree (at least 48 fds), without ending
   or rejecting any child.
