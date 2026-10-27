from __future__ import annotations


def notification_payloads() -> dict[str, dict[str, object]]:
    return {
        "agent_completion": {
            "origin": "notification",
            "kind": "agent_completion",
            "child_instance_id": "root:1",
            "child_session_path": "/sessions/root/agents/1",
            "description": "worker",
            "status": "completed",
            "text": "done",
        },
        "child_idle": {
            "origin": "notification",
            "kind": "child_idle",
            "child_instance_id": "root:1",
            "child_session_path": "/sessions/root/agents/1",
            "description": "worker",
            "status": "child_idle",
            "text": "waiting",
        },
        "task_exited": {
            "origin": "notification",
            "kind": "task_exited",
            "task_id": "task-1",
            "headline": "command finished",
            "exit_code": 0,
        },
        "project_inbox": {
            "origin": "notification",
            "kind": "project_inbox",
            "status": "new",
            "project_id": "project-1",
            "message_ids": ["message-1"],
            "text": "new project message",
        },
        "child_question": {
            "origin": "notification",
            "kind": "child_question",
            "child_instance_id": "root:1",
            "question_id": "question-1",
            "description": "worker",
            "status": "pending",
            "question": "continue?",
            "text": "worker asks a question",
        },
        "child_question_withdrawn": {
            "origin": "notification",
            "kind": "child_question_withdrawn",
            "child_instance_id": "root:1",
            "question_id": "question-1",
            "description": "worker",
            "status": "withdrawn",
            "text": "question withdrawn",
        },
    }
