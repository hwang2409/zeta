"""Provider-neutral types for the zeta agent loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator, Mapping, Protocol, Sequence


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL = "approval"
    COMPACTION = "compaction"


class ContentType(StrEnum):
    TEXT = "text"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"
    TOOL_USE = "tool_use"


@dataclass(frozen=True, slots=True)
class TextContent:
    text: str

    @property
    def type(self) -> ContentType:
        return ContentType.TEXT

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "text": self.text}


@dataclass(frozen=True, slots=True)
class ThinkingContent:
    text: str
    signature: str | None = None

    @property
    def type(self) -> ContentType:
        return ContentType.THINKING

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type.value, "text": self.text}
        if self.signature is not None:
            result["signature"] = self.signature
        return result


@dataclass(frozen=True, slots=True)
class RedactedThinkingContent:
    data: str

    @property
    def type(self) -> ContentType:
        return ContentType.REDACTED_THINKING

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "data": self.data}


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolCall:
        call_id = value.get("id")
        name = value.get("name")
        arguments = value.get("arguments")
        if type(call_id) is not str or not call_id:
            raise ValueError("tool call id must be a nonempty string")
        if type(name) is not str or not name:
            raise ValueError("tool call name must be a nonempty string")
        if type(arguments) is not dict:
            raise ValueError("tool call arguments must be an object")
        return cls(
            id=call_id,
            name=name,
            arguments=arguments,
        )


@dataclass(frozen=True, slots=True)
class ToolUseContent:
    tool_call: ToolCall

    @property
    def type(self) -> ContentType:
        return ContentType.TOOL_USE

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "tool_call": self.tool_call.to_dict()}


ContentBlock = TextContent | ThinkingContent | RedactedThinkingContent | ToolUseContent
TextBlock = TextContent
ThinkingBlock = ThinkingContent
RedactedThinkingBlock = RedactedThinkingContent
ToolUseBlock = ToolUseContent


def content_from_dict(value: Mapping[str, Any]) -> ContentBlock:
    content_type_value = value.get("type")
    if type(content_type_value) is not str:
        raise ValueError("content type must be a string")
    content_type = ContentType(content_type_value)
    if content_type is ContentType.TEXT:
        text = value.get("text")
        if type(text) is not str:
            raise ValueError("text content text must be a string")
        return TextContent(text)
    if content_type is ContentType.THINKING:
        text = value.get("text")
        if type(text) is not str:
            raise ValueError("thinking content text must be a string")
        signature = value.get("signature")
        if signature is not None and type(signature) is not str:
            raise ValueError("thinking content signature must be a string")
        return ThinkingContent(text, signature)
    if content_type is ContentType.REDACTED_THINKING:
        data = value.get("data")
        if type(data) is not str or not data:
            raise ValueError("redacted thinking data must be a nonempty string")
        return RedactedThinkingContent(data)
    tool_call = value.get("tool_call")
    if type(tool_call) is not dict:
        raise ValueError("tool-use content tool_call must be an object")
    return ToolUseContent(ToolCall.from_dict(tool_call))


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolResult:
        tool_call_id = value.get("tool_call_id")
        content = value.get("content")
        is_error = value.get("is_error")
        if type(tool_call_id) is not str or not tool_call_id:
            raise ValueError("tool result call id must be a nonempty string")
        if type(content) is not str:
            raise ValueError("tool result content must be a string")
        if type(is_error) is not bool:
            raise ValueError("tool result is_error must be a boolean")
        return cls(
            tool_call_id=tool_call_id,
            content=content,
            is_error=is_error,
        )


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: list[ContentBlock] = field(default_factory=list)
    tool_result: ToolResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": self.role.value,
            "content": [block.to_dict() for block in self.content],
        }
        if self.tool_result is not None:
            result["tool_result"] = self.tool_result.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Message:
        role_value = value.get("role")
        content_value = value.get("content")
        if type(role_value) is not str:
            raise ValueError("message role must be a string")
        if type(content_value) is not list:
            raise ValueError("message content must be an array")
        if any(type(block) is not dict for block in content_value):
            raise ValueError("message content blocks must be objects")
        tool_result_value = value.get("tool_result")
        if "tool_result" in value and type(tool_result_value) is not dict:
            raise ValueError("message tool_result must be an object")
        metadata_value = value.get("metadata", {})
        if type(metadata_value) is not dict:
            raise ValueError("message metadata must be an object")
        return cls(
            role=MessageRole(role_value),
            content=[content_from_dict(block) for block in content_value],
            tool_result=(
                ToolResult.from_dict(tool_result_value)
                if tool_result_value is not None
                else None
            ),
            metadata=dict(metadata_value),
        )


class StreamEventType(StrEnum):
    AGENT_START = "agent_start"
    TURN_START = "turn_start"
    MESSAGE_START = "message_start"
    MESSAGE_UPDATE = "message_update"
    MESSAGE_END = "message_end"
    TOOL_EXECUTION_START = "tool_execution_start"
    TOOL_EXECUTION_UPDATE = "tool_execution_update"
    TOOL_EXECUTION_END = "tool_execution_end"
    TURN_END = "turn_end"
    AGENT_END = "agent_end"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ErrorInfo:
        return cls(code=str(value["code"]), message=str(value["message"]))


@dataclass(frozen=True, slots=True)
class StreamEvent:
    type: StreamEventType
    message: Message | None = None
    content: ContentBlock | None = None
    delta: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    error: ErrorInfo | None = None
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> StreamEventType:
        return self.type

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type.value}
        if self.message is not None:
            result["message"] = self.message.to_dict()
        if self.content is not None:
            result["content"] = self.content.to_dict()
        if self.delta is not None:
            result["delta"] = self.delta
        if self.tool_call is not None:
            result["tool_call"] = self.tool_call.to_dict()
        if self.tool_result is not None:
            result["tool_result"] = self.tool_result.to_dict()
        if self.error is not None:
            result["error"] = self.error.to_dict()
        if self.data:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StreamEvent:
        content_value = value.get("content")
        error_value = value.get("error")
        return cls(
            type=StreamEventType(value["type"]),
            message=(
                Message.from_dict(value["message"])
                if isinstance(value.get("message"), Mapping)
                else None
            ),
            content=(
                content_from_dict(content_value)
                if isinstance(content_value, Mapping)
                else None
            ),
            delta=(str(value["delta"]) if "delta" in value else None),
            tool_call=(
                ToolCall.from_dict(value["tool_call"])
                if isinstance(value.get("tool_call"), Mapping)
                else None
            ),
            tool_result=(
                ToolResult.from_dict(value["tool_result"])
                if isinstance(value.get("tool_result"), Mapping)
                else None
            ),
            error=(
                ErrorInfo.from_dict(error_value)
                if isinstance(error_value, Mapping)
                else None
            ),
            data=dict(value.get("data", {})),
        )


ToolSchema = Mapping[str, Any]


class CompletionBackend(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        """Return events for one completion. The loop owns turn control."""
