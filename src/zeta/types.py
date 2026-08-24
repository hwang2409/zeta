"""Provider-neutral types for the zeta agent loop."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import (
    Any,
    AsyncIterator,
    Literal,
    Mapping,
    NotRequired,
    Protocol,
    Sequence,
    TypedDict,
)


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


class ToolAnnotations(TypedDict, total=False):
    audience: list[Literal["user", "assistant"]]
    priority: float
    lastModified: str


class ToolTextBlock(TypedDict):
    """MCP text content; metadata is always present, even when not capped."""

    type: Literal["text"]
    text: str
    truncated: bool
    full_size: int
    annotations: NotRequired[ToolAnnotations]


type StructuredContentValue = (
    str
    | int
    | bool
    | None
    | list[StructuredContentValue]
    | dict[str, StructuredContentValue]
)


class ToolImageBlock(TypedDict):
    type: Literal["image"]
    data: str
    mimeType: str
    annotations: NotRequired[ToolAnnotations]


class ToolTextResource(TypedDict):
    uri: str
    mimeType: NotRequired[str]
    text: str


class ToolBlobResource(TypedDict):
    uri: str
    mimeType: NotRequired[str]
    blob: str


class ToolTextResourceBlock(TypedDict):
    type: Literal["resource"]
    resource: ToolTextResource
    annotations: NotRequired[ToolAnnotations]


class ToolBlobResourceBlock(TypedDict):
    type: Literal["resource"]
    resource: ToolBlobResource
    annotations: NotRequired[ToolAnnotations]


ToolResourceBlock = ToolTextResourceBlock | ToolBlobResourceBlock
ToolContentBlock = ToolTextBlock | ToolImageBlock | ToolResourceBlock


def validate_tool_content_block(index: int, block: object) -> ToolContentBlock:
    prefix = f"content[{index}]"
    if type(block) is not dict:
        raise ValueError(f"{prefix} must be an object")
    block_type = block.get("type")
    if block_type == "text":
        required_keys = {"type", "text", "truncated", "full_size"}
        allowed_keys = {*required_keys, "annotations"}
        if not required_keys <= set(block) or not set(block) <= allowed_keys:
            raise ValueError(f"{prefix} has an invalid text shape")
        if type(block["text"]) is not str:
            raise ValueError(f"{prefix}.text must be a string")
        if type(block["truncated"]) is not bool:
            raise ValueError(f"{prefix}.truncated must be a boolean")
        if type(block["full_size"]) is not int or block["full_size"] < 0:
            raise ValueError(f"{prefix}.full_size must be nonnegative")
        normalized: ToolTextBlock = {
            "type": "text",
            "text": block["text"],
            "truncated": block["truncated"],
            "full_size": block["full_size"],
        }
        if "annotations" in block:
            normalized["annotations"] = _validate_annotations(
                prefix, block["annotations"]
            )
        return normalized
    if block_type == "image":
        required_keys = {"type", "data", "mimeType"}
        allowed_keys = {*required_keys, "annotations"}
        if not required_keys <= set(block) or not set(block) <= allowed_keys:
            raise ValueError(f"{prefix} has an invalid image shape")
        if type(block.get("data")) is not str:
            raise ValueError(f"{prefix}.data must be a string")
        if type(block.get("mimeType")) is not str:
            raise ValueError(f"{prefix}.mimeType must be a string")
        normalized_image: ToolImageBlock = {
            "type": "image",
            "data": block["data"],
            "mimeType": block["mimeType"],
        }
        if "annotations" in block:
            normalized_image["annotations"] = _validate_annotations(
                prefix, block["annotations"]
            )
        return normalized_image
    if block_type == "resource":
        required_keys = {"type", "resource"}
        allowed_keys = {*required_keys, "annotations"}
        if not required_keys <= set(block) or not set(block) <= allowed_keys:
            raise ValueError(f"{prefix} has an invalid resource shape")
        resource = block.get("resource")
        if type(resource) is not dict:
            raise ValueError(f"{prefix}.resource must be an object")
        resource_keys = {"uri", "mimeType", "text", "blob"}
        if not set(resource) <= resource_keys:
            raise ValueError(f"{prefix}.resource has unsupported fields")
        if type(resource.get("uri")) is not str:
            raise ValueError(f"{prefix}.resource.uri must be a string")
        has_text = "text" in resource
        has_blob = "blob" in resource
        if has_text == has_blob:
            raise ValueError(f"{prefix}.resource must contain text or blob")
        payload_key = "text" if has_text else "blob"
        if type(resource[payload_key]) is not str:
            raise ValueError(f"{prefix}.resource.{payload_key} must be a string")
        mime_type = resource.get("mimeType")
        if mime_type is not None and type(mime_type) is not str:
            raise ValueError(f"{prefix}.resource.mimeType must be a string")
        normalized_resource: ToolTextResource | ToolBlobResource
        if has_text:
            normalized_resource = {"uri": resource["uri"], "text": resource["text"]}
        else:
            normalized_resource = {"uri": resource["uri"], "blob": resource["blob"]}
        if mime_type is not None:
            normalized_resource["mimeType"] = mime_type
        normalized_block: ToolTextResourceBlock | ToolBlobResourceBlock = {
            "type": "resource",
            "resource": normalized_resource,
        }
        if "annotations" in block:
            normalized_block["annotations"] = _validate_annotations(
                prefix, block["annotations"]
            )
        return normalized_block
    raise ValueError(f"{prefix}.type is unsupported: {block_type}")


def _validate_annotations(prefix: str, value: object) -> ToolAnnotations:
    if type(value) is not dict:
        raise ValueError(f"{prefix}.annotations must be an object")
    allowed_keys = {"audience", "priority", "lastModified"}
    if not set(value) <= allowed_keys:
        raise ValueError(f"{prefix}.annotations has unsupported fields")
    normalized: ToolAnnotations = {}
    if "audience" in value:
        audience = value["audience"]
        if type(audience) is not list or any(
            type(role) is not str or role not in {"user", "assistant"}
            for role in audience
        ):
            raise ValueError(
                f"{prefix}.annotations.audience must contain user or assistant"
            )
        normalized["audience"] = list(audience)
    if "priority" in value:
        priority = value["priority"]
        if (
            type(priority) not in {int, float}
            or not math.isfinite(priority)
            or not 0 <= priority <= 1
        ):
            raise ValueError(f"{prefix}.annotations.priority must be between 0 and 1")
        normalized["priority"] = float(priority)
    if "lastModified" in value:
        last_modified = value["lastModified"]
        if type(last_modified) is not str:
            raise ValueError(f"{prefix}.annotations.lastModified must be a string")
        normalized["lastModified"] = last_modified
    return normalized


def flatten_tool_content(blocks: Sequence[ToolContentBlock]) -> str:
    values: list[str] = []
    for block in blocks:
        if block["type"] == "text":
            text = block["text"]
            if block["truncated"]:
                shown_bytes = len(text.encode("utf-8"))
                text = (
                    f"{text}\n[truncated: {shown_bytes} of "
                    f"{block['full_size']} bytes]"
                )
            values.append(text)
        elif block["type"] == "image":
            values.append("[image block]")
        else:
            values.append(f"[resource: {block['resource']['uri']}]")
    return "\n".join(values)


class StructuredToolResult(TypedDict):
    """MCP-compatible result returned by the tool registry."""

    content: list[ToolContentBlock]
    isError: bool
    structuredContent: dict[str, StructuredContentValue] | None


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
    content_blocks: list[ToolContentBlock] | None = field(default=None, compare=False)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }
        if self.content_blocks is not None:
            result["content_blocks"] = self.content_blocks
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolResult:
        tool_call_id = value.get("tool_call_id")
        content = value.get("content")
        is_error = value.get("is_error")
        content_blocks = value.get("content_blocks")
        if type(tool_call_id) is not str or not tool_call_id:
            raise ValueError("tool result call id must be a nonempty string")
        if type(content) is not str:
            raise ValueError("tool result content must be a string")
        if type(is_error) is not bool:
            raise ValueError("tool result is_error must be a boolean")
        if content_blocks is not None:
            if type(content_blocks) is not list:
                raise ValueError("tool result content_blocks must be an array")
            normalized_blocks: list[ToolContentBlock] = []
            for index, block in enumerate(content_blocks):
                try:
                    normalized_blocks.append(
                        validate_tool_content_block(index, block)
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"tool result content block is invalid: {exc}"
                    ) from exc
            content_blocks = normalized_blocks
        return cls(
            tool_call_id=tool_call_id,
            content=content,
            is_error=is_error,
            content_blocks=content_blocks,
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
    COMPACTION_START = "compaction_start"
    COMPACTION_END = "compaction_end"
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
