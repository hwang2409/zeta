"""Structured ownership for mutable spans in one runtime prompt."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..protocol.types import Message, TextContent


@dataclass(frozen=True, slots=True)
class OwnedSpan:
    """The location and digest of one owned prompt slice."""

    offset: int
    length: int
    digest: str


@dataclass(frozen=True, slots=True)
class PromptFragment:
    """Replacement text and spans owned inside that text."""

    text: str
    spans: Mapping[str, OwnedSpan]


@dataclass(frozen=True, slots=True)
class RuntimePrompt:
    """One live prompt and the metadata that locates its owned spans."""

    system_prompt: str | Message
    prompt_components: dict[str, dict[str, int | str]]


def owned_span(text: str, offset: int = 0, length: int | None = None) -> OwnedSpan:
    """Describe one exact slice of text."""

    span_length = len(text) - offset if length is None else length
    value = text[offset : offset + span_length]
    return OwnedSpan(offset, span_length, hashlib.sha256(value.encode()).hexdigest())


def _span(value: Mapping[str, int | str]) -> OwnedSpan:
    return OwnedSpan(
        offset=int(value["offset"]),
        length=int(value["length"]),
        digest=str(value["digest"]),
    )


def _serialized(span: OwnedSpan) -> dict[str, int | str]:
    return {"offset": span.offset, "length": span.length, "digest": span.digest}


def _validate_span(text: str, key: str, span: OwnedSpan) -> None:
    if span.offset < 0 or span.length < 0 or span.offset + span.length > len(text):
        raise ValueError(f"owned prompt span {key!r} is out of bounds")
    actual = owned_span(text, span.offset, span.length).digest
    if actual != span.digest:
        raise ValueError(f"owned prompt span {key!r} digest does not match")


def _strictly_contains(outer: OwnedSpan, inner: OwnedSpan) -> bool:
    return (
        outer.offset <= inner.offset
        and inner.offset + inner.length <= outer.offset + outer.length
        and (outer.offset, outer.length) != (inner.offset, inner.length)
    )


def _validate_relationships(spans: Mapping[str, OwnedSpan]) -> None:
    items = list(spans.items())
    for index, (left_key, left) in enumerate(items):
        for right_key, right in items[index + 1 :]:
            if _strictly_contains(left, right) or _strictly_contains(right, left):
                continue
            left_end = left.offset + left.length
            right_end = right.offset + right.length
            if left_end <= right.offset or right_end <= left.offset:
                continue
            raise ValueError(
                f"owned prompt spans {left_key!r} and {right_key!r} partially overlap"
            )


def _prompt_text(prompt: str | Message) -> str:
    if isinstance(prompt, str):
        return prompt
    if any(not isinstance(block, TextContent) for block in prompt.content):
        raise ValueError("owned runtime prompts must contain only text")
    return "".join(block.text for block in prompt.content)


def _with_prompt_text(prompt: str | Message, text: str) -> str | Message:
    if isinstance(prompt, str):
        return text
    return replace(prompt, content=[TextContent(text)])


def replace_runtime_prompt(
    base: RuntimePrompt,
    text: str,
    components: Mapping[str, Mapping[str, int | str]],
) -> RuntimePrompt:
    """Prepare a whole prompt replacement while preserving its wire shape."""

    return RuntimePrompt(
        _with_prompt_text(base.system_prompt, text),
        {key: dict(component) for key, component in components.items()},
    )


def replace_owned_fragment(
    prompt: RuntimePrompt, key: str, fragment: PromptFragment
) -> RuntimePrompt:
    """Replace one validated owned span and preserve all nested ownership."""

    text = _prompt_text(prompt.system_prompt)
    spans = {name: _span(value) for name, value in prompt.prompt_components.items()}
    for name, span in spans.items():
        _validate_span(text, name, span)
    _validate_relationships(spans)
    if key not in spans:
        raise KeyError(key)

    fragment_spans = dict(fragment.spans)
    if key in fragment_spans:
        raise ValueError(f"replacement fragment cannot redefine target {key!r}")
    for name, span in fragment_spans.items():
        _validate_span(fragment.text, name, span)
    _validate_relationships(fragment_spans)

    target = spans[key]
    target_end = target.offset + target.length
    replacement = fragment.text
    updated_text = text[: target.offset] + replacement + text[target_end:]
    delta = len(replacement) - target.length
    descendants = {
        name for name, span in spans.items() if _strictly_contains(target, span)
    }
    updated: dict[str, OwnedSpan] = {}
    for name, span in spans.items():
        if name == key or name in descendants:
            continue
        if _strictly_contains(span, target):
            updated[name] = owned_span(updated_text, span.offset, span.length + delta)
        elif span.offset >= target_end:
            updated[name] = OwnedSpan(span.offset + delta, span.length, span.digest)
        else:
            updated[name] = span

    updated[key] = owned_span(updated_text, target.offset, len(replacement))
    for name, span in fragment_spans.items():
        updated[name] = OwnedSpan(target.offset + span.offset, span.length, span.digest)
    _validate_relationships(updated)
    return RuntimePrompt(
        _with_prompt_text(prompt.system_prompt, updated_text),
        {name: _serialized(span) for name, span in updated.items()},
    )


__all__ = [
    "OwnedSpan",
    "PromptFragment",
    "RuntimePrompt",
    "owned_span",
    "replace_owned_fragment",
]
