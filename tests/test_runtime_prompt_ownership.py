from collections.abc import Callable

import pytest

from zeta.runtime.prompt import (
    PromptFragment,
    RuntimePrompt,
    owned_fragment,
    owned_span,
    replace_owned_fragment,
)

PromptOperation = Callable[[RuntimePrompt], object]


def _serialized(text: str, offset: int, length: int) -> dict[str, int | str]:
    span = owned_span(text, offset, length)
    return {"offset": offset, "length": length, "digest": span.digest}


def _prompt(components: dict[str, object]) -> RuntimePrompt:
    return RuntimePrompt("abcdefghij", components)  # type: ignore[arg-type]


def _operations() -> list[tuple[str, PromptOperation]]:
    return [
        ("extract", lambda prompt: owned_fragment(prompt, "outer")),
        (
            "replace",
            lambda prompt: replace_owned_fragment(
                prompt, "outer", PromptFragment("XYZ", {})
            ),
        ),
    ]


@pytest.mark.parametrize(
    ("malformed", "expected"),
    [
        ({"outer": _serialized("abcdefghij", 8, 5)}, ValueError),
        (
            {"outer": {"offset": 0, "length": 3, "digest": "wrong"}},
            ValueError,
        ),
        (
            {
                "outer": _serialized("abcdefghij", 0, 5),
                "overlap": _serialized("abcdefghij", 4, 3),
            },
            ValueError,
        ),
    ],
)
def test_owned_prompt_operations_reject_the_same_malformed_prompt(
    malformed: dict[str, object], expected: type[Exception]
) -> None:
    errors = []
    for name, operation in _operations():
        with pytest.raises(expected) as raised:
            operation(_prompt(malformed))
        errors.append((name, raised.value.__class__))

    assert errors == [("extract", expected), ("replace", expected)]


def test_owned_prompt_operations_preserve_valid_nested_spans() -> None:
    prompt = _prompt(
        {
            "outer": _serialized("abcdefghij", 1, 7),
            "inner": _serialized("abcdefghij", 3, 2),
        }
    )

    extracted = owned_fragment(prompt, "outer")
    assert extracted.text == "bcdefgh"
    assert extracted.spans["inner"] == owned_span("bcdefgh", 2, 2)

    replaced = replace_owned_fragment(prompt, "outer", extracted)
    assert replaced.system_prompt == "abcdefghij"
    assert replaced.prompt_components["inner"] == {
        "offset": 3,
        "length": 2,
        "digest": _serialized("abcdefghij", 3, 2)["digest"],
    }
