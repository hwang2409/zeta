import json

import pytest

from zeta.auth import error_body_excerpt


@pytest.mark.parametrize(
    ("message", "secrets"),
    [
        (
            "authorization: Bearer bearer-secret",
            ("Bearer", "bearer-secret"),
        ),
        (
            'Authorization =   "bEaReR quoted bearer secret"',
            ("bEaReR", "quoted bearer secret"),
        ),
        (
            'authorization: Bearer "quoted bearer tail secret"',
            ("Bearer", "quoted bearer tail secret"),
        ),
        ('"AUTHORIZATION" : Basic dXNlcjpwYXNz', ("Basic", "dXNlcjpwYXNz")),
        (
            'authorization:\tDIGEST username="digest-user", realm="digest-realm"',
            ("DIGEST", "digest-user", "digest-realm"),
        ),
        ("authorization: Token custom-scheme-marker", ("Token", "custom-scheme-marker")),
        ("authorization: ApiKey apikey-scheme-marker", ("ApiKey", "apikey-scheme-marker")),
        ("authorization: Foo unknown-scheme-marker", ("Foo", "unknown-scheme-marker")),
        (
            "cookie: session=cookie-secret; theme=dark",
            ("cookie-secret",),
        ),
        (
            'Set-Cookie: "session=set-cookie-secret; Path=/"',
            ("set-cookie-secret",),
        ),
        (
            'token = "quoted multiword token secret"',
            ("quoted multiword token secret",),
        ),
    ],
)
def test_error_body_excerpt_redacts_sensitive_text(message: str, secrets: tuple[str, ...]) -> None:
    excerpt = error_body_excerpt(json.dumps({"message": message}).encode())

    assert all(secret not in excerpt for secret in secrets)


def test_error_body_excerpt_redacts_sensitive_text_in_invalid_json() -> None:
    excerpt = error_body_excerpt(b'authorization: Bearer invalid-json-secret')

    assert "invalid-json-secret" not in excerpt
    assert "Bearer" not in excerpt


@pytest.mark.parametrize(
    "field",
    [
        "x-auth-token",
        "x-api-key",
        "sec-websocket-key",
        "proxy-authorization",
        "id_token",
        "refresh_token",
        "form-id-token",
        "websocket-key",
    ],
)
def test_error_body_excerpt_redacts_sensitive_json_fields(field: str) -> None:
    marker = f"{field}-marker"
    excerpt = error_body_excerpt(json.dumps({field: marker}).encode())

    assert marker not in excerpt


@pytest.mark.parametrize(
    "field",
    [
        "x-auth-token",
        "x-api-key",
        "sec-websocket-key",
        "proxy-authorization",
        "id_token",
        "refresh_token",
        "form-id-token",
        "websocket-key",
    ],
)
def test_error_body_excerpt_redacts_sensitive_form_fields(field: str) -> None:
    marker = f"{field}-form-marker"
    excerpt = error_body_excerpt(f"{field}={marker}".encode())

    assert marker not in excerpt


@pytest.mark.parametrize(
    "field",
    ["x-auth-token", "x-api-key", "sec-websocket-key", "proxy-authorization"],
)
def test_error_body_excerpt_redacts_sensitive_headers(field: str) -> None:
    marker = f"{field}-header-marker"
    value = f"Bearer {marker}" if field == "proxy-authorization" else marker
    excerpt = error_body_excerpt(f"{field}: {value}".encode())

    assert marker not in excerpt


def test_error_body_excerpt_redacts_multiline_authorization_record() -> None:
    excerpt = error_body_excerpt(b"authorization: Bearer\nnewline-sse-marker")

    assert "newline-sse-marker" not in excerpt
    assert "Bearer" not in excerpt
