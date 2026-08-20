import json
import time
from urllib.parse import quote

import pytest

from zeta.auth import error_body_excerpt


SENSITIVE_NAMES = (
    "authorization",
    "proxy-authorization",
    "www-authenticate",
    "authentication",
    "x-api-key",
    "x-auth-token",
    "x-amz-security-token",
    "x-amz-signature",
    "x-goog-api-key",
    "anthropic-api-key",
    "openai-api-key",
    "sec-websocket-key",
    "sec-websocket-accept",
    "cookie",
    "set-cookie",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "access_token",
    "refresh_token",
    "id_token",
    "token",
    "api_key",
    "apikey",
)
NEW_NAME_VARIANTS = (
    "accessToken",
    "access_token",
    "refreshToken",
    "refresh_token",
    "idToken",
    "id_token",
    "clientSecret",
    "client_secret",
    "sessionToken",
    "session_token",
    "privateKey",
    "private_key",
    "clientAssertion",
    "client_assertion",
    "deviceCode",
    "device_code",
    "awsAccessKeyId",
    "aws_access_key_id",
    "awsSecretAccessKey",
    "aws_secret_access_key",
    "xAmzCredential",
    "x_amz_credential",
    "xGoogCredential",
    "x_goog_credential",
    "xGoogSignature",
    "x_goog_signature",
)


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


@pytest.mark.parametrize("scheme", [
    "Bearer",
    "Basic",
    "Digest",
    "Token",
    "ApiKey",
    "Foo",
    "1FooScheme",
    "!Bar",
    "#Baz",
])
def test_error_body_excerpt_redacts_any_authorization_scheme(scheme: str) -> None:
    marker = "scheme-matrix-marker"
    excerpt = error_body_excerpt(
        json.dumps({"message": f"authorization: {scheme} {marker}"}).encode()
    )

    assert marker not in excerpt
    assert scheme not in excerpt


@pytest.mark.parametrize("field", SENSITIVE_NAMES)
@pytest.mark.parametrize("body_format", ["json", "header", "form", "multipart"])
def test_error_body_excerpt_redacts_sensitive_names_in_all_formats(
    field: str, body_format: str
) -> None:
    marker = "format-matrix-marker"
    if body_format == "json":
        body = json.dumps({field: marker}).encode()
    elif body_format == "header":
        body = f"{field}: {marker}".encode()
    elif body_format == "form":
        body = f"{quote(field)}={quote(marker)}".encode()
    else:
        body = (
            f'--boundary\r\nContent-Disposition: form-data; name="{field}"\r\n'
            f"\r\n{marker}\r\n--boundary--\r\n"
        ).encode()

    excerpt = error_body_excerpt(body)

    assert marker not in excerpt


@pytest.mark.parametrize(
    ("encoded_field", "marker"),
    [
        ("id%5Ftoken", "encoded-id-token-marker"),
        ("refresh%5Ftoken", "encoded-refresh-token-marker"),
    ],
)
def test_error_body_excerpt_percent_decodes_form_field_names(
    encoded_field: str, marker: str
) -> None:
    excerpt = error_body_excerpt(f"{encoded_field}={marker}".encode())

    assert marker not in excerpt


@pytest.mark.parametrize("field", NEW_NAME_VARIANTS)
@pytest.mark.parametrize("body_format", ["json", "header", "form", "multipart"])
def test_error_body_excerpt_redacts_camel_and_snake_names(
    field: str, body_format: str
) -> None:
    marker = "expanded-name-marker"
    if body_format == "json":
        body = json.dumps({field: marker}).encode()
    elif body_format == "header":
        body = f"{field}: {marker}".encode()
    elif body_format == "form":
        body = f"{quote(field)}={quote(marker)}".encode()
    else:
        body = (
            f'--boundary\r\nContent-Disposition: form-data; name="{field}"\r\n'
            f"\r\n{marker}\r\n--boundary--\r\n"
        ).encode()

    assert marker not in error_body_excerpt(body)


def test_error_body_excerpt_tracks_nested_multipart_boundary() -> None:
    body = (
        b"Content-Type: multipart/mixed; boundary=outer\r\n\r\n"
        b"--outer\r\n"
        b"Content-Disposition: form-data; name=container\r\n"
        b"Content-Type: multipart/mixed; boundary=inner\r\n\r\n"
        b"--inner\r\n"
        b"Content-Disposition: form-data; name=password\r\n\r\n"
        b"nested-marker\r\n"
        b"--inner--\r\n"
        b"--outer--\r\n"
    )

    assert "nested-marker" not in error_body_excerpt(body)


def test_error_body_excerpt_redacts_dashed_multipart_value() -> None:
    body = (
        b"Content-Type: multipart/form-data; boundary=outer\r\n\r\n"
        b"--outer\r\n"
        b"Content-Disposition: form-data; name=password\r\n\r\n"
        b"--dash-marker\r\n"
        b"--outer--\r\n"
    )

    assert "dash-marker" not in error_body_excerpt(body)


def test_error_body_excerpt_joins_folded_multipart_headers() -> None:
    body = (
        b"Content-Type: multipart/form-data; boundary=outer\r\n\r\n"
        b"--outer\r\n"
        b"Content-Disposition: form-data;\r\n"
        b" name=password\r\n\r\n"
        b"folded-marker\r\n"
        b"--outer--\r\n"
    )

    assert "folded-marker" not in error_body_excerpt(body)


@pytest.mark.parametrize(
    ("size_kib", "budget_seconds"),
    [(512, 0.01), (4096, 0.1), (16384, 0.4)],
)
def test_error_body_excerpt_handles_later_colon_records_quickly(
    size_kib: int, budget_seconds: float
) -> None:
    body = b"authorization: Foo\n" + b"x\n" * ((size_kib * 1024 - 19) // 2)
    body += b"later: safe-value"
    started = time.perf_counter()

    excerpt = error_body_excerpt(body)

    elapsed = time.perf_counter() - started
    assert excerpt.startswith("authorization:")
    assert elapsed < budget_seconds
