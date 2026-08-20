import json
import time
from urllib.parse import quote

import pytest

from zeta.auth import _redact_multipart, error_body_excerpt


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
    "auth_token",
    "api_secret",
    "consumer_secret",
    "signing_key",
    "credentials",
    "authToken",
    "apiSecret",
    "consumerSecret",
    "signingKey",
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


def test_error_body_excerpt_rejects_leading_space_boundary_lookalikes() -> None:
    body = (
        b"Content-Type: multipart/form-data; boundary=outer\r\n\r\n"
        b"--outer\r\n"
        b"Content-Disposition: form-data; name=password\r\n\r\n"
        b" --outer\r\n"
        b"leading-space-boundary-lookalike\r\n"
        b"--outer--\r\n"
    )

    assert "leading-space-boundary-lookalike" not in error_body_excerpt(body)


def test_error_body_excerpt_recovers_when_inner_boundary_is_not_closed() -> None:
    body = (
        b"Content-Type: multipart/mixed; boundary=outer\r\n\r\n"
        b"--outer\r\n"
        b"Content-Disposition: form-data; name=container\r\n"
        b"Content-Type: multipart/mixed; boundary=inner\r\n\r\n"
        b"--inner\r\n"
        b"Content-Disposition: form-data; name=ordinary\r\n\r\n"
        b"ordinary value\r\n"
        b"--outer\r\n"
        b"Content-Disposition: form-data; name=password\r\n\r\n"
        b"missing-inner-close-marker\r\n"
        b"--outer--\r\n"
    )

    assert "missing-inner-close-marker" not in error_body_excerpt(body)


@pytest.mark.parametrize(
    "disposition",
    [
        "form-data; (nested (comment)) name=password",
        'form-data; (comment) name="password"',
        "form-data; name (trailing)=password",
    ],
)
def test_error_body_excerpt_strips_folded_mime_comments(disposition: str) -> None:
    body = (
        "Content-Type: multipart/form-data; boundary=outer\r\n\r\n"
        "--outer\r\n"
        f"Content-Disposition: {disposition}\r\n\r\n"
        "folded-comment-marker\r\n"
        "--outer--\r\n"
    ).encode()

    assert "folded-comment-marker" not in error_body_excerpt(body)


def test_error_body_excerpt_handles_deep_multipart_without_recursion() -> None:
    depth = 2000
    lines = [f"Content-Type: multipart/mixed; boundary=b0\r\n", "\r\n"]
    for index in range(depth):
        if index + 1 < depth:
            lines.extend(
                [
                    f"--b{index}\r\n",
                    "Content-Disposition: form-data; name=container\r\n",
                    f"Content-Type: multipart/mixed; boundary=b{index + 1}\r\n",
                ]
            )
        else:
            lines.extend(
                [
                    f"--b{index}\r\n",
                    "Content-Disposition: form-data; name=password\r\n",
                ]
            )
        lines.append("\r\n")
    lines.append("deep-nesting-marker\r\n")
    for index in range(depth - 1, -1, -1):
        lines.append(f"--b{index}--\r\n")

    depths: list[int] = [1]
    redacted = _redact_multipart("".join(lines), depth_observer=depths)

    assert "deep-nesting-marker" not in redacted
    assert max(depths) == depth


def test_error_body_excerpt_joins_large_folded_header_once() -> None:
    folded = " x\r\n" * (792_000 // 3)
    body = (
        "Content-Type: multipart/form-data; boundary=outer\r\n\r\n"
        "--outer\r\n"
        "Content-Disposition: form-data;\r\n"
        f"{folded} name=password\r\n\r\n"
        "folded-header-timing-marker\r\n"
        "--outer--\r\n"
    ).encode()
    started = time.perf_counter()

    excerpt = error_body_excerpt(body)

    elapsed = time.perf_counter() - started
    assert "folded-header-timing-marker" not in excerpt
    assert elapsed < 0.2


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
