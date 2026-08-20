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
        (
            '"AUTHORIZATION" : Basic dXNlcjpwYXNz',
            ("Basic", "dXNlcjpwYXNz"),
        ),
        (
            'authorization:\tDIGEST username="digest-user", realm="digest-realm"',
            ("DIGEST", "digest-user", "digest-realm"),
        ),
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
