import pytest

from zeta.providers.codex_errors import CodexHTTPError
from zeta.providers.transport import retryable_provider_error


@pytest.mark.parametrize("status_code", [520, 521, 522, 523, 524, 529])
def test_cloudflare_transient_statuses_are_retryable(status_code: int) -> None:
    error = CodexHTTPError("provider failure", status_code=status_code)

    assert retryable_provider_error(error)


@pytest.mark.parametrize("status_code", [525, 526, 527, 528, 530, 400, 401, 404])
def test_non_retryable_statuses_stay_non_retryable(status_code: int) -> None:
    error = CodexHTTPError("provider failure", status_code=status_code)

    assert not retryable_provider_error(error)
