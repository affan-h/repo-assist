"""Rate-limited model requests are retried after the server's suggested delay."""

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from repo_assist import retry


def _flaky(errors: list[ModelHTTPError]) -> FunctionModel:
    def respond(messages, info):
        if errors:
            raise errors.pop(0)
        return ModelResponse(parts=[TextPart("ok")])
    return FunctionModel(respond)


def _error(status: int, message: str) -> ModelHTTPError:
    return ModelHTTPError(status_code=status, model_name="m", body={"error": {"message": message}})


@pytest.fixture
def sleeps(monkeypatch):
    waited = []

    async def fake_sleep(seconds):
        waited.append(seconds)
    monkeypatch.setattr(retry.asyncio, "sleep", fake_sleep)
    return waited


def test_rate_limit_waits_as_asked_then_succeeds(sleeps):
    model = retry.RetryingModel(_flaky([_error(429, "Please retry in 18.4s."), _error(503, "overloaded")]))
    result = asyncio.run(Agent(model).run("hi"))
    assert result.output == "ok"
    assert sleeps == [pytest.approx(19.4), 30]   # server's delay + 1s; then a default backoff


@pytest.mark.parametrize("error", [_error(429, "Quota exceeded for metric: GenerateRequestsPerDayPerProject"),
                                   _error(400, "bad request")])
def test_daily_quota_and_client_errors_are_not_retried(sleeps, error):
    with pytest.raises(ModelHTTPError):
        asyncio.run(Agent(retry.RetryingModel(_flaky([error]))).run("hi"))
    assert sleeps == []
