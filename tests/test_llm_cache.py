import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from evals.llm_cache import CachedModel


def test_identical_runs_replay_from_cache(tmp_path):
    calls = []

    def respond(messages, info):
        calls.append(1)
        return ModelResponse(parts=[TextPart(f"answer {len(calls)}")])

    def make():
        return CachedModel(FunctionModel(respond), tmp_path / "cache.db")

    agent = Agent(None, instructions="Be brief.")
    first = asyncio.run(agent.run("What is 2+2?", model=make()))
    replay_model = make()
    second = asyncio.run(agent.run("What is 2+2?", model=replay_model))
    assert first.output == second.output == "answer 1"
    assert len(calls) == 1 and replay_model.hits == 1          # timestamps etc. don't break the key

    different = asyncio.run(agent.run("What is 3+3?", model=make()))
    assert different.output == "answer 2" and len(calls) == 2
