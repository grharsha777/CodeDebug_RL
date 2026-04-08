"""
Tests for the core environment logic — reset, step, termination.
"""

import pytest

from codedebug_env.models import CodeDebugAction
from codedebug_env.server.environment import CodeDebugEnvironment
from codedebug_env.server.task_loader import TaskLoader


@pytest.fixture
def env() -> CodeDebugEnvironment:
    """Create a fresh environment with built-in tasks."""
    loader = TaskLoader()
    loader.load_builtin()
    return CodeDebugEnvironment(
        task_loader=loader,
        max_steps=5,
        execution_timeout_s=15.0,
    )


class TestReset:
    """Tests for environment reset."""

    def test_reset_returns_observation(self, env: CodeDebugEnvironment):
        obs = env.reset(task_id="builtin_001_fizzbuzz")
        assert obs.task_id == "builtin_001_fizzbuzz"
        assert obs.step_index == 0
        assert obs.done is False
        assert len(obs.current_code) > 0

    def test_reset_shows_failing_tests(self, env: CodeDebugEnvironment):
        obs = env.reset(task_id="builtin_001_fizzbuzz")
        # The buggy fizzbuzz should have some failing tests
        assert obs.test_summary.get("total", 0) > 0
        assert obs.test_summary.get("failed", 0) > 0

    def test_reset_with_difficulty(self, env: CodeDebugEnvironment):
        obs = env.reset(difficulty="easy")
        assert obs.task_id is not None
        assert obs.metadata.get("difficulty") == "easy"

    def test_reset_with_seed_is_deterministic(self, env: CodeDebugEnvironment):
        obs1 = env.reset(seed=42)
        task1 = obs1.task_id
        obs2 = env.reset(seed=42)
        task2 = obs2.task_id
        assert task1 == task2

    def test_reset_unknown_task_raises(self, env: CodeDebugEnvironment):
        with pytest.raises(ValueError, match="not found"):
            env.reset(task_id="nonexistent_task")


class TestStep:
    """Tests for environment step."""

    def test_step_with_correct_fix(self, env: CodeDebugEnvironment):
        env.reset(task_id="builtin_001_fizzbuzz")
        correct_code = '''\
def fizzbuzz(n: int) -> list[str]:
    """Return FizzBuzz sequence from 1 to n."""
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
'''
        action = CodeDebugAction(patched_code=correct_code)
        obs, reward, done, info = env.step(action)

        assert done is True
        assert obs.done_reason == "solved"
        assert reward > 0
        assert obs.test_summary.get("failed", -1) == 0

    def test_step_with_syntax_error(self, env: CodeDebugEnvironment):
        env.reset(task_id="builtin_001_fizzbuzz")
        action = CodeDebugAction(
            patched_code="def fizzbuzz(n):\n    return [\n"
        )
        obs, reward, done, info = env.step(action)

        assert obs.syntax_valid is False
        assert reward < 0
        assert obs.execution_status == "syntax_error"

    def test_step_increments_index(self, env: CodeDebugEnvironment):
        env.reset(task_id="builtin_001_fizzbuzz")
        action = CodeDebugAction(patched_code="def fizzbuzz(n): return []")
        obs, _, _, _ = env.step(action)
        assert obs.step_index == 1

    def test_step_without_reset_raises(self, env: CodeDebugEnvironment):
        with pytest.raises(RuntimeError, match="not initialized"):
            env.step(CodeDebugAction(patched_code="x = 1"))

    def test_step_after_done_raises(self, env: CodeDebugEnvironment):
        env.reset(task_id="builtin_001_fizzbuzz")
        correct_code = 'def fizzbuzz(n: int) -> list[str]:\n    result = []\n    for i in range(1, n + 1):\n        if i % 3 == 0 and i % 5 == 0:\n            result.append("FizzBuzz")\n        elif i % 3 == 0:\n            result.append("Fizz")\n        elif i % 5 == 0:\n            result.append("Buzz")\n        else:\n            result.append(str(i))\n    return result\n'
        action = CodeDebugAction(patched_code=correct_code)
        env.step(action)  # Should solve it
        with pytest.raises(RuntimeError, match="terminated"):
            env.step(action)


class TestTermination:
    """Tests for episode termination conditions."""

    def test_max_steps_termination(self, env: CodeDebugEnvironment):
        env.reset(task_id="builtin_001_fizzbuzz")
        action = CodeDebugAction(patched_code="def fizzbuzz(n): return []")

        done = False
        for _ in range(10):
            obs, _, done, _ = env.step(action)
            if done:
                break

        assert done is True
        assert obs.done_reason in ("max_steps", "repeated_invalid")

    def test_repeated_invalid_termination(self):
        loader = TaskLoader()
        loader.load_builtin()
        env = CodeDebugEnvironment(
            task_loader=loader,
            max_steps=20,
            max_consecutive_invalid=2,
        )
        env.reset(task_id="builtin_001_fizzbuzz")

        # Submit syntax errors repeatedly
        bad_code = "def ("
        for _ in range(3):
            obs, _, done, _ = env.step(CodeDebugAction(patched_code=bad_code))
            if done:
                break

        assert done is True
        assert obs.done_reason == "repeated_invalid"


class TestState:
    """Tests for environment state."""

    def test_state_before_reset(self, env: CodeDebugEnvironment):
        state = env.get_state()
        assert state["status"] == "not_initialized"

    def test_state_after_reset(self, env: CodeDebugEnvironment):
        env.reset(task_id="builtin_001_fizzbuzz")
        state = env.get_state()
        assert state["task_id"] == "builtin_001_fizzbuzz"
        assert state["step_index"] == 0
        assert state["done"] is False

    def test_state_tracks_progress(self, env: CodeDebugEnvironment):
        env.reset(task_id="builtin_001_fizzbuzz")
        env.step(CodeDebugAction(patched_code="def fizzbuzz(n): return []"))
        state = env.get_state()
        assert state["total_steps_taken"] == 1
