"""
Tests for the multi-dimensional reward system.
"""

import pytest

from codedebug_env.models import (
    CodeDebugAction,
    CodeDebugState,
    ExecutionResult,
    ExecutionStatus,
    RewardConfig,
    TaskSpec,
)
from codedebug_env.server.diff_utils import DiffStats
from codedebug_env.server.reward import (
    compute_reward,
    duplicate_patch_penalty,
    early_solve_bonus,
    full_solve_bonus,
    hash_code,
    noop_patch_penalty,
    partial_test_credit,
    patch_efficiency_bonus,
    reasoning_quality_bonus,
    regression_penalty,
    runtime_timeout_penalty,
    syntax_penalty,
)


@pytest.fixture
def config() -> RewardConfig:
    return RewardConfig()


@pytest.fixture
def sample_task() -> TaskSpec:
    return TaskSpec(
        task_id="test_task",
        buggy_code="def f(): pass",
        test_code="def test_f(): assert True",
    )


@pytest.fixture
def base_state(sample_task: TaskSpec) -> CodeDebugState:
    return CodeDebugState(
        task=sample_task,
        original_code="def f(): pass",
        current_code="def f(): pass",
        baseline_passed=1,
        baseline_total=5,
        max_steps=10,
    )


class TestPartialTestCredit:
    def test_improvement_gives_positive(self, config):
        reward = partial_test_credit(3, 1, 1, 5, config)
        assert reward > 0

    def test_no_change_gives_zero(self, config):
        reward = partial_test_credit(3, 3, 1, 5, config)
        assert reward == 0.0

    def test_regression_gives_negative(self, config):
        reward = partial_test_credit(1, 3, 1, 5, config)
        assert reward < 0

    def test_zero_tests(self, config):
        assert partial_test_credit(0, 0, 0, 0, config) == 0.0


class TestFullSolveBonus:
    def test_solved_gives_bonus(self, config):
        assert full_solve_bonus(True, config) == config.full_solve_bonus

    def test_unsolved_gives_zero(self, config):
        assert full_solve_bonus(False, config) == 0.0


class TestRegressionPenalty:
    def test_regression_penalized(self, config):
        penalty = regression_penalty(2, 4, config)
        assert penalty < 0

    def test_no_regression(self, config):
        assert regression_penalty(4, 2, config) == 0.0

    def test_equal_passed(self, config):
        assert regression_penalty(3, 3, config) == 0.0


class TestSyntaxPenalty:
    def test_valid_syntax_no_penalty(self, config):
        assert syntax_penalty(True, config) == 0.0

    def test_invalid_syntax_penalized(self, config):
        assert syntax_penalty(False, config) == config.syntax_error_penalty


class TestRuntimePenalty:
    def test_timeout_penalized(self, config):
        penalty = runtime_timeout_penalty(ExecutionStatus.TIMEOUT, config)
        assert penalty == config.timeout_penalty

    def test_crash_penalized(self, config):
        penalty = runtime_timeout_penalty(ExecutionStatus.CRASH, config)
        assert penalty == config.runtime_penalty

    def test_success_no_penalty(self, config):
        assert runtime_timeout_penalty(ExecutionStatus.SUCCESS, config) == 0.0


class TestPatchEfficiency:
    def test_small_patch_bonus(self, config):
        stats = DiffStats(lines_added=2, lines_removed=1, is_noop=False)
        bonus = patch_efficiency_bonus(stats, config)
        assert bonus > 0

    def test_large_patch_penalty(self, config):
        stats = DiffStats(
            lines_added=50, lines_removed=30, is_noop=False
        )
        bonus = patch_efficiency_bonus(stats, config)
        assert bonus < 0

    def test_noop_no_bonus(self, config):
        stats = DiffStats(is_noop=True)
        assert patch_efficiency_bonus(stats, config) == 0.0


class TestReasoningBonus:
    def test_good_reasoning(self, config):
        action = CodeDebugAction(
            patched_code="code",
            reasoning="The bug is an off-by-one error in the loop boundary. Fixed by changing <= to <.",
        )
        bonus = reasoning_quality_bonus(action, config)
        assert bonus > 0

    def test_no_reasoning(self, config):
        action = CodeDebugAction(patched_code="code")
        assert reasoning_quality_bonus(action, config) == 0.0

    def test_too_short_reasoning(self, config):
        action = CodeDebugAction(patched_code="code", reasoning="fix")
        assert reasoning_quality_bonus(action, config) == 0.0

    def test_reasoning_with_bug_type(self, config):
        action = CodeDebugAction(
            patched_code="code",
            reasoning="The bug is an off-by-one error in the loop.",
            declare_bug_type=["off-by-one"],
        )
        bonus = reasoning_quality_bonus(action, config)
        assert bonus > config.reasoning_bonus  # Should get multiplier


class TestDuplicatePenalty:
    def test_duplicate_penalized(self, config):
        assert duplicate_patch_penalty("abc", ["abc", "def"], config) < 0

    def test_new_patch_no_penalty(self, config):
        assert duplicate_patch_penalty("xyz", ["abc", "def"], config) == 0.0


class TestNoopPenalty:
    def test_noop_penalized(self, config):
        stats = DiffStats(is_noop=True)
        assert noop_patch_penalty(stats, config) < 0

    def test_change_no_penalty(self, config):
        stats = DiffStats(is_noop=False)
        assert noop_patch_penalty(stats, config) == 0.0


class TestEarlySolveBonus:
    def test_early_solve(self, config):
        bonus = early_solve_bonus(True, 2, 10, config)
        assert bonus > 0

    def test_late_solve(self, config):
        bonus = early_solve_bonus(True, 9, 10, config)
        assert bonus >= 0  # Should be small

    def test_no_solve(self, config):
        assert early_solve_bonus(False, 5, 10, config) == 0.0


class TestComputeReward:
    """Integration test for full reward computation."""

    def test_improvement_positive_total(self, base_state, config):
        action = CodeDebugAction(
            patched_code="def f(): return 1",
            reasoning="Fixed the return value to pass more tests.",
        )
        exec_result = ExecutionResult(
            status=ExecutionStatus.RUNTIME_ERROR,
            syntax_valid=True,
            total_tests=5,
            passed=3,
            failed=2,
        )
        diff_stats = DiffStats(
            lines_added=1, lines_removed=1, is_noop=False, hunks=1
        )
        breakdown = compute_reward(action, exec_result, base_state, diff_stats, config)
        assert breakdown.total > 0
        assert breakdown.partial_test_credit > 0

    def test_syntax_error_negative(self, base_state, config):
        action = CodeDebugAction(patched_code="def (")
        exec_result = ExecutionResult(
            status=ExecutionStatus.SYNTAX_ERROR,
            syntax_valid=False,
        )
        diff_stats = DiffStats(is_noop=False, lines_changed=1)
        breakdown = compute_reward(action, exec_result, base_state, diff_stats, config)
        assert breakdown.total < 0
        assert breakdown.syntax_penalty < 0

    def test_reward_is_clipped(self, base_state, config):
        action = CodeDebugAction(patched_code="x")
        exec_result = ExecutionResult(
            status=ExecutionStatus.SYNTAX_ERROR, syntax_valid=False
        )
        diff_stats = DiffStats(is_noop=True)
        breakdown = compute_reward(action, exec_result, base_state, diff_stats, config)
        assert breakdown.total >= config.clip_min
        assert breakdown.total <= config.clip_max


class TestHashCode:
    def test_deterministic(self):
        h1 = hash_code("def f(): pass")
        h2 = hash_code("def f(): pass")
        assert h1 == h2

    def test_different_code(self):
        h1 = hash_code("def f(): pass")
        h2 = hash_code("def g(): pass")
        assert h1 != h2

    def test_whitespace_normalized(self):
        h1 = hash_code("  def f(): pass  ")
        h2 = hash_code("def f(): pass")
        assert h1 == h2
