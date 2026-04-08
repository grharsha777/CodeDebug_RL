"""
Tests for codedebug_env.models — Pydantic model validation and serialization.
"""

from codedebug_env.models import (
    CodeDebugAction,
    CodeDebugObservation,
    Difficulty,
    DoneReason,
    ExecutionResult,
    ExecutionStatus,
    PatchFormat,
    RewardBreakdown,
    RewardConfig,
    TaskSpec,
    TestResult,
)


class TestCodeDebugAction:
    """Tests for the Action model."""

    def test_minimal_action(self):
        action = CodeDebugAction(patched_code="print('hello')")
        assert action.patched_code == "print('hello')"
        assert action.reasoning is None
        assert action.patch_format == PatchFormat.FULL_REPLACE

    def test_full_action(self):
        action = CodeDebugAction(
            patched_code="def foo(): return 1",
            reasoning="Fixed the return value",
            patch_format=PatchFormat.FULL_REPLACE,
            declare_bug_type=["logic", "off-by-one"],
            expected_test_impact="test_foo should pass",
            commit_message="fix: correct return value",
        )
        assert len(action.declare_bug_type) == 2
        assert action.commit_message == "fix: correct return value"

    def test_action_serialization(self):
        action = CodeDebugAction(patched_code="x = 1", reasoning="test")
        data = action.model_dump()
        assert data["patched_code"] == "x = 1"
        assert "reasoning" in data


class TestTaskSpec:
    """Tests for the TaskSpec model."""

    def test_minimal_task(self):
        task = TaskSpec(
            task_id="test_001",
            buggy_code="def f(): pass",
            test_code="def test_f(): assert f() is None",
        )
        assert task.difficulty == Difficulty.MEDIUM
        assert task.canonical_filename == "solution.py"

    def test_full_task(self):
        task = TaskSpec(
            task_id="test_002",
            difficulty=Difficulty.HARD,
            buggy_code="code",
            test_code="tests",
            description="Fix the bug",
            reference_solution="fixed code",
            tags=["logic", "recursion"],
        )
        assert task.difficulty == Difficulty.HARD
        assert len(task.tags) == 2


class TestExecutionResult:
    """Tests for the ExecutionResult model."""

    def test_syntax_error_result(self):
        result = ExecutionResult(
            status=ExecutionStatus.SYNTAX_ERROR,
            syntax_valid=False,
            error_detail="SyntaxError at line 1",
        )
        assert not result.syntax_valid
        assert result.total_tests == 0

    def test_success_result(self):
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            total_tests=5,
            passed=5,
            failed=0,
            test_results=[
                TestResult(name="test_1", passed=True, duration_s=0.01),
            ],
        )
        assert result.passed == 5


class TestRewardBreakdown:
    """Tests for the RewardBreakdown model."""

    def test_zero_reward(self):
        rb = RewardBreakdown()
        assert rb.total == 0.0

    def test_to_dict_filters_zeros(self):
        rb = RewardBreakdown(partial_test_credit=0.3, syntax_penalty=-0.5, total=-0.2)
        d = rb.to_dict()
        assert "partial_test_credit" in d
        assert "syntax_penalty" in d
        assert "total" in d
        # Zero values (except total) should be filtered
        assert "full_solve_bonus" not in d


class TestRewardConfig:
    """Tests for the RewardConfig model."""

    def test_defaults(self):
        config = RewardConfig()
        assert config.partial_test_weight == 0.4
        assert config.full_solve_bonus == 1.0
        assert config.clip_min == -2.0
        assert config.clip_max == 2.0

    def test_custom_config(self):
        config = RewardConfig(
            full_solve_bonus=2.0,
            syntax_error_penalty=-1.0,
        )
        assert config.full_solve_bonus == 2.0
        assert config.syntax_error_penalty == -1.0


class TestObservation:
    """Tests for the CodeDebugObservation model."""

    def test_minimal_observation(self):
        obs = CodeDebugObservation(
            task_id="test",
            current_code="code",
        )
        assert obs.step_index == 0
        assert obs.done is False
        assert obs.cumulative_score == 0.0

    def test_full_observation(self):
        obs = CodeDebugObservation(
            task_id="test",
            instruction="Fix the bug",
            current_code="def f(): return 1",
            previous_code="def f(): return 0",
            test_summary={"total": 3, "passed": 2, "failed": 1},
            syntax_valid=True,
            execution_status="runtime_error",
            step_index=2,
            max_steps=10,
            reward_breakdown={"partial_test_credit": 0.2, "total": 0.2},
            cumulative_score=0.5,
            done=False,
        )
        assert obs.test_summary["passed"] == 2


class TestEnums:
    """Tests for enum values."""

    def test_difficulty_values(self):
        assert Difficulty.EASY.value == "easy"
        assert Difficulty.MEDIUM.value == "medium"
        assert Difficulty.HARD.value == "hard"

    def test_execution_status_values(self):
        assert ExecutionStatus.SUCCESS.value == "success"
        assert ExecutionStatus.SYNTAX_ERROR.value == "syntax_error"
        assert ExecutionStatus.TIMEOUT.value == "timeout"

    def test_done_reason_values(self):
        assert DoneReason.SOLVED.value == "solved"
        assert DoneReason.MAX_STEPS.value == "max_steps"
