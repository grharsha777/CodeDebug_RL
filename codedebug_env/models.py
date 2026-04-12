"""
Core data models for the CodeDebug-RL environment.

All models use Pydantic v2 for strong typing, validation, and serialization.
These types define the contract between agent and environment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────


class Difficulty(str, Enum):
    """Task difficulty tier, used for curriculum scheduling."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ExecutionStatus(str, Enum):
    """Outcome of a code execution attempt."""
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    CRASH = "crash"
    SKIPPED = "skipped"


class DoneReason(str, Enum):
    """Why an episode terminated."""
    SOLVED = "solved"
    MAX_STEPS = "max_steps"
    REPEATED_INVALID = "repeated_invalid"
    ENV_FAILURE = "env_failure"


class PatchFormat(str, Enum):
    """Supported patch submission formats."""
    FULL_REPLACE = "full_replace"
    UNIFIED_DIFF = "unified_diff"


# ─── Task Specification ──────────────────────────────────────────────────────


class TaskSpec(BaseModel):
    """
    Defines a single debugging task. Loaded from disk or dataset.
    """
    task_id: str = Field(..., description="Unique identifier for this task")
    difficulty: Difficulty = Field(Difficulty.MEDIUM, description="Difficulty tier")
    buggy_code: str = Field(..., description="Source code of the buggy program")
    canonical_filename: str = Field(
        "solution.py", description="Filename for the source under test"
    )
    test_code: str = Field(..., description="Pytest test suite contents")
    test_filename: str = Field(
        "test_solution.py", description="Filename for the test file"
    )
    description: str = Field("", description="Natural-language bug report / task description")
    reference_solution: str | None = Field(
        None, description="Hidden canonical solution (not exposed to agent)"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Semantic tags: syntax, logic, recursion, indexing, math, etc.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Test Results ─────────────────────────────────────────────────────────────


class TestResult(BaseModel):
    """Structured result for a single test case."""
    name: str
    passed: bool
    duration_s: float = 0.0
    error_message: str | None = None
    short_trace: str | None = None


class ExecutionResult(BaseModel):
    """Full result of executing the submitted code against the test suite."""
    status: ExecutionStatus
    syntax_valid: bool = True
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errored: int = 0
    skipped: int = 0
    test_results: list[TestResult] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    error_detail: str | None = None


# ─── Action ───────────────────────────────────────────────────────────────────


class CodeDebugAction(BaseModel):
    """
    Agent's action — a proposed code fix plus optional metadata.

    The primary field is `patched_code` containing either the full
    replacement source or a unified diff, controlled by `patch_format`.
    """
    patched_code: str = Field(
        ..., description="The patched source code or unified diff"
    )
    reasoning: str | None = Field(
        None,
        description="Agent's explanation of the identified bug and fix rationale",
    )
    patch_format: PatchFormat = Field(
        PatchFormat.FULL_REPLACE,
        description="Whether patched_code is a full file or unified diff",
    )
    declare_bug_type: list[str] | None = Field(
        None,
        description="Agent's declared bug categories, e.g. ['off-by-one', 'logic']",
    )
    expected_test_impact: str | None = Field(
        None,
        description="Agent's prediction of which tests should now pass",
    )
    commit_message: str | None = Field(
        None,
        description="Short human-readable summary of the change",
    )


# ─── Observation ──────────────────────────────────────────────────────────────


class CodeDebugObservation(BaseModel):
    """
    Rich observation returned to the agent after each step.

    Contains the current state of the code, test feedback, reward breakdown,
    and episode metadata. Designed for maximum agent informativeness.
    """
    task_id: str
    instruction: str = Field("", description="Task description / bug report")
    current_code: str
    previous_code: str | None = None
    test_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Counts: total, passed, failed, errored, skipped",
    )
    test_output: str = ""
    failed_tests: list[dict[str, Any]] = Field(default_factory=list)
    passed_tests: list[str] = Field(default_factory=list)
    syntax_valid: bool = True
    execution_status: str = ExecutionStatus.SKIPPED.value
    step_index: int = 0
    max_steps: int = 10
    reward_breakdown: dict[str, float] = Field(default_factory=dict)
    cumulative_score: float = 0.0
    done: bool = False
    done_reason: str | None = None
    diff_from_previous: str | None = None
    hint: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Episode State (server-side) ─────────────────────────────────────────────


class StepRecord(BaseModel):
    """Record of a single step in the episode."""
    step_index: int
    action: CodeDebugAction
    execution_result: ExecutionResult
    reward: float
    reward_breakdown: dict[str, float]
    code_snapshot: str
    timestamp_ms: int


class EpisodeMetrics(BaseModel):
    """Aggregate metrics for the episode. Used for telemetry & analysis."""
    total_steps: int = 0
    total_reward: float = 0.0
    solved: bool = False
    solve_step: int | None = None
    peak_pass_rate: float = 0.0
    regression_count: int = 0
    syntax_error_count: int = 0
    duplicate_patch_count: int = 0
    total_runtime_s: float = 0.0


class CodeDebugState(BaseModel):
    """
    Full internal episode state maintained by the environment server.

    Not sent to the agent — used for state management, reward computation,
    and episode reconstruction.
    """
    task: TaskSpec
    original_code: str
    current_code: str
    baseline_execution: ExecutionResult | None = None
    baseline_passed: int = 0
    baseline_total: int = 0
    best_passed: int = 0
    step_index: int = 0
    max_steps: int = 10
    history: list[StepRecord] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    solved: bool = False
    done: bool = False
    done_reason: DoneReason | None = None
    consecutive_invalid: int = 0
    max_consecutive_invalid: int = 3
    seen_patches: list[str] = Field(
        default_factory=list,
        description="Hashes of previously submitted patches for duplicate detection",
    )
    metrics: EpisodeMetrics = Field(default_factory=EpisodeMetrics)


# ─── Reward Configuration ────────────────────────────────────────────────────


class RewardConfig(BaseModel):
    """
    Tunable knobs for the multi-dimensional reward function.

    These defaults are calibrated for GRPO-style training where rewards
    in [-1, 1] range are typical, with sparse large bonuses.
    """
    # Core components
    partial_test_weight: float = 0.4
    full_solve_bonus: float = 1.0
    regression_penalty: float = -0.3
    syntax_error_penalty: float = -0.5
    runtime_penalty: float = -0.2
    timeout_penalty: float = -0.4

    # Efficiency
    patch_efficiency_weight: float = 0.1
    max_efficient_diff_lines: int = 20

    # Reasoning
    reasoning_bonus: float = 0.05
    reasoning_min_length: int = 20
    reasoning_max_length: int = 500

    # Shaping
    duplicate_penalty: float = -0.3
    noop_penalty: float = -0.2
    improvement_streak_bonus: float = 0.05
    early_solve_bonus_per_step: float = 0.02
    invalid_action_penalty: float = -0.4

    # Aggregation
    clip_min: float = 0.0
    clip_max: float = 1.0
    normalize: bool = False


class RewardBreakdown(BaseModel):
    """Transparent decomposition of the reward signal for a single step."""
    partial_test_credit: float = 0.0
    full_solve_bonus: float = 0.0
    regression_penalty: float = 0.0
    syntax_penalty: float = 0.0
    runtime_penalty: float = 0.0
    patch_efficiency: float = 0.0
    reasoning_bonus: float = 0.0
    duplicate_penalty: float = 0.0
    noop_penalty: float = 0.0
    streak_bonus: float = 0.0
    early_solve_bonus: float = 0.0
    invalid_action_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize all non-zero components."""
        return {k: v for k, v in self.model_dump().items() if v != 0.0 or k == "total"}
