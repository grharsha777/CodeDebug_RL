"""
Reward System — Multi-dimensional reward shaping for CodeDebug-RL.

This is the environment's secret weapon: a transparent, configurable,
multi-component reward function that provides rich gradient signal for
GRPO-style training. Each component addresses a specific aspect of
code debugging quality.

Components:
1. Partial test credit — proportional improvement
2. Full solve bonus — sparse terminal reward
3. Regression penalty — passing tests that now fail
4. Syntax error penalty — unparsable submissions
5. Runtime/timeout penalty — hanging or crashing code
6. Patch efficiency bonus — minimal targeted fixes
7. Reasoning quality bonus — plausible bug identification
8. Shaping terms — duplicate, no-op, streak, early solve

All components are independently configurable via RewardConfig.
"""

from __future__ import annotations

import hashlib
import logging

from codedebug_env.models import (
    CodeDebugAction,
    CodeDebugState,
    ExecutionResult,
    ExecutionStatus,
    RewardBreakdown,
    RewardConfig,
)
from codedebug_env.server.diff_utils import DiffStats

logger = logging.getLogger("codedebug.reward")


# ─── Individual Reward Components ─────────────────────────────────────────────


def partial_test_credit(
    current_passed: int,
    previous_passed: int,
    baseline_passed: int,
    total_tests: int,
    config: RewardConfig,
) -> float:
    """
    Reward proportional to improvement in test pass rate over baseline.

    Uses delta from previous step to encourage monotonic progress.
    Normalized by total tests so reward scale is consistent across tasks.
    """
    if total_tests == 0:
        return 0.0

    # Improvement over previous step
    delta = (current_passed - previous_passed) / total_tests

    if delta == 0:
        return 0.0  # No change this step → no partial credit

    # Also reward absolute progress from baseline (only when improving)
    progress = max(0, current_passed - baseline_passed) / total_tests

    # Weighted combination: mostly step-delta, some absolute progress
    score = 0.7 * delta + 0.3 * progress
    return score * config.partial_test_weight


def full_solve_bonus(
    all_passed: bool,
    config: RewardConfig,
) -> float:
    """Large terminal bonus when all tests pass."""
    return config.full_solve_bonus if all_passed else 0.0


def regression_penalty(
    current_passed: int,
    previous_passed: int,
    config: RewardConfig,
) -> float:
    """
    Penalize when previously passing tests start failing.
    Scaled by the number of regressions.
    """
    regressions = max(0, previous_passed - current_passed)
    if regressions == 0:
        return 0.0
    return config.regression_penalty * regressions


def syntax_penalty(
    syntax_valid: bool,
    config: RewardConfig,
) -> float:
    """Strong penalty for syntactically invalid submissions."""
    return 0.0 if syntax_valid else config.syntax_error_penalty


def runtime_timeout_penalty(
    status: ExecutionStatus,
    config: RewardConfig,
) -> float:
    """Penalize submissions that crash or timeout."""
    if status == ExecutionStatus.TIMEOUT:
        return config.timeout_penalty
    if status == ExecutionStatus.CRASH:
        return config.runtime_penalty
    return 0.0


def patch_efficiency_bonus(
    diff_stats: DiffStats,
    config: RewardConfig,
) -> float:
    """
    Reward minimal, targeted patches over large rewrites.

    A patch that changes fewer lines relative to the max threshold
    receives a higher efficiency bonus. No-ops get zero.
    """
    if diff_stats.is_noop:
        return 0.0

    churn = diff_stats.churn
    max_lines = config.max_efficient_diff_lines

    if churn <= max_lines:
        # Linear reward: smaller patches get more credit
        efficiency = 1.0 - (churn / max_lines)
    else:
        # Penalty that increases with excess churn
        excess_ratio = churn / max_lines
        efficiency = -0.2 * (excess_ratio - 1.0)

    return efficiency * config.patch_efficiency_weight


def reasoning_quality_bonus(
    action: CodeDebugAction,
    config: RewardConfig,
) -> float:
    """
    Small bonus for providing concise, relevant reasoning.

    Heuristics:
    - Must have non-empty reasoning
    - Must be within length bounds (too short = vague, too long = rambling)
    - Bonus scaled by bug type declaration (shows understanding)
    """
    if not action.reasoning:
        return 0.0

    reasoning = action.reasoning.strip()
    length = len(reasoning)

    if length < config.reasoning_min_length:
        return 0.0  # too short to be meaningful
    if length > config.reasoning_max_length:
        return config.reasoning_bonus * 0.5  # penalize verbosity

    base = config.reasoning_bonus

    # Bonus if agent also declares bug type (shows structured thinking)
    if action.declare_bug_type and len(action.declare_bug_type) > 0:
        base *= 1.5

    return min(base, config.reasoning_bonus * 2)  # cap


def duplicate_patch_penalty(
    code_hash: str,
    seen_hashes: list[str],
    config: RewardConfig,
) -> float:
    """Penalize re-submitting identical patches."""
    if code_hash in seen_hashes:
        return config.duplicate_penalty
    return 0.0


def noop_patch_penalty(
    diff_stats: DiffStats,
    config: RewardConfig,
) -> float:
    """Penalize submitting code identical to the previous version."""
    if diff_stats.is_noop:
        return config.noop_penalty
    return 0.0


def improvement_streak_bonus(
    state: CodeDebugState,
    current_passed: int,
    config: RewardConfig,
) -> float:
    """
    Bonus for consecutive steps that improve the pass count.
    Encourages sustained monotonic progress.
    """
    if len(state.history) < 2:
        return 0.0

    # Check if last N steps were all improvements
    streak = 0
    prev_passed = current_passed
    for record in reversed(state.history):
        step_passed = record.execution_result.passed
        if step_passed < prev_passed:
            break
        streak += 1
        prev_passed = step_passed

    if streak >= 2:
        return config.improvement_streak_bonus * min(streak, 5)
    return 0.0


def early_solve_bonus(
    solved: bool,
    step_index: int,
    max_steps: int,
    config: RewardConfig,
) -> float:
    """Bonus for solving early — fewer steps = more efficient agent."""
    if not solved:
        return 0.0
    remaining = max_steps - step_index - 1
    return config.early_solve_bonus_per_step * remaining


# ─── Aggregation ──────────────────────────────────────────────────────────────


def compute_reward(
    action: CodeDebugAction,
    execution_result: ExecutionResult,
    state: CodeDebugState,
    diff_stats: DiffStats,
    config: RewardConfig | None = None,
) -> RewardBreakdown:
    """
    Compute the full multi-dimensional reward for a single step.

    Aggregates all individual components, clips, and optionally normalizes.
    Returns a transparent RewardBreakdown for observability.
    """
    config = config or RewardConfig()

    # Determine previous pass count
    if state.history:
        prev_passed = state.history[-1].execution_result.passed
    else:
        prev_passed = state.baseline_passed

    current_passed = execution_result.passed
    total_tests = execution_result.total_tests
    all_tests_pass = (
        execution_result.status == ExecutionStatus.SUCCESS
        and current_passed == total_tests
        and total_tests > 0
    )

    # Compute code hash for duplicate detection
    code_hash = hashlib.sha256(action.patched_code.encode()).hexdigest()[:16]

    # Build reward breakdown
    breakdown = RewardBreakdown()

    # 1. Partial test credit
    if execution_result.syntax_valid and total_tests > 0:
        breakdown.partial_test_credit = partial_test_credit(
            current_passed, prev_passed, state.baseline_passed, total_tests, config
        )

    # 2. Full solve bonus
    breakdown.full_solve_bonus = full_solve_bonus(all_tests_pass, config)

    # 3. Regression penalty
    if execution_result.syntax_valid:
        breakdown.regression_penalty = regression_penalty(
            current_passed, prev_passed, config
        )

    # 4. Syntax penalty
    breakdown.syntax_penalty = syntax_penalty(execution_result.syntax_valid, config)

    # 5. Runtime/timeout penalty
    breakdown.runtime_penalty = runtime_timeout_penalty(execution_result.status, config)

    # 6. Patch efficiency
    if execution_result.syntax_valid:
        breakdown.patch_efficiency = patch_efficiency_bonus(diff_stats, config)

    # 7. Reasoning quality
    breakdown.reasoning_bonus = reasoning_quality_bonus(action, config)

    # 8. Shaping terms
    breakdown.duplicate_penalty = duplicate_patch_penalty(
        code_hash, state.seen_patches, config
    )
    breakdown.noop_penalty = noop_patch_penalty(diff_stats, config)
    breakdown.streak_bonus = improvement_streak_bonus(state, current_passed, config)
    breakdown.early_solve_bonus = early_solve_bonus(
        all_tests_pass, state.step_index, state.max_steps, config
    )

    # Aggregate
    total = (
        breakdown.partial_test_credit
        + breakdown.full_solve_bonus
        + breakdown.regression_penalty
        + breakdown.syntax_penalty
        + breakdown.runtime_penalty
        + breakdown.patch_efficiency
        + breakdown.reasoning_bonus
        + breakdown.duplicate_penalty
        + breakdown.noop_penalty
        + breakdown.streak_bonus
        + breakdown.early_solve_bonus
        + breakdown.invalid_action_penalty
    )

    # Clip
    total = max(config.clip_min, min(config.clip_max, total))
    breakdown.total = round(total, 6)

    return breakdown


def hash_code(code: str) -> str:
    """Deterministic hash of code content for duplicate detection."""
    return hashlib.sha256(code.strip().encode()).hexdigest()[:16]
