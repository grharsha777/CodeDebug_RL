"""
CodeDebugEnvironment — Core RL environment implementing reset/step/state.

This is the main environment class that orchestrates:
- Episode lifecycle (reset → step → done)
- Action validation and patch application
- Execution delegation to the executor
- Reward computation via the reward module
- State management and observation construction

Follows OpenEnv conventions: reset() → observation,
step(action) → (observation, reward, done, info).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from codedebug_env.models import (
    CodeDebugAction,
    CodeDebugObservation,
    CodeDebugState,
    DoneReason,
    EpisodeMetrics,
    ExecutionStatus,
    PatchFormat,
    RewardBreakdown,
    RewardConfig,
    StepRecord,
)
from codedebug_env.server.diff_utils import (
    apply_unified_diff,
    compute_diff_stats,
    compute_unified_diff,
    truncate_diff,
)
from codedebug_env.server.executor import execute_submission
from codedebug_env.server.reward import compute_reward, hash_code
from codedebug_env.server.sandbox import SandboxConfig
from codedebug_env.server.task_loader import TaskLoader
from codedebug_env.server.telemetry import (
    EpisodeSummary,
    StepMetric,
    TelemetryCollector,
)

logger = logging.getLogger("codedebug.environment")


class CodeDebugEnvironment:
    """
    OpenEnv-compatible RL environment for iterative code debugging.

    The environment presents a buggy Python program and test suite to the agent.
    At each step, the agent submits a patched version of the code.
    The environment executes the tests, computes a multi-dimensional reward,
    and returns a rich observation.

    Episodes terminate when all tests pass, max steps are reached,
    or repeated invalid actions occur.
    """

    def __init__(
        self,
        task_loader: TaskLoader | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 10,
        max_consecutive_invalid: int = 3,
        execution_timeout_s: float = 30.0,
        sandbox_config: SandboxConfig | None = None,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        self.task_loader = task_loader or self._default_loader()
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.max_consecutive_invalid = max_consecutive_invalid
        self.execution_timeout_s = execution_timeout_s
        self.sandbox_config = sandbox_config or SandboxConfig()
        self.telemetry = telemetry or TelemetryCollector()

        self._state: CodeDebugState | None = None
        self._episode_id: str = ""
        self._episode_start: float = 0.0

    @staticmethod
    def _default_loader() -> TaskLoader:
        loader = TaskLoader()
        loader.load_builtin()
        return loader

    @property
    def state(self) -> CodeDebugState | None:
        """Current episode state (None if no episode active)."""
        return self._state

    # ─── Reset ────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: str | None = None,
        difficulty: str | None = None,
        seed: int | None = None,
    ) -> CodeDebugObservation:
        """
        Start a new episode.

        Args:
            task_id: Specific task to load (if None, samples randomly)
            difficulty: Filter by difficulty tier
            seed: Random seed for deterministic task selection

        Returns:
            Initial observation with buggy code and failing test report
        """
        self._episode_id = uuid.uuid4().hex[:12]
        self._episode_start = time.time()

        # Load task
        if task_id:
            task = self.task_loader.get_task(task_id)
            if task is None:
                raise ValueError(f"Task not found: {task_id}")
        else:
            task = self.task_loader.sample(difficulty=difficulty, seed=seed)

        logger.info(
            "Episode %s: reset with task %s (%s)",
            self._episode_id,
            task.task_id,
            task.difficulty.value,
        )

        # Run baseline tests on the buggy code (non-fatal — execution may fail in restricted envs)
        try:
            baseline_result = execute_submission(
                source_code=task.buggy_code,
                test_code=task.test_code,
                source_filename=task.canonical_filename,
                test_filename=task.test_filename,
                timeout_s=self.execution_timeout_s,
                sandbox_config=self.sandbox_config,
            )
        except Exception as exc:
            logger.warning("Baseline execution failed (non-fatal): %s", exc)
            baseline_result = ExecutionResult(
                status=ExecutionStatus.CRASH,
                syntax_valid=True,
                error_detail=f"Baseline execution unavailable: {exc}",
            )

        # Initialize episode state
        self._state = CodeDebugState(
            task=task,
            original_code=task.buggy_code,
            current_code=task.buggy_code,
            baseline_execution=baseline_result,
            baseline_passed=baseline_result.passed,
            baseline_total=baseline_result.total_tests,
            best_passed=baseline_result.passed,
            step_index=0,
            max_steps=self.max_steps,
            cumulative_reward=0.0,
            metrics=EpisodeMetrics(),
        )

        self.telemetry.log_event(
            "episode_reset",
            episode_id=self._episode_id,
            task_id=task.task_id,
            baseline_passed=baseline_result.passed,
            total_tests=baseline_result.total_tests,
        )

        # Build initial observation
        return self._build_observation(
            execution_result_dict=self._execution_to_obs(baseline_result),
            reward_breakdown=RewardBreakdown(),
        )

    # ─── Step ─────────────────────────────────────────────────────────────

    def step(
        self, action: CodeDebugAction
    ) -> tuple[CodeDebugObservation, float, bool, dict[str, Any]]:
        """
        Execute one debugging step.

        Args:
            action: Agent's proposed code patch

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._state is None:
            raise RuntimeError(
                "Environment not initialized: session expired or not started. "
                "Please click 'Reset' or refresh to start a new task."
            )

        if self._state.done:
            raise RuntimeError("Episode already terminated. Call reset() to start a new one.")

        state = self._state
        step_start = time.monotonic()

        # ── Apply patch ──────────────────────────────────────────────────
        previous_code = state.current_code

        if action.patch_format == PatchFormat.UNIFIED_DIFF:
            new_code = apply_unified_diff(previous_code, action.patched_code)
        else:
            new_code = action.patched_code

        # Compute diff stats
        diff_stats = compute_diff_stats(previous_code, new_code)
        diff_text = compute_unified_diff(
            previous_code, new_code, state.task.canonical_filename
        )

        # ── Execute tests ────────────────────────────────────────────────
        exec_result = execute_submission(
            source_code=new_code,
            test_code=state.task.test_code,
            source_filename=state.task.canonical_filename,
            test_filename=state.task.test_filename,
            timeout_s=self.execution_timeout_s,
            sandbox_config=self.sandbox_config,
        )

        # ── Compute reward ───────────────────────────────────────────────
        reward_breakdown = compute_reward(
            action=action,
            execution_result=exec_result,
            state=state,
            diff_stats=diff_stats,
            config=self.reward_config,
        )

        # ── Update state ─────────────────────────────────────────────────
        code_hash = hash_code(new_code)
        is_solved = (
            exec_result.status == ExecutionStatus.SUCCESS
            and exec_result.passed == exec_result.total_tests
            and exec_result.total_tests > 0
        )

        # Track consecutive invalid actions
        is_invalid = (
            not exec_result.syntax_valid or diff_stats.is_noop
        )
        if is_invalid:
            state.consecutive_invalid += 1
        else:
            state.consecutive_invalid = 0

        # Update state
        state.current_code = new_code
        state.best_passed = max(state.best_passed, exec_result.passed)
        state.cumulative_reward += reward_breakdown.total
        state.seen_patches.append(code_hash)

        # Record step
        step_record = StepRecord(
            step_index=state.step_index,
            action=action,
            execution_result=exec_result,
            reward=reward_breakdown.total,
            reward_breakdown=reward_breakdown.to_dict(),
            code_snapshot=new_code,
            timestamp_ms=int(time.time() * 1000),
        )
        state.history.append(step_record)

        # Update metrics
        state.metrics.total_steps += 1
        state.metrics.total_reward = state.cumulative_reward
        if exec_result.total_tests > 0:
            pass_rate = exec_result.passed / exec_result.total_tests
            state.metrics.peak_pass_rate = max(state.metrics.peak_pass_rate, pass_rate)
        if not exec_result.syntax_valid:
            state.metrics.syntax_error_count += 1
        if reward_breakdown.duplicate_penalty < 0:
            state.metrics.duplicate_patch_count += 1
        if len(state.history) >= 2:
            prev_passed = state.history[-2].execution_result.passed
            if exec_result.passed < prev_passed:
                state.metrics.regression_count += 1

        # ── Check termination ────────────────────────────────────────────
        done = False
        done_reason = None

        if is_solved:
            done = True
            done_reason = DoneReason.SOLVED
            state.solved = True
            state.metrics.solved = True
            state.metrics.solve_step = state.step_index
        elif state.step_index >= state.max_steps - 1:
            done = True
            done_reason = DoneReason.MAX_STEPS
        elif state.consecutive_invalid >= state.max_consecutive_invalid:
            done = True
            done_reason = DoneReason.REPEATED_INVALID

        state.done = done
        state.done_reason = done_reason
        state.step_index += 1

        # ── Telemetry ────────────────────────────────────────────────────
        step_duration = time.monotonic() - step_start
        self.telemetry.log_step(
            StepMetric(
                episode_id=self._episode_id,
                task_id=state.task.task_id,
                step_index=state.step_index - 1,
                reward=reward_breakdown.total,
                cumulative_reward=state.cumulative_reward,
                passed_tests=exec_result.passed,
                total_tests=exec_result.total_tests,
                syntax_valid=exec_result.syntax_valid,
                execution_status=exec_result.status.value,
                diff_lines=diff_stats.total_diff_lines,
                duration_ms=step_duration * 1000,
                done=done,
                done_reason=done_reason.value if done_reason else None,
                reward_components=reward_breakdown.to_dict(),
            )
        )

        if done:
            state.metrics.total_runtime_s = time.time() - self._episode_start
            self.telemetry.log_episode(
                EpisodeSummary(
                    episode_id=self._episode_id,
                    task_id=state.task.task_id,
                    difficulty=state.task.difficulty.value,
                    total_steps=state.metrics.total_steps,
                    total_reward=state.metrics.total_reward,
                    solved=state.metrics.solved,
                    solve_step=state.metrics.solve_step,
                    peak_pass_rate=state.metrics.peak_pass_rate,
                    regression_count=state.metrics.regression_count,
                    syntax_error_count=state.metrics.syntax_error_count,
                    total_duration_s=state.metrics.total_runtime_s,
                )
            )

        # ── Build observation ────────────────────────────────────────────
        try:
            observation = self._build_observation(
                execution_result_dict=self._execution_to_obs(exec_result),
                reward_breakdown=reward_breakdown,
                previous_code=previous_code,
                diff_text=diff_text,
            )

            # Robust info dict calculation
            try:
                prev_pass = (
                    state.history[-2].execution_result.passed
                    if len(state.history) >= 2
                    else state.baseline_passed
                )
                is_improvement = bool(exec_result.passed > prev_pass)
            except (IndexError, AttributeError):
                is_improvement = False

            total = exec_result.total_tests
            grader_score = (float(exec_result.passed) / float(total)) if total > 0 else 0.0

            info: dict[str, Any] = {
                "episode_id": self._episode_id,
                "step_duration_ms": round(step_duration * 1000, 2),
                "is_improvement": is_improvement,
                "grader_score": grader_score,
                "total_tests": total,
                "passed_tests": exec_result.passed,
            }

            return observation, float(reward_breakdown.total), done, info

        except Exception as e:
            logger.exception("Final processing failure in step()")
            # Fallback to a bare-minimum successful return if observation building fails
            # but usually, this will bubble up to the app.py 500 handler which is now handled by UI.
            raise RuntimeError(f"Step completion failed: {e}")

    # ─── Get State ────────────────────────────────────────────────────────

    def get_state(self) -> dict[str, Any]:
        """
        Return serializable state for the OpenEnv state() endpoint.
        """
        if self._state is None:
            return {"status": "not_initialized"}

        state = self._state
        return {
            "episode_id": self._episode_id,
            "task_id": state.task.task_id,
            "difficulty": state.task.difficulty.value,
            "step_index": state.step_index,
            "max_steps": state.max_steps,
            "baseline_passed": state.baseline_passed,
            "baseline_total": state.baseline_total,
            "best_passed": state.best_passed,
            "cumulative_reward": round(state.cumulative_reward, 4),
            "solved": state.solved,
            "done": state.done,
            "done_reason": state.done_reason.value if state.done_reason else None,
            "total_steps_taken": state.metrics.total_steps,
            "peak_pass_rate": round(state.metrics.peak_pass_rate, 4),
            "regression_count": state.metrics.regression_count,
            "syntax_error_count": state.metrics.syntax_error_count,
        }

    def get_episode_view(self) -> dict[str, Any]:
        """Return a richer session view tailored for the human-facing UI."""
        if self._state is None:
            return {
                "initialized": False,
                "summary": {
                    "status": "idle",
                    "current_score": 0.0,
                    "baseline_score": 0.0,
                    "best_score": 0.0,
                    "latest_reward": 0.0,
                    "cumulative_reward": 0.0,
                    "step_index": 0,
                    "max_steps": self.max_steps,
                    "progress_ratio": 0.0,
                },
                "history": [],
                "baseline": None,
                "latest_run": None,
                "latest_action": None,
                "code": {
                    "original": "",
                    "current": "",
                    "previous": "",
                },
                "task": None,
            }

        state = self._state
        baseline_exec = state.baseline_execution
        episode_metrics = {
            metric.step_index: metric
            for metric in self.telemetry.get_episode_steps(self._episode_id)
        }
        history: list[dict[str, Any]] = []
        previous_passed = state.baseline_passed
        previous_code = state.original_code

        for record in state.history:
            exec_result = record.execution_result
            metric = episode_metrics.get(record.step_index)
            total_tests = exec_result.total_tests or state.baseline_total
            score = self._normalize_score(exec_result.passed, total_tests)
            previous_score = self._normalize_score(previous_passed, total_tests or state.baseline_total)
            full_diff = compute_unified_diff(
                previous_code,
                record.code_snapshot,
                state.task.canonical_filename,
            )
            history.append(
                {
                    "step_index": record.step_index + 1,
                    "status": self._derive_run_status(
                        exec_result,
                        previous_score,
                        score,
                    ),
                    "score": score,
                    "score_delta": round(score - previous_score, 4),
                    "reward_delta": round(record.reward, 4),
                    "cumulative_reward": round(
                        sum(item.reward for item in state.history[: record.step_index + 1]),
                        4,
                    ),
                    "passed": exec_result.passed,
                    "failed": exec_result.failed,
                    "errored": exec_result.errored,
                    "skipped": exec_result.skipped,
                    "total": total_tests,
                    "duration_ms": round(metric.duration_ms, 2) if metric else 0.0,
                    "diff_lines": metric.diff_lines if metric else 0,
                    "patch_format": record.action.patch_format.value,
                    "commit_message": record.action.commit_message,
                    "reasoning": record.action.reasoning,
                    "expected_test_impact": record.action.expected_test_impact,
                    "declare_bug_type": record.action.declare_bug_type or [],
                    "reward_breakdown": record.reward_breakdown,
                    "execution_status": exec_result.status.value,
                    "syntax_valid": exec_result.syntax_valid,
                    "failures": self._serialize_failures(exec_result),
                    "stdout": exec_result.stdout,
                    "stderr": exec_result.stderr,
                    "error_detail": exec_result.error_detail,
                    "diff_preview": truncate_diff(full_diff, max_lines=18) if full_diff else "",
                }
            )
            previous_passed = exec_result.passed
            previous_code = record.code_snapshot

        latest_execution = state.history[-1].execution_result if state.history else baseline_exec
        latest_reward = state.history[-1].reward if state.history else 0.0
        current_passed = latest_execution.passed if latest_execution else state.baseline_passed
        current_total = (
            latest_execution.total_tests
            if latest_execution and latest_execution.total_tests > 0
            else state.baseline_total
        )
        current_score = self._normalize_score(current_passed, current_total)
        baseline_score = self._normalize_score(state.baseline_passed, state.baseline_total)
        best_score = self._normalize_score(state.best_passed, state.baseline_total)
        latest_action = state.history[-1].action if state.history else None
        previous_code_snapshot = (
            state.history[-2].code_snapshot
            if len(state.history) >= 2
            else state.original_code
        )

        return {
            "initialized": True,
            "episode_id": self._episode_id,
            "task": {
                "task_id": state.task.task_id,
                "difficulty": state.task.difficulty.value,
                "instruction": state.task.description,
                "tags": state.task.tags,
                "metadata": state.task.metadata,
                "test_filename": state.task.test_filename,
                "source_filename": state.task.canonical_filename,
            },
            "summary": {
                "status": (
                    state.done_reason.value
                    if state.done and state.done_reason is not None
                    else "active"
                ),
                "execution_status": latest_execution.status.value if latest_execution else "skipped",
                "current_score": current_score,
                "baseline_score": baseline_score,
                "best_score": best_score,
                "latest_reward": round(latest_reward, 4),
                "cumulative_reward": round(state.cumulative_reward, 4),
                "step_index": state.step_index,
                "max_steps": state.max_steps,
                "progress_ratio": round(
                    state.step_index / state.max_steps,
                    4,
                ) if state.max_steps else 0.0,
                "solved": state.solved,
                "done": state.done,
                "done_reason": state.done_reason.value if state.done_reason else None,
                "baseline_passed": state.baseline_passed,
                "baseline_total": state.baseline_total,
                "current_passed": current_passed,
                "current_total": current_total,
                "best_passed": state.best_passed,
                "peak_pass_rate": round(state.metrics.peak_pass_rate, 4),
                "regression_count": state.metrics.regression_count,
                "syntax_error_count": state.metrics.syntax_error_count,
                "duplicate_patch_count": state.metrics.duplicate_patch_count,
            },
            "baseline": self._build_baseline_view(state),
            "latest_run": history[-1] if history else None,
            "history": history,
            "latest_action": (
                {
                    "reasoning": latest_action.reasoning,
                    "commit_message": latest_action.commit_message,
                    "patch_format": latest_action.patch_format.value,
                    "expected_test_impact": latest_action.expected_test_impact,
                    "declare_bug_type": latest_action.declare_bug_type or [],
                }
                if latest_action
                else None
            ),
            "code": {
                "original": state.original_code,
                "current": state.current_code,
                "previous": previous_code_snapshot,
            },
            "hint": self._generate_hint(state),
        }

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _build_observation(
        self,
        execution_result_dict: dict[str, Any],
        reward_breakdown: RewardBreakdown,
        previous_code: str | None = None,
        diff_text: str | None = None,
    ) -> CodeDebugObservation:
        """Construct a CodeDebugObservation from current state."""
        state = self._state
        assert state is not None

        return CodeDebugObservation(
            task_id=state.task.task_id,
            instruction=state.task.description,
            current_code=state.current_code,
            previous_code=previous_code,
            test_summary=execution_result_dict.get("test_summary", {}),
            test_output=execution_result_dict.get("test_output", ""),
            failed_tests=execution_result_dict.get("failed_tests", []),
            passed_tests=execution_result_dict.get("passed_tests", []),
            syntax_valid=execution_result_dict.get("syntax_valid", True),
            execution_status=execution_result_dict.get("execution_status", "skipped"),
            step_index=state.step_index,
            max_steps=state.max_steps,
            reward_breakdown=reward_breakdown.to_dict(),
            cumulative_score=round(state.cumulative_reward, 4),
            done=state.done,
            done_reason=state.done_reason.value if state.done_reason else None,
            diff_from_previous=truncate_diff(diff_text) if diff_text else None,
            hint=self._generate_hint(state),
            metadata={
                "task_tags": state.task.tags,
                "difficulty": state.task.difficulty.value,
                "baseline_passed": state.baseline_passed,
                "baseline_total": state.baseline_total,
                "best_passed": state.best_passed,
                "episode_id": self._episode_id,
            },
        )

    @staticmethod
    def _execution_to_obs(exec_result) -> dict[str, Any]:
        """Convert ExecutionResult to observation-friendly dict."""
        return {
            "test_summary": {
                "total": exec_result.total_tests,
                "passed": exec_result.passed,
                "failed": exec_result.failed,
                "errored": exec_result.errored,
                "skipped": exec_result.skipped,
            },
            "test_output": exec_result.stdout[:2000] if exec_result.stdout else "",
            "failed_tests": [
                {
                    "name": t.name,
                    "error": t.error_message,
                    "trace": t.short_trace,
                }
                for t in exec_result.test_results
                if not t.passed
            ],
            "passed_tests": [
                t.name for t in exec_result.test_results if t.passed
            ],
            "syntax_valid": exec_result.syntax_valid,
            "execution_status": exec_result.status.value,
        }

    @staticmethod
    def _generate_hint(state: CodeDebugState) -> str | None:
        """
        Generate an optional lightweight hint based on progress.
        Hints are revealed progressively to keep difficulty appropriate.
        """
        if state.step_index < 2:
            return None  # No hints early on

        if state.step_index >= state.max_steps - 2 and not state.solved:
            # Hint near end of episode
            tags = state.task.tags
            if tags:
                return f"Consider focusing on: {', '.join(tags[:2])}"

        if state.metrics.syntax_error_count >= 2:
            return "Multiple syntax errors detected. Double-check indentation and brackets."

        if state.metrics.regression_count >= 2:
            return "Your recent changes are causing regressions. Try making smaller, targeted edits."

        return None

    def _build_baseline_view(self, state: CodeDebugState) -> dict[str, Any] | None:
        """Summarize the baseline run performed during reset()."""
        exec_result = state.baseline_execution
        if exec_result is None:
            return None

        return {
            "status": exec_result.status.value,
            "score": self._normalize_score(state.baseline_passed, state.baseline_total),
            "passed": state.baseline_passed,
            "failed": exec_result.failed,
            "errored": exec_result.errored,
            "skipped": exec_result.skipped,
            "total": state.baseline_total,
            "syntax_valid": exec_result.syntax_valid,
            "duration_ms": round(exec_result.duration_s * 1000, 2),
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
            "error_detail": exec_result.error_detail,
            "failures": self._serialize_failures(exec_result),
        }

    @staticmethod
    def _normalize_score(passed: int, total: int) -> float:
        """Normalize pass counts into a benchmark score in [0, 1]."""
        if total <= 0:
            return 0.0
        return round(float(passed) / float(total), 4)

    @staticmethod
    def _derive_run_status(
        exec_result,
        previous_score: float,
        current_score: float,
    ) -> str:
        """Derive a human-readable run status for product presentation."""
        if exec_result.status == ExecutionStatus.SYNTAX_ERROR:
            return "syntax_error"
        if exec_result.status == ExecutionStatus.TIMEOUT:
            return "timeout"
        if exec_result.status == ExecutionStatus.CRASH:
            return "crash"
        if exec_result.status == ExecutionStatus.RUNTIME_ERROR and exec_result.errored > 0:
            return "runtime_error"
        if current_score >= 1.0 and exec_result.total_tests > 0:
            return "solved"
        if current_score > previous_score:
            return "improved"
        if current_score < previous_score:
            return "regressed"
        if exec_result.status == ExecutionStatus.SUCCESS:
            return "steady"
        return exec_result.status.value

    @staticmethod
    def _serialize_failures(exec_result) -> list[dict[str, Any]]:
        """Convert execution failures into concise UI-friendly summaries."""
        import re

        failures: list[dict[str, Any]] = []
        severity = "assertion"
        if exec_result.status == ExecutionStatus.SYNTAX_ERROR:
            severity = "syntax"
        elif exec_result.status == ExecutionStatus.TIMEOUT:
            severity = "timeout"
        elif exec_result.status in (ExecutionStatus.RUNTIME_ERROR, ExecutionStatus.CRASH):
            severity = "runtime"

        for test in exec_result.test_results:
            if test.passed:
                continue

            trace_excerpt = test.short_trace or test.error_message or exec_result.error_detail or ""
            trace_lines = [line.strip() for line in trace_excerpt.splitlines() if line.strip()]
            assertion = test.error_message or (trace_lines[-1] if trace_lines else exec_result.error_detail)
            root_cause = next(
                (
                    line
                    for line in trace_lines
                    if ".py:" in line or line.startswith("E ") or "AssertionError" in line
                ),
                assertion,
            )
            line_match = re.search(r":(\d+):", root_cause or "")
            source_line = int(line_match.group(1)) if line_match else None
            failures.append(
                {
                    "test_name": test.name,
                    "severity": severity,
                    "title": test.name.split("::")[-1],
                    "assertion": assertion,
                    "root_cause": root_cause,
                    "source_line": source_line,
                    "trace_excerpt": trace_excerpt,
                }
            )

        if failures or not exec_result.error_detail:
            return failures

        return [
            {
                "test_name": "execution",
                "severity": severity,
                "title": exec_result.status.value.replace("_", " ").title(),
                "assertion": exec_result.error_detail,
                "root_cause": exec_result.error_detail,
                "source_line": None,
                "trace_excerpt": exec_result.stderr or exec_result.stdout,
            }
        ]
