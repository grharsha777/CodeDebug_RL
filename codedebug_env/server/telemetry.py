"""
Telemetry — Structured logging and metrics for observability.

Provides a clean logging interface for episode lifecycle events,
step-level metrics, and environment health. Designed for easy
integration with external monitoring (Prometheus, W&B, etc.).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger("codedebug.telemetry")


@dataclass
class StepMetric:
    """Metrics for a single environment step."""
    episode_id: str
    task_id: str
    step_index: int
    reward: float
    cumulative_reward: float
    passed_tests: int
    total_tests: int
    syntax_valid: bool
    execution_status: str
    diff_lines: int
    duration_ms: float
    done: bool
    done_reason: str | None = None
    reward_components: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=None, default=str)


@dataclass
class EpisodeSummary:
    """Summary metrics for a completed episode."""
    episode_id: str
    task_id: str
    difficulty: str
    total_steps: int
    total_reward: float
    solved: bool
    solve_step: int | None
    peak_pass_rate: float
    regression_count: int
    syntax_error_count: int
    total_duration_s: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=None, default=str)


class TelemetryCollector:
    """
    Collects and emits structured telemetry events.

    In production, this would push to Prometheus/Grafana/W&B.
    For hackathon scope, it logs structured JSON and buffers
    episode metrics for retrieval.
    """

    def __init__(self, enable_structured_logging: bool = True) -> None:
        self.enable_structured = enable_structured_logging
        self._episode_summaries: list[EpisodeSummary] = []
        self._step_metrics: list[StepMetric] = []

    def log_step(self, metric: StepMetric) -> None:
        """Record a step metric."""
        self._step_metrics.append(metric)
        if self.enable_structured:
            logger.info(
                "STEP_METRIC | episode=%s task=%s step=%d reward=%.4f "
                "passed=%d/%d status=%s done=%s",
                metric.episode_id,
                metric.task_id,
                metric.step_index,
                metric.reward,
                metric.passed_tests,
                metric.total_tests,
                metric.execution_status,
                metric.done,
            )

    def log_episode(self, summary: EpisodeSummary) -> None:
        """Record an episode summary."""
        self._episode_summaries.append(summary)
        if self.enable_structured:
            logger.info(
                "EPISODE_SUMMARY | id=%s task=%s solved=%s steps=%d "
                "reward=%.4f peak_pass=%.2f",
                summary.episode_id,
                summary.task_id,
                summary.solved,
                summary.total_steps,
                summary.total_reward,
                summary.peak_pass_rate,
            )

    def log_event(self, event: str, **kwargs: Any) -> None:
        """Log a generic structured event."""
        if self.enable_structured:
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            logger.info("EVENT | %s %s", event, extra)

    def get_recent_episodes(self, n: int = 10) -> list[EpisodeSummary]:
        """Return the N most recent episode summaries."""
        return self._episode_summaries[-n:]

    def get_aggregate_stats(self) -> dict[str, Any]:
        """Compute aggregate statistics across all recorded episodes."""
        if not self._episode_summaries:
            return {"total_episodes": 0}

        episodes = self._episode_summaries
        solved = [e for e in episodes if e.solved]
        return {
            "total_episodes": len(episodes),
            "solve_rate": len(solved) / len(episodes) if episodes else 0.0,
            "avg_reward": sum(e.total_reward for e in episodes) / len(episodes),
            "avg_steps": sum(e.total_steps for e in episodes) / len(episodes),
            "avg_solve_step": (
                sum(e.solve_step for e in solved if e.solve_step is not None)
                / len(solved)
                if solved
                else None
            ),
        }
