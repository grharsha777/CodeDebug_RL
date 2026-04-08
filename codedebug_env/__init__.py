"""
CodeDebug-RL Environment
========================

An OpenEnv-compatible reinforcement learning environment for training
self-correcting coding agents through iterative bug repair.

The agent receives a buggy Python program + pytest suite, and must
iteratively patch the code until all tests pass — receiving shaped,
multi-dimensional rewards for partial progress, efficiency, and reasoning.

Designed for GRPO-style post-training loops and benchmark-grade evaluation.
"""

__version__ = "1.0.0"
__author__ = "CodeDebug-RL Team"

from codedebug_env.models import (
    CodeDebugAction,
    CodeDebugObservation,
    CodeDebugState,
    TaskSpec,
    RewardBreakdown,
    RewardConfig,
    TestResult,
    ExecutionResult,
    EpisodeMetrics,
)

__all__ = [
    "CodeDebugAction",
    "CodeDebugObservation",
    "CodeDebugState",
    "TaskSpec",
    "RewardBreakdown",
    "RewardConfig",
    "TestResult",
    "ExecutionResult",
    "EpisodeMetrics",
]
