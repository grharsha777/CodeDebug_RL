"""
FastAPI application for the CodeDebug-RL environment.

The core OpenEnv endpoints remain stable while the optional web interface
serves a richer benchmark/workbench UI for demo and hackathon review.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from codedebug_env.models import CodeDebugAction, CodeDebugObservation, RewardConfig
from codedebug_env.server.environment import CodeDebugEnvironment
from codedebug_env.server.sandbox import SandboxConfig
from codedebug_env.server.task_loader import TaskLoader
from codedebug_env.server.telemetry import TelemetryCollector

logger = logging.getLogger("codedebug.app")

# ─── Configuration ────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[2]
SERVER_DIR = Path(__file__).resolve().parent
STATIC_DIR = SERVER_DIR / "static"
OPENENV_SPEC_PATH = ROOT_DIR / "openenv.yaml"
LIVE_BASELINE_PATH = ROOT_DIR / "baseline_results.json"
REFERENCE_BASELINE_PATH = ROOT_DIR / "configs" / "reference_baseline.json"
DEFAULT_TASK_DIR = ROOT_DIR / "data" / "tasks"

MAX_STEPS = int(os.environ.get("CODEDEBUG_MAX_STEPS", "10"))
EXECUTION_TIMEOUT = float(os.environ.get("CODEDEBUG_TIMEOUT", "30"))
ENABLE_WEB = os.environ.get("ENABLE_WEB_INTERFACE", "false").lower() == "true"
TASK_DIR = os.environ.get("CODEDEBUG_TASK_DIR", "")
PORT = int(os.environ.get("PORT", "8000"))

# ─── Globals ──────────────────────────────────────────────────────────────────

env: CodeDebugEnvironment | None = None
telemetry = TelemetryCollector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the environment on startup."""
    global env

    loader = TaskLoader()
    loader.load_builtin()

    candidate_dirs = [DEFAULT_TASK_DIR]
    if TASK_DIR:
        candidate_dirs.append(Path(TASK_DIR))

    seen_dirs: set[Path] = set()
    for candidate in candidate_dirs:
        resolved = candidate.resolve()
        if resolved in seen_dirs or not resolved.exists():
            continue
        loader.load_directory(resolved)
        seen_dirs.add(resolved)

    env = CodeDebugEnvironment(
        task_loader=loader,
        reward_config=RewardConfig(),
        max_steps=MAX_STEPS,
        execution_timeout_s=EXECUTION_TIMEOUT,
        sandbox_config=SandboxConfig(),
        telemetry=telemetry,
    )

    logger.info(
        "CodeDebug-RL environment initialized: %d tasks, max_steps=%d",
        loader.task_count,
        MAX_STEPS,
    )
    yield
    logger.info("CodeDebug-RL environment shutting down")


app = FastAPI(
    title="CodeDebug-RL Environment",
    description="OpenEnv-compatible RL environment for training self-correcting coding agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Request/Response Models ─────────────────────────────────────────────────


class ResetRequest(BaseModel):
    task_id: str | None = None
    difficulty: str | None = None
    seed: int | None = None


class ResetResponse(BaseModel):
    observation: CodeDebugObservation


class StepRequest(BaseModel):
    action: CodeDebugAction


class StepResponse(BaseModel):
    observation: CodeDebugObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    state: dict[str, Any]


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _get_env() -> CodeDebugEnvironment:
    assert env is not None, "Environment not initialized"
    return env


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_baseline_snapshot() -> dict[str, Any]:
    if LIVE_BASELINE_PATH.exists():
        payload = _load_json(LIVE_BASELINE_PATH)
        payload["source"] = "live"
        return payload
    if REFERENCE_BASELINE_PATH.exists():
        payload = _load_json(REFERENCE_BASELINE_PATH)
        payload["source"] = "reference"
        return payload
    return {
        "source": "missing",
        "model": None,
        "results": [],
        "average_score": None,
    }


def _build_benchmark_payload(environment: CodeDebugEnvironment) -> dict[str, Any]:
    catalog = environment.task_loader.get_task_catalog()
    baseline = _load_baseline_snapshot()
    baseline_map = {
        item["task_id"]: item for item in baseline.get("results", [])
        if isinstance(item, dict) and item.get("task_id")
    }
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}

    tasks = []
    distribution = {"easy": 0, "medium": 0, "hard": 0}
    for task in sorted(
        catalog,
        key=lambda item: (
            difficulty_order.get(item["difficulty"], 9),
            item["task_id"],
        ),
    ):
        distribution[task["difficulty"]] = distribution.get(task["difficulty"], 0) + 1
        baseline_row = baseline_map.get(task["task_id"], {})
        tasks.append(
            {
                "task_id": task["task_id"],
                "difficulty": task["difficulty"],
                "objective": task["description"],
                "tags": task["tags"],
                "baseline_score": baseline_row.get("score"),
                "baseline_success": baseline_row.get("success"),
                "baseline_steps": baseline_row.get("steps"),
                "grader_range": "0.0-1.0",
            }
        )

    return {
        "task_count": len(tasks),
        "difficulty_distribution": distribution,
        "tasks": tasks,
        "baseline": baseline,
        "constraints": {
            "max_inference_runtime_min": 20,
            "compute_budget": "2 vCPU / 8 GB RAM",
            "docker_required": True,
            "space_required": True,
            "openenv_required": True,
        },
    }


def _build_compliance_payload(environment: CodeDebugEnvironment) -> dict[str, Any]:
    spec = _load_yaml(OPENENV_SPEC_PATH)
    return {
        "environment_name": spec.get("name", "codedebug-rl"),
        "version": spec.get("version", "1.0.0"),
        "typed_models": {
            "action": "codedebug_env.models.CodeDebugAction",
            "observation": "codedebug_env.models.CodeDebugObservation",
            "reward_config": "codedebug_env.models.RewardConfig",
        },
        "api": {
            "reset": {"method": "POST", "path": "/reset"},
            "step": {"method": "POST", "path": "/step"},
            "state": {"method": "GET", "path": "/state"},
            "metrics": {"method": "GET", "path": "/metrics"},
        },
        "validator_status": {
            "openenv_yaml_present": OPENENV_SPEC_PATH.exists(),
            "dockerfile_present": (ROOT_DIR / "Dockerfile").exists(),
            "inference_script_present": (ROOT_DIR / "inference.py").exists(),
            "web_interface_enabled": (STATIC_DIR / "index.html").exists(),
            "hf_space_ready": (STATIC_DIR / "index.html").exists(),
            "task_count": environment.task_loader.task_count,
        },
        "required_env": {
            "API_BASE_URL": os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
            "MODEL_NAME": os.environ.get("MODEL_NAME", "gpt-4o-mini"),
            "HF_TOKEN": "configured" if os.environ.get("HF_TOKEN") else "required",
        },
        "openenv": spec.get("environment", {}),
        "docker": spec.get("docker", {}),
        "tasks": spec.get("tasks", {}),
        "reward": spec.get("reward", {}),
    }


def _recent_episode_payload(limit: int = 6) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": item.episode_id,
            "task_id": item.task_id,
            "difficulty": item.difficulty,
            "solved": item.solved,
            "steps": item.total_steps,
            "reward": round(item.total_reward, 4),
            "peak_pass_rate": round(item.peak_pass_rate, 4),
            "solve_step": item.solve_step,
            "duration_s": round(item.total_duration_s, 3),
        }
        for item in telemetry.get_recent_episodes(limit)
    ]


# ─── OpenEnv Endpoints ───────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "environment": "codedebug-rl", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest | None = None) -> ResetResponse:
    """Reset the environment to start a new debugging episode."""
    environment = _get_env()

    req = request or ResetRequest()
    try:
        observation = environment.reset(
            task_id=req.task_id,
            difficulty=req.difficulty,
            seed=req.seed,
        )
        return ResetResponse(observation=observation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error during reset")
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}")


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest) -> StepResponse:
    """Submit an action (code patch) and receive the next observation."""
    environment = _get_env()

    try:
        observation, reward, done, info = environment.step(request.action)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error during step")
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}")


@app.get("/state", response_model=StateResponse)
async def state() -> StateResponse:
    """Get the current environment state."""
    environment = _get_env()
    return StateResponse(state=environment.get_state())


@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    """List all available debugging tasks."""
    environment = _get_env()
    return {
        "tasks": environment.task_loader.list_tasks(),
        "count": environment.task_loader.task_count,
    }


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Return aggregate telemetry metrics."""
    return {
        "aggregate": telemetry.get_aggregate_stats(),
        "recent_episodes": _recent_episode_payload(limit=5),
    }


# ─── UI Endpoints ────────────────────────────────────────────────────────────


@app.get("/ui/session")
async def ui_session() -> dict[str, Any]:
    """Return the current episode view for the product UI."""
    environment = _get_env()
    return environment.get_episode_view()


@app.get("/ui/bootstrap")
async def ui_bootstrap() -> dict[str, Any]:
    """Return benchmark, compliance, and current session data for the UI."""
    environment = _get_env()
    return {
        "product": {
            "name": "CodeDebug-RL",
            "title": "Iterative Debugging Benchmark for Self-Correcting Code Agents",
            "subtitle": (
                "A real OpenEnv environment for code repair with isolated test execution, "
                "dense reward shaping, and stepwise episode telemetry."
            ),
            "proof_points": [
                "OpenEnv-compatible API",
                "3+ graded tasks",
                "HF Space + Docker ready",
                "Structured reward + telemetry",
            ],
        },
        "benchmark": _build_benchmark_payload(environment),
        "compliance": _build_compliance_payload(environment),
        "metrics": {
            "aggregate": telemetry.get_aggregate_stats(),
            "recent_episodes": _recent_episode_payload(limit=6),
        },
        "session": environment.get_episode_view(),
    }


# ─── Web Interface ────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def web_interface() -> FileResponse:
    """Serve the human-facing benchmark/workbench UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI bundle missing")
    return FileResponse(index_file)
