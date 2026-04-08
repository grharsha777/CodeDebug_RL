"""
FastAPI Application — HTTP server for the CodeDebug-RL environment.

Exposes the environment via OpenEnv-compatible REST endpoints:
- POST /reset — Start a new episode
- POST /step — Submit an action
- GET  /state — Get current environment state
- GET  /health — Health check
- GET  /tasks — List available tasks
- GET  /metrics — Telemetry summary

Optionally serves a web interface when ENABLE_WEB_INTERFACE=true.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from codedebug_env.models import CodeDebugAction, CodeDebugObservation, RewardConfig
from codedebug_env.server.environment import CodeDebugEnvironment
from codedebug_env.server.task_loader import TaskLoader
from codedebug_env.server.telemetry import TelemetryCollector
from codedebug_env.server.sandbox import SandboxConfig

logger = logging.getLogger("codedebug.app")

# ─── Configuration ────────────────────────────────────────────────────────────

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
    """Initialize environment on startup."""
    global env

    loader = TaskLoader()
    loader.load_builtin()
    if TASK_DIR:
        loader.load_directory(TASK_DIR)

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


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "environment": "codedebug-rl", "version": "1.0.0"}


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest | None = None) -> ResetResponse:
    """
    Reset the environment to start a new debugging episode.
    """
    assert env is not None, "Environment not initialized"

    req = request or ResetRequest()
    try:
        observation = env.reset(
            task_id=req.task_id,
            difficulty=req.difficulty,
            seed=req.seed,
        )
        return ResetResponse(observation=observation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error during reset")
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest) -> StepResponse:
    """
    Submit an action (code patch) and receive the next observation.
    """
    assert env is not None, "Environment not initialized"

    try:
        observation, reward, done, info = env.step(request.action)
        return StepResponse(
            observation=observation,
            reward=reward,
            done=done,
            info=info,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error during step")
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state", response_model=StateResponse)
async def state() -> StateResponse:
    """
    Get the current environment state.
    """
    assert env is not None, "Environment not initialized"
    return StateResponse(state=env.get_state())


@app.get("/tasks")
async def list_tasks() -> dict[str, Any]:
    """List all available debugging tasks."""
    assert env is not None, "Environment not initialized"
    return {"tasks": env.task_loader.list_tasks(), "count": env.task_loader.task_count}


@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """Return aggregate telemetry metrics."""
    return {
        "aggregate": telemetry.get_aggregate_stats(),
        "recent_episodes": [
            {
                "episode_id": e.episode_id,
                "task_id": e.task_id,
                "solved": e.solved,
                "steps": e.total_steps,
                "reward": round(e.total_reward, 4),
            }
            for e in telemetry.get_recent_episodes(5)
        ],
    }


# ─── Web Interface ────────────────────────────────────────────────────────────

if ENABLE_WEB:

    @app.get("/", response_class=HTMLResponse)
    async def web_interface() -> str:
        """Interactive web interface for demo purposes."""
        return _WEB_UI_HTML


_WEB_UI_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CodeDebug-RL — Interactive Debugger</title>
<style>
  :root {
    --bg: #0f0f1a;
    --surface: #1a1a2e;
    --surface2: #16213e;
    --primary: #e94560;
    --accent: #0f3460;
    --green: #00d68f;
    --orange: #ff9f43;
    --text: #eaeaea;
    --muted: #8892b0;
    --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
    --mono: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: var(--font); min-height: 100vh; }

  .header {
    background: linear-gradient(135deg, var(--surface) 0%, var(--accent) 100%);
    padding: 1.5rem 2rem; border-bottom: 2px solid var(--primary);
    display: flex; align-items: center; gap: 1rem;
  }
  .header h1 { font-size: 1.5rem; }
  .header .badge {
    background: var(--primary); color: white; padding: 0.2rem 0.6rem;
    border-radius: 12px; font-size: 0.75rem; font-weight: 600;
  }

  .container { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 1rem; max-width: 1600px; margin: 0 auto; }

  .panel {
    background: var(--surface); border-radius: 12px; padding: 1rem;
    border: 1px solid rgba(255,255,255,0.06);
  }
  .panel h3 {
    color: var(--primary); margin-bottom: 0.8rem; font-size: 0.95rem;
    text-transform: uppercase; letter-spacing: 1px;
  }

  textarea, pre {
    width: 100%; background: var(--bg); color: var(--text); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 0.8rem; font-family: var(--mono); font-size: 0.85rem;
    resize: vertical;
  }
  textarea { min-height: 300px; }
  pre { max-height: 300px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }

  .controls { display: flex; gap: 0.5rem; margin-top: 0.8rem; flex-wrap: wrap; }
  .btn {
    padding: 0.6rem 1.2rem; border: none; border-radius: 8px; font-weight: 600;
    cursor: pointer; transition: all 0.2s; font-size: 0.85rem;
  }
  .btn-primary { background: var(--primary); color: white; }
  .btn-primary:hover { background: #c73651; transform: translateY(-1px); }
  .btn-secondary { background: var(--accent); color: white; }
  .btn-secondary:hover { background: #1a4a7a; }
  .btn-success { background: var(--green); color: #0f0f1a; }

  .reward-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;
    font-size: 0.8rem;
  }
  .reward-item {
    display: flex; justify-content: space-between; padding: 0.3rem 0.5rem;
    background: rgba(255,255,255,0.03); border-radius: 6px;
  }
  .reward-item .val { font-family: var(--mono); font-weight: 600; }
  .reward-item .val.pos { color: var(--green); }
  .reward-item .val.neg { color: var(--primary); }

  .status-bar {
    display: flex; gap: 1rem; padding: 0.8rem 2rem; background: var(--surface);
    border-top: 1px solid rgba(255,255,255,0.06); font-size: 0.85rem;
  }
  .status-bar .stat { display: flex; gap: 0.3rem; align-items: center; }
  .status-bar .stat .label { color: var(--muted); }

  .full-width { grid-column: 1 / -1; }

  #reasoning { min-height: 60px; margin-top: 0.5rem; }

  .test-badge {
    display: inline-block; padding: 0.15rem 0.4rem; border-radius: 4px;
    font-size: 0.75rem; margin: 0.15rem; font-family: var(--mono);
  }
  .test-pass { background: rgba(0,214,143,0.15); color: var(--green); }
  .test-fail { background: rgba(233,69,96,0.15); color: var(--primary); }
</style>
</head>
<body>
<div class="header">
  <h1>🐛 CodeDebug-RL</h1>
  <span class="badge">OpenEnv v1.0</span>
  <span class="badge" style="background:var(--accent)">Interactive Demo</span>
</div>

<div class="container">
  <div class="panel">
    <h3>📝 Buggy Code</h3>
    <textarea id="code" placeholder="Click 'Reset' to load a task..."></textarea>
    <textarea id="reasoning" placeholder="Reasoning (optional): Explain the bug you identified..."></textarea>
    <div class="controls">
      <button class="btn btn-secondary" onclick="resetEnv()">🔄 Reset</button>
      <button class="btn btn-primary" onclick="submitPatch()">🚀 Submit Patch</button>
      <select id="difficulty" class="btn btn-secondary" style="appearance:auto;">
        <option value="">Any Difficulty</option>
        <option value="easy">Easy</option>
        <option value="medium">Medium</option>
        <option value="hard">Hard</option>
      </select>
    </div>
  </div>

  <div class="panel">
    <h3>📊 Test Results</h3>
    <div id="testBadges" style="margin-bottom:0.8rem;"></div>
    <pre id="testOutput">No tests run yet. Click Reset to start.</pre>
  </div>

  <div class="panel">
    <h3>🎯 Reward Breakdown</h3>
    <div id="rewardGrid" class="reward-grid">
      <div class="reward-item"><span>Waiting for first step...</span></div>
    </div>
  </div>

  <div class="panel">
    <h3>📜 Task Info</h3>
    <pre id="taskInfo">Reset to load a debugging task.</pre>
  </div>

  <div class="panel full-width">
    <h3>🔀 Diff from Previous</h3>
    <pre id="diffView" style="max-height:200px;">No diff yet.</pre>
  </div>
</div>

<div class="status-bar" id="statusBar">
  <div class="stat"><span class="label">Step:</span> <span id="stepNum">-</span></div>
  <div class="stat"><span class="label">Cumulative:</span> <span id="cumScore">0.00</span></div>
  <div class="stat"><span class="label">Status:</span> <span id="envStatus">idle</span></div>
  <div class="stat"><span class="label">Task:</span> <span id="taskId">-</span></div>
</div>

<script>
const API = window.location.origin;

async function resetEnv() {
  const diff = document.getElementById('difficulty').value;
  const body = {};
  if (diff) body.difficulty = diff;

  try {
    const res = await fetch(`${API}/reset`, {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    const data = await res.json();
    if (res.ok && data.observation) {
      const obs = data.observation;
      document.getElementById('code').value = obs.current_code;
      updateUI(obs);
      document.getElementById('envStatus').textContent = 'active';
      document.getElementById('diffView').textContent = 'Episode started — submit your first patch.';
    } else {
      alert('Reset failed: ' + (data.detail || 'Unknown server error'));
    }
  } catch(e) { alert('Reset failed (network/parsing): ' + e.message); }
}

async function submitPatch() {
  const code = document.getElementById('code').value;
  const reasoning = document.getElementById('reasoning').value;
  const action = { patched_code: code, patch_format: 'full_replace' };
  if (reasoning.trim()) action.reasoning = reasoning.trim();

  try {
    const res = await fetch(`${API}/step`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ action })
    });
    const data = await res.json();
    if (res.ok && data.observation) {
      updateUI(data.observation);
      if (data.done) {
        document.getElementById('envStatus').textContent =
          data.observation.done_reason === 'solved' ? '✅ SOLVED!' : '❌ ' + data.observation.done_reason;
      }
    } else {
      alert('Step failed: ' + (data.detail || 'Unknown server error'));
    }
  } catch(e) { alert('Step failed (network/parsing): ' + e.message); }
}

function updateUI(obs) {
  document.getElementById('stepNum').textContent = `${obs.step_index}/${obs.max_steps}`;
  document.getElementById('cumScore').textContent = obs.cumulative_score.toFixed(4);
  document.getElementById('taskId').textContent = obs.task_id;

  // Test badges
  const badges = document.getElementById('testBadges');
  let html = '';
  (obs.passed_tests||[]).forEach(t => { html += `<span class="test-badge test-pass">✓ ${t.split('::').pop()}</span>`; });
  (obs.failed_tests||[]).forEach(t => { html += `<span class="test-badge test-fail">✗ ${(t.name||'').split('::').pop()}</span>`; });
  badges.innerHTML = html;

  document.getElementById('testOutput').textContent = obs.test_output || 'No output';

  // Reward
  const grid = document.getElementById('rewardGrid');
  const rb = obs.reward_breakdown || {};
  grid.innerHTML = Object.entries(rb).map(([k,v]) =>
    `<div class="reward-item"><span>${k}</span><span class="val ${v>0?'pos':v<0?'neg':''}">${v>0?'+':''}${v.toFixed(4)}</span></div>`
  ).join('');

  // Task info
  document.getElementById('taskInfo').textContent = JSON.stringify({
    task_id: obs.task_id,
    instruction: obs.instruction,
    syntax_valid: obs.syntax_valid,
    execution_status: obs.execution_status,
    hint: obs.hint,
    metadata: obs.metadata,
  }, null, 2);

  // Diff
  if (obs.diff_from_previous) {
    document.getElementById('diffView').textContent = obs.diff_from_previous;
  }
}
</script>
</body>
</html>
"""
