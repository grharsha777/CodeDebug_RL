<div align="center">

# 🐛 CodeDebug-RL Environment

### An OpenEnv-compatible reinforcement learning environment for training self-correcting coding agents through iterative bug repair

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-orange.svg)](#)
[![GRPO Ready](https://img.shields.io/badge/GRPO-ready-purple.svg)](#)

**Code debugging is the ultimate test of self-correction.** Unlike code generation, debugging requires an agent to *read*, *diagnose*, *reason about failure modes*, and *iteratively repair* — exactly the capabilities that matter most for post-training with RL.

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Reward System](#-reward-system) · [Demo](#-demo) · [Docker](#-docker-deployment) · [Training Integration](#-training-integration)

</div>

---

## 🎯 Why CodeDebug-RL?

Most coding environments give binary pass/fail rewards. This makes RL training extremely sparse. **CodeDebug-RL is different:**

| Feature | Simple Env | CodeDebug-RL |
|---------|-----------|-------------|
| Reward signal | Binary pass/fail | **7-component shaped reward** |
| Feedback | "Wrong" | Structured test traces, diff, hints |
| Iteration | Single-shot | **Multi-step episodes (up to 10 steps)** |
| Observability | Opaque | Full reward breakdown, telemetry |
| Bug types | Random | Curated by difficulty & category |
| Training compat. | Custom glue | **OpenEnv + GRPO/TRL native** |
| Deployment | Scripts | **Docker + HF Spaces ready** |

### Why Debugging > Generation for RL

1. **Dense learning signal** — Each test provides granular feedback on specific behaviors
2. **Natural curriculum** — Easy bugs (typos) → Hard bugs (logic errors) → Multi-bug repair
3. **Grounded reasoning** — Test failures ground the agent's reasoning in observable facts
4. **Self-correction loop** — The core post-training capability Meta, DeepSeek, and OpenAI are investing in
5. **Measurable progress** — Test pass rates give clean, interpretable metrics

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CodeDebug-RL                              │
│                                                                  │
│  ┌──────────┐    ┌─────────────────────────────────────────┐    │
│  │  Client   │    │              Server                     │    │
│  │          │    │                                         │    │
│  │ HTTP or  │───▶│  ┌───────────────────────────────────┐  │    │
│  │ Local    │    │  │      CodeDebugEnvironment          │  │    │
│  │          │◀───│  │                                   │  │    │
│  │ reset()  │    │  │  reset() ──▶ load task            │  │    │
│  │ step()   │    │  │             run baseline tests     │  │    │
│  │ state()  │    │  │             return observation      │  │    │
│  │          │    │  │                                   │  │    │
│  └──────────┘    │  │  step()  ──▶ validate action       │  │    │
│                  │  │             apply patch             │  │    │
│                  │  │             ┌──────────┐           │  │    │
│                  │  │             │ Sandbox  │           │  │    │
│                  │  │             │ executor │           │  │    │
│                  │  │             └──────────┘           │  │    │
│                  │  │             parse results           │  │    │
│                  │  │             ┌──────────┐           │  │    │
│                  │  │             │ Reward   │           │  │    │
│                  │  │             │ Engine   │           │  │    │
│                  │  │             └──────────┘           │  │    │
│                  │  │             return (obs, r, done)   │  │    │
│                  │  └───────────────────────────────────┘  │    │
│                  │                                         │    │
│                  │  Task Loader ─── Telemetry ─── Diff     │    │
│                  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
codedebug-rl/
├── codedebug_env/
│   ├── __init__.py              # Package exports
│   ├── models.py                # Pydantic models: Action, Observation, State, Reward
│   ├── client.py                # HTTP + Local client implementations
│   └── server/
│       ├── __init__.py
│       ├── environment.py       # Core RL environment (reset/step/state)
│       ├── app.py               # FastAPI server + web interface
│       ├── executor.py          # Safe pytest execution engine
│       ├── reward.py            # 7-component reward system
│       ├── task_loader.py       # Task dataset loading + sampling
│       ├── diff_utils.py        # Diff generation + patch metrics
│       ├── sandbox.py           # Isolated execution directories
│       ├── telemetry.py         # Structured logging + metrics
│       └── Dockerfile           # Production container
├── configs/
│   ├── default.yaml             # Environment defaults
│   └── rewards.yaml             # Reward tuning knobs
├── data/tasks/                  # Extensible task dataset
│   ├── easy/                    # Tier 1: type errors, simple fixes
│   ├── medium/                  # Tier 2: logic bugs, off-by-one
│   └── hard/                    # Tier 3: multi-bug, algorithmic
├── examples/                    # Standalone example files
├── tests/                       # Comprehensive test suite
│   ├── test_models.py
│   ├── test_environment_logic.py
│   ├── test_reward_logic.py
│   └── test_client_server_contract.py
├── scripts/
│   ├── run_local_demo.sh
│   ├── build_docker.sh
│   └── validate_submission.sh
├── demo.py                      # Interactive demo script
├── openenv.yaml                 # OpenEnv specification
├── pyproject.toml               # Project metadata + deps
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/codedebug-rl/codedebug-rl.git
cd codedebug-rl

# Install with pip (Python 3.11+)
pip install -e ".[dev]"
```

### Run the Demo

```bash
python demo.py
```

This runs a complete 3-step episode showing:
1. A partial fix (some tests pass)
2. A syntax error (negative reward)
3. The correct fix (solve bonus!)

### Start the Server

```bash
# API only
uvicorn codedebug_env.server.app:app --reload

# With interactive web interface
ENABLE_WEB_INTERFACE=true uvicorn codedebug_env.server.app:app --reload
```

Then open `http://localhost:8000` for the interactive debugger UI.

---

## 🎮 Action & Observation Schema

### Action (Agent → Environment)

```python
class CodeDebugAction:
    patched_code: str              # Full replacement or unified diff
    reasoning: str | None          # Bug analysis (earns reward bonus!)
    patch_format: "full_replace" | "unified_diff"
    declare_bug_type: list[str]    # e.g. ["off-by-one", "logic"]
    expected_test_impact: str      # Agent's prediction
    commit_message: str            # Human-readable summary
```

### Observation (Environment → Agent)

```python
class CodeDebugObservation:
    task_id: str                   # Current task identifier
    instruction: str               # Bug report / task description
    current_code: str              # Latest code version
    previous_code: str | None      # For diff comparison
    test_summary: dict             # {total, passed, failed, errored}
    test_output: str               # Raw pytest output
    failed_tests: list[dict]       # Structured failure details
    passed_tests: list[str]        # Names of passing tests
    syntax_valid: bool             # Did the code parse?
    execution_status: str          # success|syntax_error|runtime_error|timeout
    step_index: int                # Current step (0-indexed)
    max_steps: int                 # Episode budget
    reward_breakdown: dict         # Transparent reward components
    cumulative_score: float        # Running total
    done: bool                     # Episode terminated?
    done_reason: str | None        # solved|max_steps|repeated_invalid
    diff_from_previous: str | None # Human-readable diff
    hint: str | None               # Progressive hints
    metadata: dict                 # Tags, difficulty, etc.
```

---

## 🎯 Reward System

The reward system is CodeDebug-RL's **core differentiator** — a 7-component shaped reward that provides rich gradient signal for RL training.

### Component Breakdown

| # | Component | Weight | Signal |
|---|-----------|--------|--------|
| 1 | **Partial Test Credit** | `0.4` | Proportional improvement in test pass rate over baseline |
| 2 | **Full Solve Bonus** | `1.0` | Large sparse reward when all tests pass |
| 3 | **Regression Penalty** | `-0.3` | Per test that was passing but now fails |
| 4 | **Syntax Error Penalty** | `-0.5` | Unparseable code submission |
| 5 | **Runtime/Timeout Penalty** | `-0.2/-0.4` | Crash or hanging code |
| 6 | **Patch Efficiency** | `0.1` | Minimal targeted fixes > large rewrites |
| 7 | **Reasoning Quality** | `0.05` | Structured bug analysis bonus |

### Shaping Terms

| Term | Weight | Purpose |
|------|--------|---------|
| Duplicate penalty | `-0.3` | Penalize re-submitting identical patches |
| No-op penalty | `-0.2` | Penalize unchanged code |
| Improvement streak | `+0.05/step` | Reward monotonic progress |
| Early solve bonus | `+0.02/remaining step` | Efficiency incentive |
| Invalid action | `-0.4` | Malformed submissions |

### Why This Matters for GRPO

```
Standard environment:    r = 1.0 if solved else 0.0
                         → 99% of samples get r=0 → no gradient

CodeDebug-RL:           r = Σ(7 shaped components + 5 shaping terms)
                         → Every step produces learning signal
                         → Partial progress is rewarded
                         → Regressions are penalized
                         → Efficient agents learn faster
```

All weights are configurable via `configs/rewards.yaml`.

---

## 🐳 Docker Deployment

### Build & Run

```bash
# Build the image
./scripts/build_docker.sh

# Run with web interface
docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true codedebug-rl:latest
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `CODEDEBUG_MAX_STEPS` | `10` | Max steps per episode |
| `CODEDEBUG_TIMEOUT` | `30` | Execution timeout (seconds) |
| `ENABLE_WEB_INTERFACE` | `false` | Enable interactive web UI |
| `CODEDEBUG_TASK_DIR` | `""` | Path to custom task directory |

### HuggingFace Spaces

```bash
# Deploy to HF Spaces
# 1. Create a new Space with Docker SDK
# 2. Push this repo
# 3. Ensure the space is tagged with "openenv"
# 4. Set environment variables in Space settings:
#    ENABLE_WEB_INTERFACE=true
# 5. The Dockerfile handles everything else on PORT 7860
```

---

## 🤖 Baseline Performance & LLM Inference

CodeDebug-RL provides a reproducible baseline inference script to evaluate agents. The script evaluates your LLM on a subset of built-in tasks and strictly follows the OpenEnv output format requirements.

### Running Inference

Run the provided `inference.py` using the OpenAI client spec:

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token_here"

python inference.py
```

### Expected Baseline Scores

Scores represent the `grader_score` normalized pass rate on the final step for a typical inference run with `gpt-4o-mini`:

| Task ID | Difficulty | Expected Score |
|---------|-----------|----------------|
| `builtin_001_fizzbuzz` | Easy | 1.0 (100%) |
| `builtin_002_binary_search` | Medium | 1.0 (100%) |
| `builtin_003_flatten_nested` | Hard | 0.8+ (80%+) |

---

## 🔄 Training Integration

### Direct Python (No Server)

```python
from codedebug_env.client import CodeDebugLocalClient
from codedebug_env.models import CodeDebugAction

env = CodeDebugLocalClient(max_steps=10)

for episode in range(100):
    obs = env.reset(difficulty="medium")

    done = False
    while not done:
        # Your LLM generates the action
        action = CodeDebugAction(
            patched_code=your_llm.generate(obs.current_code, obs.test_output),
            reasoning="...",
        )
        obs, reward, done, info = env.step(action)
        # reward is the shaped multi-component signal
```

### GRPO / TRL Integration

```python
from trl import GRPOTrainer
from codedebug_env.client import CodeDebugLocalClient

env = CodeDebugLocalClient(max_steps=5)

def reward_fn(completions, prompts):
    """Reward function for GRPO — uses CodeDebug-RL environment."""
    rewards = []
    for completion in completions:
        obs = env.reset(seed=hash(completion) % 10000)
        action = CodeDebugAction(patched_code=parse_code(completion))
        obs, reward, done, info = env.step(action)
        rewards.append(reward)  # Multi-dimensional shaped reward
    return rewards

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    ...
)
```

### HTTP Client (For Distributed Training)

```python
from codedebug_env.client import CodeDebugClient

# Connect to the server
client = CodeDebugClient("http://localhost:8000")

obs = client.reset(task_id="builtin_002_binary_search")
obs, reward, done, info = client.step(
    CodeDebugAction(patched_code="...", reasoning="...")
)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_reward_logic.py -v
pytest tests/test_environment_logic.py -v
pytest tests/test_client_server_contract.py -v

# With coverage
pytest tests/ --cov=codedebug_env --cov-report=term-missing
```

---

## 📊 Telemetry & Observability

The environment emits structured telemetry for every step and episode:

```
STEP_METRIC | episode=a3f2 task=builtin_001 step=0 reward=0.2800 passed=3/4 status=runtime_error done=False
STEP_METRIC | episode=a3f2 task=builtin_001 step=1 reward=1.1200 passed=4/4 status=success done=True
EPISODE_SUMMARY | id=a3f2 task=builtin_001 solved=True steps=2 reward=1.4000 peak_pass=1.00
```

Access aggregate metrics via the `/metrics` endpoint:
```json
{
  "aggregate": {
    "total_episodes": 47,
    "solve_rate": 0.72,
    "avg_reward": 0.83,
    "avg_steps": 3.2,
    "avg_solve_step": 2.1
  }
}
```

---

## 🗺️ Roadmap

- [ ] **v1.1** — Expand task dataset to 50+ curated debugging tasks
- [ ] **v1.2** — Curriculum learning scheduler (auto-difficulty progression)
- [ ] **v1.3** — Batched parallel evaluation for training throughput  
- [ ] **v1.4** — Multi-file debugging tasks
- [ ] **v1.5** — HuggingFace dataset integration (`load_dataset("codedebug-rl/tasks")`)
- [ ] **v2.0** — Docker-in-Docker sandboxing for full isolation
- [ ] **v2.1** — Leaderboard + benchmark suite for published models
- [ ] **v2.2** — Integration with veRL / OpenRLHF training pipelines

---

## 🏆 What Makes This Stand Out

1. **Not a toy** — Production-grade architecture with typed models, configurable rewards, and structured telemetry
2. **Reward design depth** — 7 components + 5 shaping terms, each independently tuned and transparent
3. **Training-native** — Direct GRPO/TRL compatibility with both local and HTTP interfaces
4. **Benchmark-ready** — Curated tasks by difficulty, extensible dataset format, deterministic seeding
5. **Full observability** — Every reward component is exposed in the observation for debugging and analysis
6. **Self-correction focus** — Multi-step episodes with history, regression tracking, and progressive hints
7. **Deploy anywhere** — Docker + HF Spaces + local, with one command

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the RL post-training era** 🚀

*Training agents to debug code is training them to think.*

</div>
