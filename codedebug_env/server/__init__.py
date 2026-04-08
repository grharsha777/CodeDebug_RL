"""
CodeDebug-RL Server
===================

Server-side environment implementation including:
- CodeDebugEnvironment: core reset/step/state logic
- Executor: safe code execution and test parsing
- Reward: multi-dimensional reward computation
- TaskLoader: dataset loading and sampling
- Sandbox: isolated execution directories
- DiffUtils: human-readable patch diffs
- Telemetry: structured metrics and logging
"""
