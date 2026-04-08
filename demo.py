#!/usr/bin/env python3
"""
CodeDebug-RL Demo — Interactive demonstration of the environment.

Shows a complete episode lifecycle:
1. Reset with a sample task
2. Submit a bad patch (partial fix)
3. Submit a syntax error
4. Submit the correct fix
5. Print structured observations and reward breakdowns

This demo uses the local (in-process) client for simplicity.
For HTTP server usage, see the README.
"""

from __future__ import annotations


from codedebug_env.client import CodeDebugLocalClient
from codedebug_env.models import CodeDebugAction


def separator(title: str) -> None:
    print(f"\n{'━' * 70}")
    print(f"  {title}")
    print(f"{'━' * 70}\n")


def print_observation(obs, step_label: str) -> None:
    """Pretty-print an observation."""
    print(f"  📋 Task:       {obs.task_id}")
    print(f"  📝 Step:       {obs.step_index}/{obs.max_steps}")
    print(f"  ✅ Passed:     {obs.test_summary.get('passed', 0)}/{obs.test_summary.get('total', 0)}")
    print(f"  ❌ Failed:     {obs.test_summary.get('failed', 0)}")
    print(f"  🔧 Syntax OK:  {obs.syntax_valid}")
    print(f"  📊 Status:     {obs.execution_status}")
    print(f"  🎯 Cumulative: {obs.cumulative_score:.4f}")
    print(f"  🏁 Done:       {obs.done}", end="")
    if obs.done_reason:
        print(f"  ({obs.done_reason})")
    else:
        print()

    if obs.hint:
        print(f"  💡 Hint:       {obs.hint}")

    # Reward breakdown
    if obs.reward_breakdown:
        print("\n  Reward Breakdown:")
        for k, v in obs.reward_breakdown.items():
            if v != 0:
                sign = "+" if v > 0 else ""
                print(f"    {k:.<30} {sign}{v:.4f}")


def run_demo() -> None:
    """Run the interactive demo."""
    separator("🐛 CodeDebug-RL — Demo Episode")

    # Initialize local client
    client = CodeDebugLocalClient(max_steps=10)

    # ── Step 0: Reset ────────────────────────────────────────────────
    separator("RESET — Loading 'FizzBuzz' debugging task")
    obs = client.reset(task_id="builtin_001_fizzbuzz")
    print_observation(obs, "Initial State")
    print("\n  📄 Buggy Code (first 5 lines):")
    for line in obs.current_code.splitlines()[:5]:
        print(f"    | {line}")

    # ── Step 1: Partial Fix (wrong approach) ─────────────────────────
    separator("STEP 1 — Partial fix attempt (wrong approach)")
    partial_fix = obs.current_code.replace(
        'result.append(i)  # BUG: should be str(i)',
        'result.append(str(i))',
    )
    # Intentionally introduce a new bug
    partial_fix = partial_fix.replace(
        'result.append("FizzBuzz")',
        'result.append("fizzbuzz")',  # wrong case
    )

    action1 = CodeDebugAction(
        patched_code=partial_fix,
        reasoning="Fixed the type error where integers were appended instead of strings. Changed append(i) to append(str(i)).",
        declare_bug_type=["type-error"],
        commit_message="fix: convert non-fizzbuzz numbers to strings",
    )
    obs, reward, done, info = client.step(action1)
    print(f"  ⚡ Reward: {reward:+.4f}")
    print_observation(obs, "After Step 1")

    if obs.diff_from_previous:
        print("\n  Diff:")
        for line in obs.diff_from_previous.splitlines()[:10]:
            print(f"    {line}")

    # ── Step 2: Syntax Error ─────────────────────────────────────────
    separator("STEP 2 — Submitting code with syntax error")
    broken_code = "def fizzbuzz(n):\n    return [  # unclosed bracket\n"

    action2 = CodeDebugAction(
        patched_code=broken_code,
        reasoning="Attempting a complete rewrite",
    )
    obs, reward, done, info = client.step(action2)
    print(f"  ⚡ Reward: {reward:+.4f}  (expected negative — syntax error)")
    print_observation(obs, "After Step 2")

    # ── Step 3: Correct Fix ──────────────────────────────────────────
    separator("STEP 3 — Correct fix!")
    correct_code = '''\
def fizzbuzz(n: int) -> list[str]:
    """Return FizzBuzz sequence from 1 to n."""
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
'''
    action3 = CodeDebugAction(
        patched_code=correct_code,
        reasoning="Two bugs found: 1) append(i) should be append(str(i)) to ensure all elements are strings, 2) FizzBuzz casing was correct in original.",
        declare_bug_type=["type-error"],
        expected_test_impact="All 4 tests should now pass",
        commit_message="fix: return strings for all elements in fizzbuzz",
    )
    obs, reward, done, info = client.step(action3)
    print(f"  ⚡ Reward: {reward:+.4f}  (expected large positive — solved!)")
    print_observation(obs, "After Step 3")

    # ── Summary ──────────────────────────────────────────────────────
    separator("EPISODE SUMMARY")
    state = client.get_state()
    print(f"  Episode ID:      {state.get('episode_id', 'N/A')}")
    print(f"  Task:            {state.get('task_id')}")
    print(f"  Steps taken:     {state.get('total_steps_taken')}")
    print(f"  Solved:          {'✅ YES' if state.get('solved') else '❌ NO'}")
    print(f"  Final Reward:    {state.get('cumulative_reward', 0):.4f}")
    print(f"  Peak Pass Rate:  {state.get('peak_pass_rate', 0):.1%}")
    print(f"  Regressions:     {state.get('regression_count', 0)}")
    print(f"  Syntax Errors:   {state.get('syntax_error_count', 0)}")

    separator("Demo Complete 🎉")
    print("  The environment is ready for RL training loops!")
    print("  See README.md for GRPO integration and Docker deployment.\n")


if __name__ == "__main__":
    run_demo()
