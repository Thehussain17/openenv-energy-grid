"""
inference.py — Energy Grid Environment LLM Baseline Agent.

Runs all 3 tasks using an LLM (via OpenAI-compatible API) as the dispatch agent.
The LLM observes the grid state and outputs structured GridAction decisions.

Required environment variables:
    API_BASE_URL  — OpenAI-compatible API endpoint
    MODEL_NAME    — Model identifier (e.g. "Qwen/Qwen2.5-72B-Instruct")
    HF_TOKEN      — API key / HF token

Log format (stdout only, no deviation):
    [START] task=<task_name> episode=<n>
    [STEP] step=<n> action=<action> reward=<float> done=<bool>
    [END] task=<task_name> score=<float> steps=<n>
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
import subprocess
from typing import Any

from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── Server management ──────────────────────────────────────────────────────────
_SERVER_URL = "http://localhost:8000"
_server_proc: subprocess.Popen | None = None


def _start_server() -> None:
    """Start the uvicorn server in a background process."""
    global _server_proc
    _server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    # Wait for the server to be ready
    import urllib.request, urllib.error
    for _ in range(30):
        try:
            urllib.request.urlopen(f"{_SERVER_URL}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Server did not become ready in 30 s")


def _stop_server() -> None:
    global _server_proc
    if _server_proc is not None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        _server_proc = None


# ── LLM action selection ───────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert energy grid dispatch operator.
Given the current grid observation, output EXACTLY one JSON object (no markdown, no explanation):
{
  "decision": "<one of: battery_discharge | battery_charge | buy_external | curtail_load | idle>",
  "magnitude": <float between 0.0 and 1.0>
}

Decision guide:
- If demand >> supply and battery_soc > 0.2: use battery_discharge with magnitude 0.6–0.9
- If demand >> supply and battery_soc <= 0.2: use buy_external with magnitude 0.5–0.8
- If supply >> demand and battery_soc < 0.8: use battery_charge with magnitude 0.4–0.7
- If electricity_price is very high (>7) and you must buy: minimize magnitude
- If renewable_fraction is already high and supply > demand: use idle
- Avoid blackouts at all costs (supply must meet demand)
- Prefer renewables over external purchases
"""


def _llm_action(obs: dict) -> dict:
    """Ask the LLM for a dispatch action given the observation dict."""
    user_msg = (
        f"Grid state:\n"
        f"  Hour: {obs['time_of_day']:02d}:00\n"
        f"  Solar: {obs['solar_output']:.1f} kW | Wind: {obs['wind_output']:.1f} kW\n"
        f"  Demand: {obs['demand']:.1f} kW\n"
        f"  Battery SoC: {obs['battery_soc']:.2%}\n"
        f"  Grid frequency: {obs['grid_frequency']:.3f} Hz\n"
        f"  Electricity price: ₹{obs['electricity_price']:.1f}/kWh\n"
        f"  Renewable fraction: {obs['renewable_fraction']:.2%}\n"
        f"\nWhat is your dispatch decision?"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=64,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action = json.loads(raw)
        decision = str(action.get("decision", "idle"))
        magnitude = float(action.get("magnitude", 0.5))
        # Validate decision
        valid = {"battery_discharge", "battery_charge", "buy_external", "curtail_load", "idle"}
        if decision not in valid:
            decision = "idle"
        magnitude = max(0.0, min(1.0, magnitude))
        return {"decision": decision, "magnitude": magnitude}
    except Exception:
        # Fallback heuristic: if supply < demand → buy, else idle
        supply = obs.get("solar_output", 0) + obs.get("wind_output", 0)
        demand = obs.get("demand", 250)
        soc = obs.get("battery_soc", 0.5)
        if supply < demand * 0.9:
            if soc > 0.25:
                return {"decision": "battery_discharge", "magnitude": 0.7}
            return {"decision": "buy_external", "magnitude": 0.6}
        elif supply > demand * 1.1 and soc < 0.8:
            return {"decision": "battery_charge", "magnitude": 0.5}
        return {"decision": "idle", "magnitude": 0.0}


# ── Client wrapper ─────────────────────────────────────────────────────────────

def _make_env():
    """Return a sync env client connected to the local server."""
    try:
        from client import EnergyGridEnv
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from client import EnergyGridEnv
    return EnergyGridEnv(base_url=_SERVER_URL).sync()


# ══════════════════════════════════════════════════════════════════════════════
# Task 1 — Cost Minimization (8-hour episode, stable demand)
# ══════════════════════════════════════════════════════════════════════════════
_TASK1_MAX_HOURS = 8
# Threshold cost: if agent bought 50% of max external capacity every hour
_TASK1_THRESHOLD_COST_RS = 0.0  # computed dynamically after reset


def _task1_grader(actual_cost: float, threshold_cost: float) -> float:
    """score = max(0, 1 - actual_cost / threshold_cost); clamped [0, 1]."""
    if threshold_cost <= 0:
        return 1.0
    score = max(0.0, 1.0 - actual_cost / threshold_cost)
    return round(min(1.0, score), 4)


def run_task1(episode: int = 1) -> tuple[float, int]:
    """
    Task 1: Cost Minimization over 8 hours with stable demand.
    Returns (score, steps_taken).
    """
    print(f"[START] task=cost_minimization episode={episode}")

    env = _make_env()
    with env:
        result = env.reset(seed=42)
        obs = result.observation

        # Threshold cost: buying 400 kW × 50% each hour at TOU price
        from server.energy_grid_environment import _TOU_PRICE, MAX_BUY_KW
        threshold_cost = sum(
            _TOU_PRICE[h] * (MAX_BUY_KW * 0.5) * 1e-3
            for h in range(_TASK1_MAX_HOURS)
        )

        actual_cost = 0.0
        steps = 0

        for step in range(_TASK1_MAX_HOURS):
            obs_dict = obs.model_dump()
            action_dict = _llm_action(obs_dict)
            from models import GridAction
            action = GridAction(**action_dict)

            result = env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            # Track cost from metadata
            meta = obs.metadata or {}
            actual_cost = float(meta.get("total_cost", actual_cost))
            steps += 1

            print(
                f"[STEP] step={step+1} action={action.decision}({action.magnitude:.2f})"
                f" reward={reward:.4f} done={done}"
            )
            if done:
                break

        # Override total_cost from state
        state = env.state()
        actual_cost = state.total_cost
        score = _task1_grader(actual_cost, threshold_cost)
        print(f"[END] task=cost_minimization score={score:.4f} steps={steps}")
        return score, steps


# ══════════════════════════════════════════════════════════════════════════════
# Task 2 — Blackout Prevention (24-hour, 2 demand spikes)
# ══════════════════════════════════════════════════════════════════════════════
_TASK2_SPIKE_HOURS = [11, 19]


def _task2_grader(blackout_steps: int, blackout_events: int, total_steps: int) -> float:
    """score = 1.0 - (blackout_steps / total_steps) - 0.2 × events; clamped [0, 1]."""
    if total_steps == 0:
        return 0.0
    raw = 1.0 - (blackout_steps / total_steps) - 0.2 * blackout_events
    return round(max(0.0, min(1.0, raw)), 4)


def run_task2(episode: int = 1) -> tuple[float, int]:
    """
    Task 2: Blackout Prevention over 24 hours with 2 amplified demand spikes.
    Returns (score, steps_taken).
    """
    print(f"[START] task=blackout_prevention episode={episode}")

    env = _make_env()
    with env:
        result = env.reset(
            seed=123,
            demand_spike_hours=_TASK2_SPIKE_HOURS,
        )
        obs = result.observation

        prev_blackout_count = 0
        blackout_events = 0
        steps = 0

        for step in range(24):
            obs_dict = obs.model_dump()
            action_dict = _llm_action(obs_dict)
            from models import GridAction
            action = GridAction(**action_dict)

            result = env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            meta = obs.metadata or {}
            current_bc = int(meta.get("blackout_count", prev_blackout_count))
            if current_bc > prev_blackout_count:
                blackout_events += current_bc - prev_blackout_count
            prev_blackout_count = current_bc
            steps += 1

            print(
                f"[STEP] step={step+1} action={action.decision}({action.magnitude:.2f})"
                f" reward={reward:.4f} done={done}"
            )
            if done:
                break

        state = env.state()
        score = _task2_grader(state.blackout_count, blackout_events, steps)
        print(f"[END] task=blackout_prevention score={score:.4f} steps={steps}")
        return score, steps


# ══════════════════════════════════════════════════════════════════════════════
# Task 3 — Renewable Maximization (24-hour, full stochastic)
# ══════════════════════════════════════════════════════════════════════════════

def _task3_grader(
    renewable_fraction: float,
    stability_score: float,
    cost_score: float,
) -> float:
    """
    composite = 0.4 × renewable_fraction + 0.4 × stability_score + 0.2 × cost_score
    All inputs must be in [0, 1]. Output clamped to [0, 1].
    """
    composite = (
        0.4 * renewable_fraction
        + 0.4 * stability_score
        + 0.2 * cost_score
    )
    return round(max(0.0, min(1.0, composite)), 4)


def run_task3(episode: int = 1) -> tuple[float, int]:
    """
    Task 3: Renewable Maximization over 24 hours under stochastic conditions.
    Returns (score, steps_taken).
    """
    print(f"[START] task=renewable_maximization episode={episode}")

    env = _make_env()
    with env:
        result = env.reset(seed=777)
        obs = result.observation

        total_renewable_fraction = 0.0
        steps = 0

        for step in range(24):
            obs_dict = obs.model_dump()
            action_dict = _llm_action(obs_dict)
            from models import GridAction
            action = GridAction(**action_dict)

            result = env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            total_renewable_fraction += obs.renewable_fraction
            steps += 1

            print(
                f"[STEP] step={step+1} action={action.decision}({action.magnitude:.2f})"
                f" reward={reward:.4f} done={done}"
            )
            if done:
                break

        state = env.state()

        # Aggregate metrics
        avg_renewable_fraction = total_renewable_fraction / max(steps, 1)
        stability_score = max(
            0.0,
            1.0 - state.frequency_violation_steps / max(steps, 1),
        )

        # Budget cost: what a "fair" agent would spend (30% of max buy × TOU avg)
        from server.energy_grid_environment import _TOU_PRICE, MAX_BUY_KW
        budget_cost = sum(
            _TOU_PRICE[h] * (MAX_BUY_KW * 0.3) * 1e-3
            for h in range(24)
        )
        cost_score = max(0.0, 1.0 - state.total_cost / max(budget_cost, 1e-9))
        cost_score = min(1.0, cost_score)

        score = _task3_grader(
            renewable_fraction=avg_renewable_fraction,
            stability_score=stability_score,
            cost_score=cost_score,
        )
        print(f"[END] task=renewable_maximization score={score:.4f} steps={steps}")
        return score, steps


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()

    # Start the environment server
    _start_server()

    try:
        scores: dict[str, float] = {}

        s1, st1 = run_task1(episode=1)
        scores["cost_minimization"] = s1

        s2, st2 = run_task2(episode=1)
        scores["blackout_prevention"] = s2

        s3, st3 = run_task3(episode=1)
        scores["renewable_maximization"] = s3

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  Energy Grid Baseline Results  ({elapsed:.0f}s)")
        print(f"{'='*60}")
        for task, score in scores.items():
            status = "✓ PASS" if score >= 0.7 else "✗ FAIL"
            print(f"  {task:<28}  score={score:.4f}  {status}")
        print(f"{'='*60}")

    finally:
        _stop_server()


if __name__ == "__main__":
    main()
