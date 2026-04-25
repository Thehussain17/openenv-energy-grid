"""
inference.py — Energy Grid Environment v2.1 LLM Baseline Agent & SFT Generator.

Runs all 3 tasks using a rule-based heuristic (with LLM opt-in via env vars).
Actions are now 4-dim MultiDiscrete: [bess, hospital, industrial, residential].

Log format (stdout, required by OpenEnv validator):
    [START] task=<name> episode=<n>
    [STEP]  step=<n> action=[b,h,i,r] reward=<float> done=<bool>
    [END]   task=<name> score=<float> steps=<n>
"""

from __future__ import annotations

import json
import os
import sys
import time
import subprocess
import random
from typing import Any

from openai import OpenAI

# ── Environment variables ──────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN:     str = os.environ.get("HF_TOKEN", "")

client = OpenAI(api_key=HF_TOKEN or "placeholder", base_url=API_BASE_URL)

# ── Server management ──────────────────────────────────────────────────────────
_SERVER_URL = "http://localhost:7860"
_server_proc: subprocess.Popen | None = None

def _start_server() -> None:
    global _server_proc
    _server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", "7860"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(f"{_SERVER_URL}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("Server did not become ready in 30s")

def _stop_server() -> None:
    global _server_proc
    if _server_proc is not None:
        _server_proc.terminate()
        try:
            _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_proc.kill()
        _server_proc = None

# ── Client helper ──────────────────────────────────────────────────────────────
def _make_env():
    try:
        from client import EnergyGridEnv
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from client import EnergyGridEnv
    return EnergyGridEnv(base_url=_SERVER_URL).sync()

def _action_from_dict(d: dict):
    try:
        from models import GridAction
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models import GridAction
    return GridAction(
        bess=int(d.get("bess", 0)),
        hospital=int(d.get("hospital", 0)),
        industrial=int(d.get("industrial", 0)),
        residential=int(d.get("residential", 0)),
    )

def _heuristic_action_with_reasoning(obs) -> tuple[dict, str]:
    soc         = obs.battery_soc
    solar       = obs.solar_norm
    wind        = obs.wind_norm
    freq_norm   = obs.frequency_norm
    solar_swan  = obs.solar_swan_active > 0.5
    wind_swan   = obs.wind_swan_active  > 0.5
    shed_over   = obs.cumulative_shed_ratio_norm > 1.0
    any_swan    = solar_swan or wind_swan

    fc_solar_1h = obs.forecast_solar_1h
    fc_wind_1h  = obs.forecast_wind_1h

    reasoning_parts = []
    
    renewable_available = solar + wind
    fc_renewable_1h = fc_solar_1h + fc_wind_1h

    tight = renewable_available < 0.4
    abundant = renewable_available > 0.7 and freq_norm > 0.55
    will_be_tight = fc_renewable_1h < 0.4

    if tight and soc > 0.25:
        bess = 4
        reasoning_parts.append("Renewables are low, discharging BESS 50% to support load.")
    elif freq_norm < 0.45 and soc > 0.15:
        bess = 5 if soc > 0.4 else 4
        reasoning_parts.append("Grid frequency dropping critically, discharging BESS to stabilize.")
    elif tight and soc <= 0.25:
        bess = 0
        reasoning_parts.append("Renewables are low but BESS SoC is critical, idling BESS.")
    elif abundant and soc < 0.85:
        bess = 2
        reasoning_parts.append("Renewables are abundant, charging BESS 50%.")
    elif not tight and will_be_tight and soc < 0.85:
        bess = 1
        reasoning_parts.append("Forecast predicts drop in renewables next hour, charging BESS 25% preemptively.")
    else:
        bess = 0
        reasoning_parts.append("Grid is stable, BESS is idle.")

    if any_swan and soc > 0.30:
        bess = 5
        reasoning_parts.append("Black swan detected! Discharging BESS fully.")

    hospital = 0
    reasoning_parts.append("Hospital must be protected, holding at 100%.")

    freq_bad = abs(freq_norm - 0.5) > 0.15
    if shed_over:
        industrial = 0
        reasoning_parts.append("Shed budget exceeded, not shedding industrial.")
    elif any_swan or freq_bad:
        industrial = 2
        reasoning_parts.append("Grid under stress, shedding industrial to 80%.")
    else:
        industrial = 0
        reasoning_parts.append("Industrial held at 100%.")

    if shed_over:
        residential = 0
        reasoning_parts.append("Shed budget exceeded, not shedding residential.")
    elif any_swan and freq_bad:
        residential = 3
        reasoning_parts.append("Severe grid stress, shedding residential to 60%.")
    elif any_swan or freq_bad:
        residential = 2
        reasoning_parts.append("Grid under stress, shedding residential to 70%.")
    else:
        residential = 0
        reasoning_parts.append("Residential held at 100%.")

    reasoning = " ".join(reasoning_parts)
    action = {"bess": bess, "hospital": hospital, "industrial": industrial, "residential": residential}
    return action, reasoning

def _heuristic_action(obs) -> dict:
    act, _ = _heuristic_action_with_reasoning(obs)
    return act

_SYSTEM_PROMPT = """You are an expert energy grid dispatch operator managing a 500kW microgrid.
Given the 22-dimensional normalized grid observation vector, output EXACTLY one JSON object (no markdown):
{
  "bess":        <int 0-5>,  // 0=Idle,1=Charge25%,2=Charge50%,3=Dis25%,4=Dis50%,5=Dis100%
  "hospital":    <int 0-1>,  // 0=100% supply, 1=98% (emergency only)
  "industrial":  <int 0-3>,  // 0=100%, 1=90%, 2=80%, 3=70%
  "residential": <int 0-3>   // 0=100%, 1=85%, 2=70%, 3=60%
}
Rules: Hospital MUST be served >=95% always (3 consecutive failures = episode termination).
During black swans, shed industrial/residential aggressively and discharge BESS.
Leverage the 1h and 3h forecasts to manage the battery proactively."""

def _llm_action(obs) -> dict:
    vec = obs.to_vector() if hasattr(obs, "to_vector") else []
    user_msg = (
        f"Obs vector (22 values): {[round(v, 3) for v in vec]}\n"
        f"solar_swan={obs.solar_swan_active:.0f} wind_swan={obs.wind_swan_active:.0f} "
        f"soc={obs.battery_soc:.2f} freq_norm={obs.frequency_norm:.3f}\n"
        f"hosp_ratio={obs.hosp_served_ratio:.3f} ind_ratio={obs.ind_served_ratio:.3f} "
        f"res_ratio={obs.res_served_ratio:.3f}\nWhat is your dispatch decision?"
    )
    try:
        if not HF_TOKEN:
            raise ValueError("No HF_TOKEN — using heuristic")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=64,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        action = json.loads(raw)
        return {
            "bess":        max(0, min(5, int(action.get("bess", 0)))),
            "hospital":    max(0, min(1, int(action.get("hospital", 0)))),
            "industrial":  max(0, min(3, int(action.get("industrial", 0)))),
            "residential": max(0, min(3, int(action.get("residential", 0)))),
        }
    except Exception:
        return _heuristic_action(obs)

# ══════════════════════════════════════════════════════════════════════════════
# Task 1 — Cost Minimization (8-hour, stable conditions)
# ══════════════════════════════════════════════════════════════════════════════
def run_task1(episode: int = 1) -> tuple[float, int]:
    print(f"[START] task=cost_minimization episode={episode}")
    env = _make_env()
    with env:
        result = env.reset(seed=42, scenario=0)
        obs = result.observation

        threshold_cost = 500.0 * 0.30 * 8 * 0.005
        steps = 0

        for step in range(8):
            action = _action_from_dict(_llm_action(obs))
            result = env.step(action)
            obs    = result.observation
            reward = result.reward or 0.0
            done   = result.done

            print(f"[STEP] step={step+1} action=[{action.bess},{action.hospital},"
                  f"{action.industrial},{action.residential}]"
                  f" reward={reward:.4f} done={done}")
            steps += 1
            if done: break

        state = env.state()
        actual_cost = state.total_cost
        score = max(0.0, min(1.0, round(1.0 - actual_cost / max(threshold_cost, 1e-9), 4)))
        print(f"[END] task=cost_minimization score={score:.4f} steps={steps}")
        return score, steps

# ══════════════════════════════════════════════════════════════════════════════
# Task 2 — Hospital Resilience (24-hour, black swans enabled)
# ══════════════════════════════════════════════════════════════════════════════
def run_task2(episode: int = 1) -> tuple[float, int]:
    print(f"[START] task=hospital_resilience episode={episode}")
    env = _make_env()
    with env:
        result = env.reset(seed=123, scenario=0)
        obs = result.observation
        steps = 0

        for step in range(24):
            action = _action_from_dict(_llm_action(obs))
            result = env.step(action)
            obs    = result.observation
            reward = result.reward or 0.0
            done   = result.done

            print(f"[STEP] step={step+1} action=[{action.bess},{action.hospital},"
                  f"{action.industrial},{action.residential}]"
                  f" reward={reward:.4f} done={done}")
            steps += 1
            if done: break

        state = env.state()
        hosp_fail  = state.hospital_failure_steps
        terminal   = state.hospital_terminal_triggered
        raw = 1.0 - hosp_fail / max(steps, 1) - (0.3 if terminal else 0.0)
        score = round(max(0.0, min(1.0, raw)), 4)
        print(f"[END] task=hospital_resilience score={score:.4f} steps={steps}")
        return score, steps

# ══════════════════════════════════════════════════════════════════════════════
# Task 3 — Renewable Maximization (24-hour, full stochastic)
# ══════════════════════════════════════════════════════════════════════════════
def run_task3(episode: int = 1) -> tuple[float, int]:
    print(f"[START] task=renewable_maximization episode={episode}")
    env = _make_env()
    with env:
        result = env.reset(seed=777, scenario=2) # Test on storm scenario
        obs = result.observation
        total_renew_served = 0.0
        steps = 0

        for step in range(24):
            action = _action_from_dict(_llm_action(obs))
            result = env.step(action)
            obs    = result.observation
            reward = result.reward or 0.0
            done   = result.done
            total_renew_served += obs.solar_norm + obs.wind_norm
            steps += 1

            print(f"[STEP] step={step+1} action=[{action.bess},{action.hospital},"
                  f"{action.industrial},{action.residential}]"
                  f" reward={reward:.4f} done={done}")
            if done: break

        state = env.state()
        avg_renew = (state.renewable_energy_used / max(steps, 1)) / (200 + 150)
        stability = max(0.0, 1.0 - state.frequency_violation_steps / max(steps, 1))
        shed_ratio = (state.total_energy_shed_kwh / max(state.total_energy_demanded_kwh, 1.0))
        shed_score = max(0.0, 1.0 - shed_ratio / 0.20)

        composite = (0.4 * min(1.0, avg_renew) + 0.4 * stability + 0.2 * shed_score)
        score = round(max(0.0, min(1.0, composite)), 4)
        print(f"[END] task=renewable_maximization score={score:.4f} steps={steps}")
        return score, steps

# ══════════════════════════════════════════════════════════════════════════════
# Generate SFT Trajectories
# ══════════════════════════════════════════════════════════════════════════════
def generate_trajectories(num_episodes: int = 500, output_file: str = "grid_expert_sft.jsonl"):
    print(f"Generating {num_episodes} expert SFT trajectories...")
    env = _make_env()
    written_count = 0
    with open(output_file, 'w') as f:
        with env:
            for ep in range(num_episodes):
                scenario_probs = [0.5, 0.15, 0.15, 0.15, 0.05]
                scenario = random.choices(range(5), weights=scenario_probs)[0]
                
                result = env.reset(seed=ep, scenario=scenario)
                obs = result.observation
                
                for step in range(24):
                    # For SFT, format the prompt and completion
                    vec_str = [round(v, 3) for v in obs.to_vector()]
                    prompt = (
                        f"Obs vector (22 values): {vec_str}\n"
                        f"solar_swan={obs.solar_swan_active:.0f} wind_swan={obs.wind_swan_active:.0f} "
                        f"soc={obs.battery_soc:.2f} freq_norm={obs.frequency_norm:.3f}\n"
                        f"hosp_ratio={obs.hosp_served_ratio:.3f} ind_ratio={obs.ind_served_ratio:.3f} "
                        f"res_ratio={obs.res_served_ratio:.3f}\nWhat is your dispatch decision?"
                    )
                    
                    act_dict, reasoning = _heuristic_action_with_reasoning(obs)
                    action = _action_from_dict(act_dict)
                    
                    completion = (
                        f"### Thought: {reasoning}\n"
                        f"### Action: [{act_dict['bess']}, {act_dict['hospital']}, {act_dict['industrial']}, {act_dict['residential']}]"
                    )
                    
                    f.write(json.dumps({
                        "prompt": _SYSTEM_PROMPT + "\n\n" + prompt,
                        "completion": completion
                    }) + "\n")
                    written_count += 1
                    
                    result = env.step(action)
                    obs = result.observation
                    if result.done: break
                
                if (ep + 1) % 50 == 0:
                    print(f"  ... completed {ep + 1}/{num_episodes} episodes")
                    
    print(f"Successfully wrote {written_count} steps to {output_file}")


def main() -> None:
    if "--generate-sft" in sys.argv:
        _start_server()
        try:
            generate_trajectories()
        finally:
            _stop_server()
        return

    t0 = time.time()
    _start_server()
    try:
        scores: dict[str, float] = {}
        s1, st1 = run_task1(episode=1)
        scores["cost_minimization"] = s1
        s2, st2 = run_task2(episode=1)
        scores["hospital_resilience"] = s2
        s3, st3 = run_task3(episode=1)
        scores["renewable_maximization"] = s3

        elapsed = time.time() - t0
        print(f"\n{'='*60}")
        print(f"  Energy Grid v2.1 Baseline Results  ({elapsed:.0f}s)")
        print(f"{'='*60}")
        thresholds = {"cost_minimization": 0.70,
                      "hospital_resilience": 0.80,
                      "renewable_maximization": 0.75}
        for task, score in scores.items():
            thr = thresholds[task]
            status = "✓ PASS" if score >= thr else "✗ FAIL"
            print(f"  {task:<28}  score={score:.4f}  {status}")
        print(f"{'='*60}")
    finally:
        _stop_server()


if __name__ == "__main__":
    main()
