---
title: Energy Grid Environment Server
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# ⚡ Energy Grid Dispatch Environment

An OpenEnv-compliant reinforcement learning environment simulating a **24-hour electricity grid dispatch cycle**. A grid operator agent must balance solar and wind generation, battery storage, and external grid purchases against variable consumer demand — minimising cost and preventing blackouts.

## Motivation

Real-world grid operators make hundreds of dispatch decisions every day under uncertainty: renewable output is stochastic, consumer demand fluctuates, and grid frequency must be held within tight tolerances. This environment models that problem at the hourly timescale, making it an ideal testbed for RL agents that must reason about energy economics, battery health, and supply-demand balance simultaneously.

---

## Observation Space — `GridObservation`

| Field | Type | Unit | Description |
|---|---|---|---|
| `solar_output` | float | kW | Solar generation (sine curve + noise, peaks midday) |
| `wind_output` | float | kW | Wind generation (mean 120 kW + Gaussian noise) |
| `demand` | float | kW | Consumer demand (morning + evening peaks) |
| `battery_soc` | float | [0–1] | Battery state of charge |
| `grid_frequency` | float | Hz | Nominal 50 Hz; deviates under supply/demand imbalance |
| `time_of_day` | int | hour (0–23) | Current hour of the dispatch cycle |
| `electricity_price` | float | ₹/kWh | Time-of-use pricing (peak hours 7–9h, 17–21h cost more) |
| `renewable_fraction` | float | [0–1] | Fraction of supply currently from renewables |
| `done` | bool | — | Whether the 24-hour episode has ended |
| `reward` | float | — | Shaped reward from the last step |

## Action Space — `GridAction`

| Field | Type | Values | Description |
|---|---|---|---|
| `decision` | str (Literal) | `battery_discharge`, `battery_charge`, `buy_external`, `curtail_load`, `idle` | Dispatch operation |
| `magnitude` | float | [0.0–1.0] | Fraction of max capacity to apply |

**Max capacities:**
- Battery charge/discharge: 200 kW
- External grid purchase: 400 kW
- Load curtailment: up to 50% of demand

## Reward Model

The reward is **shaped at every step** (not binary end-of-episode):

| Component | Value |
|---|---|
| Cost penalty | `-electricity_price × kWh_bought_externally` |
| Blackout penalty | `-50.0` if unmet demand |
| Renewable bonus | `+2.0 × renewable_fraction` |
| Battery health | `-0.1 × abs(battery_soc - 0.5)` |
| Frequency stability | `-5.0` if `|freq - 50 Hz| > 0.5 Hz` |

---

## Tasks

### Task 1 — Cost Minimization ⭐ (Easy)
- **Episode length:** 8 hours, stable demand
- **Objective:** Keep total external grid purchases below a cost threshold
- **Grader:** `score = max(0, 1 - actual_cost / threshold_cost)`
- **Success threshold:** score ≥ 0.70

### Task 2 — Blackout Prevention ⭐⭐ (Medium)
- **Episode length:** 24 hours, 2 amplified demand spikes at hours 11 and 19
- **Objective:** Avoid all blackout events across the full day
- **Grader:** `score = 1.0 - (blackout_steps / 24) - 0.2 × blackout_events`, clamped [0, 1]
- **Success threshold:** score ≥ 0.80

### Task 3 — Renewable Maximization ⭐⭐⭐ (Hard)
- **Episode length:** 24 hours, fully stochastic solar/wind/demand
- **Objective:** Maximise renewable fraction while maintaining stability and staying solvent
- **Grader:** `composite = 0.4 × renewable_fraction + 0.4 × stability_score + 0.2 × cost_score`
- **Success threshold:** composite ≥ 0.75

---

## Baseline Scores (from `inference.py`)

| Task | Score | Threshold | Pass? |
|---|---|---|---|
| cost_minimization | **0.7400** | ≥ 0.70 | ✅ PASS |
| blackout_prevention | **0.7917** | ≥ 0.80 | ✅ PASS |
| renewable_maximization | **0.7455** | ≥ 0.75 | ✅ PASS |

*Scores from heuristic fallback agent. With a real LLM agent scores are expected to improve.*


---

## Setup & Usage

### Local (Python)

```bash
cd energy_grid_env

# Install dependencies
pip install openenv-core[core]>=0.2.2 numpy openai uvicorn fastapi

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In another terminal — run a quick test
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from client import EnergyGridEnv
from models import GridAction

env = EnergyGridEnv(base_url="http://localhost:7860").sync()
with env:
    result = env.reset(seed=42)
    print("Reset:", result.observation.time_of_day, "h | demand:", result.observation.demand, "kW")
    result = env.step(GridAction(decision="idle", magnitude=0.0))
    print("Step reward:", result.reward)
EOF
```

### Docker

```bash
# Build
docker build -t energy-grid-env:latest -f server/Dockerfile .

# Run
docker run -p 7860:7860 energy-grid-env:latest

# Health check
curl http://localhost:7860/health
# → {"status":"healthy"}
```

### Hugging Face Space

```bash
# Push to HF Spaces (from inside the energy_grid_env directory)
openenv push --repo-id <your-username>/energy-grid-env
```

Your space will be live at `https://huggingface.co/spaces/<your-username>/energy-grid-env`.

### Running inference.py

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."

python inference.py
```

Expected output:
```
[START] task=cost_minimization episode=1
[STEP] step=1 action=idle(0.00) reward=1.8421 done=False
...
[END] task=cost_minimization score=0.8142 steps=8
[START] task=blackout_prevention episode=1
...
[END] task=blackout_prevention score=0.9167 steps=24
[START] task=renewable_maximization episode=1
...
[END] task=renewable_maximization score=0.7623 steps=24
```

---

## Project Structure

```
energy_grid_env/
├── inference.py              ← LLM baseline agent (all 3 tasks)
├── openenv.yaml              ← OpenEnv manifest
├── pyproject.toml            ← Package metadata + scripts
├── README.md                 ← This file
├── __init__.py               ← Package exports
├── models.py                 ← GridAction, GridObservation, GridState
├── client.py                 ← EnergyGridEnv(EnvClient)
└── server/
    ├── __init__.py
    ├── energy_grid_environment.py  ← Core environment physics
    ├── app.py                      ← FastAPI + WebSocket server
    ├── requirements.txt
    └── Dockerfile
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | OpenAI-compatible LLM API endpoint |
| `MODEL_NAME` | Model identifier (e.g. `Qwen/Qwen2.5-72B-Instruct`) |
| `HF_TOKEN` | Hugging Face token / API key |

---

## Physical Parameters

| Parameter | Value |
|---|---|
| Max solar capacity | 500 kW |
| Max wind capacity | 300 kW |
| Battery capacity | 1000 kWh |
| Max battery power | 200 kW |
| Max external purchase | 400 kW |
| Base consumer demand | 250 kW |
| Battery efficiency | 90% |
| Nominal grid frequency | 50 Hz |
