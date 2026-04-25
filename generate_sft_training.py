import json
import random
import os
from server.energy_grid_environment import EnergyGridEnvironment
from models import GridAction

# Use the exact expert heuristic reasoning from the inference baseline
from inference import _heuristic_action_with_reasoning, _SYSTEM_PROMPT

def generate_trajectories(num_episodes=500, output_file="grid_expert_sft.jsonl"):
    print(f"Generating {num_episodes} expert SFT trajectories...")
    env = EnergyGridEnvironment()
    dataset = []
    written_count = 0

    for ep in range(num_episodes):
        # Sample a random scenario (0-4) for diversity
        scenario = random.randint(0, 4)
        
        # In v2.1, reset returns (obs, info) because of the recent update
        obs, info = env.reset(seed=ep, scenario=scenario)
        done = False
        
        while not done:
            # 1. Get Expert Action & Reasoning
            act_dict, reasoning = _heuristic_action_with_reasoning(obs)
            
            # Construct the Action Object for the environment
            action = GridAction(
                bess=act_dict['bess'],
                hospital=act_dict['hospital'],
                industrial=act_dict['industrial'],
                residential=act_dict['residential']
            )
            
            # 2. Format SFT Prompt (Match exactly what GRPO/inference sees)
            vec_str = [round(v, 3) for v in obs.to_vector()]
            prompt = (
                f"Obs vector (22 values): {vec_str}\n"
                f"solar_swan={obs.solar_swan_active:.0f} wind_swan={obs.wind_swan_active:.0f} "
                f"soc={obs.battery_soc:.2f} freq_norm={obs.frequency_norm:.3f}\n"
                f"hosp_ratio={obs.hosp_served_ratio:.3f} ind_ratio={obs.ind_served_ratio:.3f} "
                f"res_ratio={obs.res_served_ratio:.3f}\nWhat is your dispatch decision?"
            )
            
            completion = (
                f"### Thought: {reasoning}\n"
                f"### Action: [{act_dict['bess']}, {act_dict['hospital']}, {act_dict['industrial']}, {act_dict['residential']}]"
            )

            dataset.append({
                "prompt": _SYSTEM_PROMPT + "\n\n" + prompt,
                "completion": completion
            })
            
            # 3. Step the environment
            obs = env.step(action)
            done = obs.done
            written_count += 1
            
        if (ep + 1) % 50 == 0:
            print(f"  ... completed {ep + 1}/{num_episodes} episodes")

    # 4. Write Dataset
    with open(output_file, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully wrote {written_count} steps to {output_file}")

if __name__ == "__main__":
    generate_trajectories()