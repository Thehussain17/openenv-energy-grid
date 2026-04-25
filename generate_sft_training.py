import json
import numpy as np
from server.energy_grid_environment import EnergyGridEnvironment
from inference import _heuristic_action 

def generate_trajectories(num_episodes=500):
    env = EnergyGridEnvironment()
    dataset = []

    for ep in range(num_episodes):
        # Sample a random scenario (0-4) for diversity
        scenario = np.random.randint(0, 5)
        obs, info = env.reset(options={"scenario": scenario})
        done = False
        
        while not done:
            action_vector = _heuristic_action(obs)
            
            # Construct the reasoning trace (Thought)
            # This teaches the model to correlate obs[12] (freq) and obs[14/15] (swans)
            thought = f"Step {int(obs[16]*24)}: "
            if obs[14] > 0 or obs[15] > 0:
                thought += "Emergency detected (Black Swan active). "
            
            if obs[12] < 0.45: # Freq < 49.9Hz
                thought += "Frequency dropping below nominal. Discharging BESS and shedding non-essential load. "
            elif obs[4] < 0.2:
                thought += "BESS SoC critical. Prioritizing hospital while shedding residential. "
            else:
                thought += "Grid stable. Balancing renewable generation with load."

            dataset.append({
                "prompt": f"Manage the 500kW microgrid. Observation: {obs.tolist()}",
                "completion": f"### Thought: {thought} ### Action: {action_vector.tolist()}"
            })
            
            obs, reward, term, trunc, info = env.step(action_vector)
            done = term or trunc

    with open("grid_expert_sft.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    generate_trajectories()