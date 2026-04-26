import re
import numpy as np
import torch
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset

# Import the OpenEnv Client
from client import EnergyGridEnv
from models import GridAction

# Connect to your LIVE environment hosted on HF!
ENV_URL = "https://mhussain17-energy-grid-env.hf.space"
env = EnergyGridEnv(base_url=ENV_URL).sync()

def extract_action(text: str) -> list[int]:
    """Parse the [b, h, i, r] action from the completion text."""
    match = re.search(r"### Action:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
    if match:
        return [int(x) for x in match.groups()]
    return None

def remote_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    TRL Reward function hitting the Hugging Face Space API.
    completions is a list of strings (Group Size = 4).
    """
    rewards = []
    
    # Ensure the environment is reset for the start of the evaluation
    env.reset()
    
    for completion in completions:
        action_vals = extract_action(completion)
        if action_vals is None:
            rewards.append(-500.0) # Harsh penalty for bad formatting
            continue
            
        b, h, i, r = action_vals
        try:
            # Hit your remote API!
            result = env.step(GridAction(bess=b, hospital=h, industrial=i, residential=r))
            rewards.append(result.reward)
        except Exception as e:
            print(f"API Error: {e}")
            rewards.append(-50.0) # Penalty for crashing the step
            
    # Local Group-Wise Normalization to prevent Black Swan gradient explosions
    if len(rewards) > 1 and np.std(rewards) > 0:
        rewards = ((np.array(rewards) - np.mean(rewards)) / (np.std(rewards) + 1e-8)).tolist()
        
    return rewards

def main():
    print("Connecting to OpenEnv Server and Loading Model...")
    
    # Load Model using Unsloth for hyper-fast GPU training
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-1.5B-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit = True,
    )
    
    # Setup LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # Load your SFT dataset
    # We map it to pure string text because GRPO expects raw prompts
    raw_dataset = load_dataset("json", data_files="grid_expert_sft_train.jsonl", split="train")
    
    # GRPO formatting expects a simple list of messages
    def format_for_grpo(example):
        return {"prompt": [{"role": "user", "content": example["prompt"]}]}
    
    dataset = raw_dataset.map(format_for_grpo)

    config = GRPOConfig(
        output_dir="./grpo-grid-model",
        learning_rate=1e-5,
        group_size=4, # Keep this small since remote API calls add latency
        max_steps=200, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=remote_reward_fn,
        args=config,
        train_dataset=dataset,
    )

    print("Starting RL Training via Remote API...")
    trainer.train()
    
    # Save the finetuned adapters
    model.save_pretrained("grpo-grid-model-final")
    tokenizer.save_pretrained("grpo-grid-model-final")

if __name__ == "__main__":
    main()
