from typing import Callable

import gymnasium as gym
import safety_gymnasium
import torch
import numpy as np
import random

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool =False,
    gamma: float = 0.99,
    **kwargs,
):
    envs = safety_gymnasium.vector.SafetySyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma)])
    agent = Model(envs, **kwargs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    episodic_costs = []
    while len(episodic_returns) < eval_episodes:
        returns = agent.get_action_and_value(torch.Tensor(obs).to(device))
        actions = returns[0]
        next_obs, _, _, _, _, infos = envs.step(actions.cpu().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}, episodic_cost={info['episode']['c']}")
                episodic_returns += [info["episode"]["r"]]
                episodic_costs += [info["episode"]["c"]]
        obs = next_obs

    return episodic_returns, episodic_costs

if __name__ == "__main__":

  
    model_path = f"runs/SafetyPointGoal1-v0__ppo_lagrangian__1__1748432298/ppo_lagrangian.cleanrl_model"
    import sys
    sys.path.append('.')
    from safety_cleanrl.ppo_lagrangian import make_env, Agent
    env_id = "SafetyPointGoal1-v0"
    device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")
    total_returns, total_costs = [], []
    for _ in range(20):
        episodic_returns, episodic_costs = evaluate(
            model_path,
            make_env,
            env_id,
            eval_episodes=100,
            run_name=f"eval",
            Model=Agent,
            device=device,
            gamma=0.99,
     
            
        )
        total_returns.extend(episodic_returns)
        total_costs.extend(episodic_costs)
    mean_return, std_return = np.mean(total_returns), np.std(total_returns)
    mean_cost, std_cost = np.mean(total_costs), np.std(total_costs)
    print(f"eval_return={mean_return}+-{std_return}, eval_cost={mean_cost}+-{std_cost}")
