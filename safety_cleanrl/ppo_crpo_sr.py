import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import safety_gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from utils.wraps import RecordEpisodeStatistics

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "SafetyHalfCheetahVelocity-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    sr_learning_rate: float = 1e-4
    """the learning rate for safety representation network"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    sr_coef: float = 1.0
    """coefficient of the safety representation loss"""
    max_grad_norm: float = 40
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    sr_latent_dim: int = 64
    """dimension of safety representation latent space"""

    # Add CRPO specific arguments
    tolerance: float = 2.0
    """the tolerance of crpo method"""
    cost_limit: float = 0.0
    """the cost limit"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            safety_env = safety_gymnasium.make(env_id, render_mode="rgb_array")
        else:
            safety_env = safety_gymnasium.make(env_id)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        safety_env = safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium(env)
        safety_env = RecordEpisodeStatistics(safety_env)
        return safety_env

    return thunk


def layer_init(layer, std=0.1, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SafetyRepresentationNetwork(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, latent_dim))
        )
        
        self.decoder = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, obs_dim))
        )
        
        self.safety_predictor = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1))
        )

    def forward(self, obs):
        z = self.encoder(obs)
        obs_reconstructed = self.decoder(z)
        safety_pred = self.safety_predictor(z)
        return z, obs_reconstructed, safety_pred


class Agent(nn.Module):
    def __init__(self, envs, sr_latent_dim):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        self.sr_network = SafetyRepresentationNetwork(obs_dim, sr_latent_dim)
        
        # Policy network takes both observation and safety representation as input
        self.critic_reward = nn.Sequential(
            layer_init(nn.Linear(obs_dim + sr_latent_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.critic_cost = nn.Sequential(
            layer_init(nn.Linear(obs_dim + sr_latent_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim + sr_latent_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_safety_representation(self, x):
        return self.sr_network(x)

    def get_value(self, x):
        z, _, _ = self.sr_network(x)
        augmented_x = torch.cat([x, z], dim=-1)
        return self.critic_reward(augmented_x), self.critic_cost(augmented_x)

    def get_action_and_value(self, x, action=None):
        z, obs_reconstructed, safety_pred = self.sr_network(x)
        augmented_x = torch.cat([x, z], dim=-1)
        
        action_mean = self.actor_mean(augmented_x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic_reward(augmented_x), self.critic_cost(augmented_x), obs_reconstructed, safety_pred


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = safety_gymnasium.vector.SafetySyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs, args.sr_latent_dim).to(device)
    optimizer = optim.Adam([
        {'params': [p for n, p in agent.named_parameters() if 'sr_network' not in n], 'lr': args.learning_rate},
        {'params': agent.sr_network.parameters(), 'lr': args.sr_learning_rate}
    ], eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    costs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_reward = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_cost = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            for param_group in optimizer.param_groups:
                param_group["lr"] = frac * (args.learning_rate if 'sr_network' not in str(param_group['params'][0]) else args.sr_learning_rate)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value_reward, value_cost, _, safety_pred = agent.get_action_and_value(next_obs)
                values_reward[step] = value_reward.flatten()
                values_cost[step] = value_cost.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game and log data
            next_obs, reward, cost, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)
            costs[step] = torch.tensor(cost, dtype=torch.float32).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}, episodic_cost={info['episode']['c']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        writer.add_scalar("charts/episodic_cost", info["episode"]["c"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_reward, next_value_cost = agent.get_value(next_obs)
            next_value_reward = next_value_reward.reshape(1, -1)
            next_value_cost = next_value_cost.reshape(1, -1)
            advantages_reward = torch.zeros_like(rewards).to(device)
            advantages_cost = torch.zeros_like(costs).to(device)
            lastgaelam_reward = 0
            lastgaelam_cost = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues_reward = next_value_reward
                    nextvalues_cost = next_value_cost
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues_reward = values_reward[t + 1]
                    nextvalues_cost = values_cost[t + 1]
                # Compute reward advantage
                delta_reward = rewards[t] + args.gamma * nextvalues_reward * nextnonterminal - values_reward[t]
                advantages_reward[t] = lastgaelam_reward = delta_reward + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_reward
                # Compute cost advantage
                delta_cost = costs[t] + args.gamma * nextvalues_cost * nextnonterminal - values_cost[t]
                advantages_cost[t] = lastgaelam_cost = delta_cost + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_cost
            returns_reward = advantages_reward + values_reward
            returns_cost = advantages_cost + values_cost

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages_reward = advantages_reward.reshape(-1)
        b_advantages_cost = advantages_cost.reshape(-1)
        b_returns_reward = returns_reward.reshape(-1)
        b_returns_cost = returns_cost.reshape(-1)
        b_values_reward = values_reward.reshape(-1)
        b_values_cost = values_cost.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue_reward, newvalue_cost, obs_reconstructed, safety_pred = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages_reward = b_advantages_reward[mb_inds]
                mb_advantages_cost = b_advantages_cost[mb_inds]
                if args.norm_adv:
                    mb_advantages_reward = (mb_advantages_reward - mb_advantages_reward.mean()) / (mb_advantages_reward.std() + 1e-8)
                    mb_advantages_cost = (mb_advantages_cost - mb_advantages_cost.mean()) / (mb_advantages_cost.std() + 1e-8)

                # CRPO logic for advantage selection
                mb_returns_cost = b_returns_cost[mb_inds]
                if mb_returns_cost.mean() < args.cost_limit + args.tolerance:
                    mb_advantages = mb_advantages_reward
                else:
                    mb_advantages = -mb_advantages_cost

                # Policy loss 
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value losses for both reward and cost critics
                if args.clip_vloss:
                    # Reward value loss
                    v_loss_unclipped_reward = (newvalue_reward - b_returns_reward[mb_inds]) ** 2
                    v_clipped_reward = b_values_reward[mb_inds] + torch.clamp(
                        newvalue_reward - b_values_reward[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped_reward = (v_clipped_reward - b_returns_reward[mb_inds]) ** 2
                    v_loss_max_reward = torch.max(v_loss_unclipped_reward, v_loss_clipped_reward)
                    v_loss_reward = 0.5 * v_loss_max_reward.mean()

                    # Cost value loss
                    v_loss_unclipped_cost = (newvalue_cost - b_returns_cost[mb_inds]) ** 2
                    v_clipped_cost = b_values_cost[mb_inds] + torch.clamp(
                        newvalue_cost - b_values_cost[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped_cost = (v_clipped_cost - b_returns_cost[mb_inds]) ** 2
                    v_loss_max_cost = torch.max(v_loss_unclipped_cost, v_loss_clipped_cost)
                    v_loss_cost = 0.5 * v_loss_max_cost.mean()
                else:
                    v_loss_reward = 0.5 * ((newvalue_reward - b_returns_reward[mb_inds]) ** 2).mean()
                    v_loss_cost = 0.5 * ((newvalue_cost - b_returns_cost[mb_inds]) ** 2).mean()

                # Safety Representation losses
                reconstruction_loss = nn.MSELoss()(obs_reconstructed, b_obs[mb_inds])
                safety_prediction_loss = nn.MSELoss()(safety_pred.squeeze(), b_returns_cost[mb_inds])
                sr_loss = reconstruction_loss + safety_prediction_loss

                # Total loss combining all components
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + \
                       (v_loss_reward + v_loss_cost) * args.vf_coef + \
                       sr_loss * args.sr_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values_reward.cpu().numpy(), b_returns_reward.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss_reward", v_loss_reward.item(), global_step)
        writer.add_scalar("losses/value_loss_cost", v_loss_cost.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/reconstruction_loss", reconstruction_loss.item(), global_step)
        writer.add_scalar("losses/safety_prediction_loss", safety_prediction_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from eval import evaluate

        episodic_returns, episodic_costs = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=lambda envs: Agent(envs, args.sr_latent_dim),
            device=device,
            gamma=args.gamma,
        )
        
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)
        for idx, episodic_cost in enumerate(episodic_costs):
            writer.add_scalar("eval/episodic_cost", episodic_cost, idx)
            
        mean_return, std_return = np.mean(episodic_returns), np.std(episodic_returns)
        mean_cost, std_cost = np.mean(episodic_costs), np.std(episodic_costs)
        print(f"eval_return={mean_return}+-{std_return}, eval_cost={mean_cost}+-{std_cost}")

    envs.close()
    writer.close() 