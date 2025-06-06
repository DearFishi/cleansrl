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
    max_grad_norm: float = 40
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.02
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    init_value: float = 0.001
    """the init value of lagrangian multi"""
    lagrangian_learning_rate: float = 0.035
    """the learning rate of the lagrangian optimizer"""
    cost_limit: float = 0.0
    """the cost limit"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            safety_env = safety_gymnasium.make(env_id, render_mode="rgb_array")
            # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            safety_env = safety_gymnasium.make(env_id)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        safety_env = safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium(env)
        safety_env = RecordEpisodeStatistics(safety_env)
        return safety_env

    return thunk


def layer_init(layer, std=0.1, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic_reward = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.critic_cost = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic_reward(x), self.critic_cost(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic_reward(x), self.critic_cost(x)


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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = safety_gymnasium.vector.SafetySyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=1e-4)    # optimizer = optim.Adam([agent.actor_logstd]+list(agent.actor_mean.parameters())+list(agent.critic_reward.parameters()), lr=args.learning_rate, eps=1e-5, weight_decay=1e-4)

    lagrangian_multiplier = nn.Parameter(torch.as_tensor(args.init_value), requires_grad=True)
    lagrangian_optimizer = optim.Adam([lagrangian_multiplier], lr=args.lagrangian_learning_rate, eps=1e-5)

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
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value_reward, value_cost = agent.get_action_and_value(next_obs)
                values_reward[step] = value_reward.flatten()
                values_cost[step] = value_cost.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
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
                delta_reward = rewards[t] + args.gamma * nextvalues_reward * nextnonterminal - values_reward[t]
                advantages_reward[t] = lastgaelam_reward = delta_reward + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_reward
                delta_cost = costs[t] + args.gamma * nextvalues_cost * nextnonterminal - values_cost[t]
                advantages_cost[t] = lastgaelam_cost = delta_cost + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam_cost
            returns_reward = advantages_reward + values_reward
            returns_cost = advantages_cost + values_cost

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages_reward = advantages_reward.reshape(-1)
        b_returns_reward = returns_reward.reshape(-1)
        b_values_reward = values_reward.reshape(-1)
        b_advantages_cost = advantages_cost.reshape(-1)
        b_returns_cost = returns_cost.reshape(-1)
        b_values_cost = values_cost.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)

            # Optimizing the lagrangian multiplier
            lagrangian_optimizer.zero_grad()
            penalty_loss = -lagrangian_multiplier * (cost - args.cost_limit).mean()
            penalty_loss.backward()
            lagrangian_optimizer.step()

            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue_reward, newvalue_cost = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages_reward = b_advantages_reward[mb_inds]
                mb_advantages_cost = b_advantages_cost[mb_inds]
                if args.norm_adv:
                    mb_advantages_reward = (mb_advantages_reward - mb_advantages_reward.mean()) / (mb_advantages_reward.std() + 1e-8)
                    mb_advantages_cost = (mb_advantages_cost - mb_advantages_cost.mean()) / (mb_advantages_cost.std() + 1e-8)

                penalty = lagrangian_multiplier.item()
                mb_advantages = (mb_advantages_reward - max(0, penalty) * mb_advantages_cost) / (1 + max(0, penalty))

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue_reward = newvalue_reward.view(-1)
                newvalue_cost = newvalue_cost.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped_reward = (newvalue_reward - b_returns_reward[mb_inds]) ** 2
                    v_clipped_reward = b_values_reward[mb_inds] + torch.clamp(
                        newvalue_reward - b_values_reward[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped_reward = (v_clipped_reward - b_returns_reward[mb_inds]) ** 2
                    v_loss_max_reward = torch.max(v_loss_unclipped_reward, v_loss_clipped_reward)
                    v_loss_reward = 0.5 * v_loss_max_reward.mean()

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
                    v_loss_cost= 0.5 * ((newvalue_cost - b_returns_cost[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss_reward* args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            


            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values_reward.cpu().numpy(), b_returns_reward.cpu().numpy()
        var_y = np.var(y_true)
        explained_var_reward = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        y_pred, y_true = b_values_cost.cpu().numpy(), b_returns_cost.cpu().numpy()
        var_y = np.var(y_true)
        explained_var_cost = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_reward_loss", v_loss_reward.item(), global_step)
        writer.add_scalar("losses/value_cost_loss", v_loss_cost.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance_reward", explained_var_reward, global_step)
        writer.add_scalar("losses/explained_variance_cost", explained_var_cost, global_step)
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
            Model=Agent,
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