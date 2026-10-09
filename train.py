"""
Train DDPG or TD3 on LunarLanderContinuous-v3 and measure the critic's bias

uv run train.py --algo td3 --seed 1
uv run train.py --algo ddpg --layernorm --seed 3 --steps 300000

Each run writes one JSON file in `results/` with its configuration, all the
evaluations, and the bias measurements.
"""
import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from rl_mind.collectors import TransitionCollector
from rl_mind.core import Action, Actor
from rl_mind.data import ReplayBuffer, Transitions
from rl_mind.env import VecEnv
from rl_mind.evaluation import Evaluator
from rl_mind.nn import build_mlp, soft_update

# Agents

class ContinuousQNetwork(nn.Module):
    # Q-network $Q(s, a)$ for continuous actions
    def __init__(
        self,
        obs_dim: int,
        hidden: tuple[int, ...],
        action_dim: int,
        layernorm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        n_in = obs_dim + action_dim
        for n_out in hidden:
            layers.append(nn.Linear(n_in, n_out))
            if layernorm:
                layers.append(nn.LayerNorm(n_out))
            layers.append(nn.ReLU())
            n_in = n_out
        layers.append(nn.Linear(n_in, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        # Compute Q(s, a) for a batch [B, obs_dim] x [B, action_dim] -> [B]
        return self.model(torch.cat([obs, action], dim=1)).squeeze(-1)


class ContinuousDeterministicActor(Actor[Action]):
    def __init__(self, obs_dim: int, hidden: tuple[int, ...], action_dim: int):
        super().__init__()
        self.model = build_mlp(
            [obs_dim, *hidden, action_dim], output_activation=nn.Tanh()
        )

    def forward(self, obs: Tensor) -> Action:
        return Action(value=self.model(obs))


class GaussianNoise(Actor[Action]):
    # Adds Gaussian noise to the actions of another actor (at training time)

    def __init__(self, actor: Actor[Action], sigma: float):
        super().__init__()
        self.actor = actor
        self.sigma = sigma

    def forward(self, obs: Tensor) -> Action:
        action = self.actor(obs).value
        return Action(value=action + self.sigma * torch.randn_like(action))

    def act(self, obs: Tensor) -> Tensor:
        return self.actor.act(obs)

# Losses

def compute_critic_loss(
    gamma: float, batch: Transitions[Action], q_values: Tensor, next_q_values: Tensor
) -> Tensor:
    target = batch.reward + gamma * (~batch.terminated).float() * next_q_values
    return F.mse_loss(q_values, target)


def compute_actor_loss(q_values: Tensor) -> Tensor:
    return -q_values.mean()

# Hyper-parameters

@dataclass(frozen=True)
class DDPGConfig:
    env_name: str = "LunarLanderContinuous-v3"
    seed: int = 1

    # Total number of environment steps
    max_steps: int = 300_000
    # Number of parallel training environments
    n_envs: int = 1
    # Environment steps between two gradient updates
    steps_per_update: int = 1
    # Steps before learning starts
    learning_starts: int = 10_000

    # Replay buffer capacity
    buffer_size: int = 1_000_000
    batch_size: int = 256

    # Discount factor
    gamma: float = 0.99
    # Target network update coefficient
    tau: float = 0.005
    # Exploration noise
    action_noise: float = 0.1

    actor_hidden: tuple[int, ...] = (256, 256)
    critic_hidden: tuple[int, ...] = (256, 256)
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4

    # Steps between two evaluations, and number of evaluation episodes
    eval_interval: int = 10_000
    n_eval_envs: int = 10

    # LayerNorm in the critic.s
    layernorm: bool = False
    # Bias measurement
    n_bias_episodes: int = 5
    bias_horizon: int = 500


@dataclass(frozen=True)
class TD3Config(DDPGConfig):
    # Number of critic updates between two policy updates
    policy_delay: int = 2
    # Std of the noise added to the target policy actions
    target_noise: float = 0.2
    # Clipping of the target policy noise
    target_noise_clip: float = 0.5

# Bias measurement: critic's Q(s, a) vs Monte Carlo returns

def measure_bias(
    actor: Actor, critics: dict[str, nn.Module], cfg: DDPGConfig, seed: int
) -> dict:
    # Compare the critics' Q-values with Monte Carlo returns

    env = gym.make(cfg.env_name)
    all_obs, all_actions, all_returns = [], [], []
    n_terminated = 0

    for episode in range(cfg.n_bias_episodes):
        obs, _ = env.reset(seed=seed + episode)
        observations, actions, rewards = [], [], []
        done = False
        while not done:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = actor.act(obs_tensor)[0].numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs
            done = terminated or truncated

        T = len(rewards)
        returns = np.zeros(T)
        g = 0.0
        for t in reversed(range(T)):
            g = rewards[t] + cfg.gamma * g
            returns[t] = g

        if terminated:
            n_terminated += 1
            keep = T
        else:
            keep = max(T - cfg.bias_horizon, 0)
        all_obs.extend(observations[:keep])
        all_actions.extend(actions[:keep])
        all_returns.extend(returns[:keep])

    env.close()

    result = {
        "n_samples": len(all_returns),
        "n_terminated_episodes": n_terminated,
        "mean_return_mc": float(np.mean(all_returns)) if all_returns else None,
        "mean_abs_return_mc": float(np.mean(np.abs(all_returns))) if all_returns else None,
    }
    if not all_returns:
        return result

    obs_tensor = torch.as_tensor(np.array(all_obs), dtype=torch.float32)
    action_tensor = torch.as_tensor(np.array(all_actions), dtype=torch.float32)
    returns_tensor = torch.as_tensor(np.array(all_returns), dtype=torch.float32)
    with torch.no_grad():
        q_by_name = {name: critic(obs_tensor, action_tensor) for name, critic in critics.items()}
    if len(q_by_name) == 2:
        q_by_name["qmin"] = torch.min(*q_by_name.values())
    for name, q in q_by_name.items():
        diff = q - returns_tensor
        result[name] = {
            "mean_q": float(q.mean()),
            "mean_bias": float(diff.mean()),
            "std_bias": float(diff.std()),
        }
    return result

def log_evaluation(log: dict, result, actor, critics, cfg: DDPGConfig):
    bias = measure_bias(actor, critics, cfg, seed=cfg.seed * 10_000_000 + result.step)
    log["evaluations"].append(
        {
            "step": result.step,
            "rewards": result.rewards.tolist(),
            "mean_reward": result.mean,
            "bias": bias,
        }
    )
    main = bias.get("q1", bias.get("q"))
    bias_text = f"{main['mean_bias']:8.1f}" if main else "     n/a"
    print(
        f"step={result.step:7d}  eval={result.mean:7.1f}  bias={bias_text}"
        f"  (n={bias['n_samples']})",
        flush=True,
    )

# DDPG

def run_ddpg(cfg: DDPGConfig) -> dict:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)

    actor = ContinuousDeterministicActor(
        env.observation_dim, cfg.actor_hidden, env.action_dim
    )
    critic = ContinuousQNetwork(
        env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.layernorm
    )
    target_critic = copy.deepcopy(critic)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    evaluator = Evaluator(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,
    )
    log = {"config": asdict(cfg), "algo": "ddpg", "evaluations": []}

    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        if len(buffer) < cfg.learning_starts:
            continue

        batch = buffer.sample(cfg.batch_size)

        # Update the critic
        q_values = critic(batch.obs, batch.action.value)
        with torch.no_grad():
            next_q_values = target_critic(batch.next_obs, actor(batch.next_obs).value)
        critic_loss = compute_critic_loss(cfg.gamma, batch, q_values, next_q_values)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update the actor
        actor_loss = compute_actor_loss(critic(batch.obs, actor(batch.obs).value))

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Update the target critic
        soft_update(critic, target_critic, cfg.tau)

        if result := evaluator.run_if_needed(collector.steps, actor):
            log_evaluation(log, result, actor, {"q": critic}, cfg)

    return log

# TD3

def run_td3(cfg: TD3Config) -> dict:
    torch.manual_seed(cfg.seed)
    env = VecEnv(cfg.env_name, cfg.n_envs, seed=cfg.seed)

    actor = ContinuousDeterministicActor(
        env.observation_dim, cfg.actor_hidden, env.action_dim
    )
    target_actor = copy.deepcopy(actor)

    critic_1 = ContinuousQNetwork(
        env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.layernorm
    )
    critic_2 = ContinuousQNetwork(
        env.observation_dim, cfg.critic_hidden, env.action_dim, cfg.layernorm
    )

    target_critic_1 = copy.deepcopy(critic_1)
    target_critic_2 = copy.deepcopy(critic_2)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_optimizer = torch.optim.Adam(
        list(critic_1.parameters()) + list(critic_2.parameters()), lr=cfg.lr_critic
    )

    collector = TransitionCollector(env, GaussianNoise(actor, cfg.action_noise))
    buffer = ReplayBuffer(cfg.buffer_size)
    
    evaluator = Evaluator(
        VecEnv(cfg.env_name, cfg.n_eval_envs, seed=cfg.seed + 100),
        every=cfg.eval_interval,
    )
    log = {"config": asdict(cfg), "algo": "td3", "evaluations": []} 

    updates = 0
    while collector.steps < cfg.max_steps:
        buffer.add(collector.collect(cfg.steps_per_update))
        if len(buffer) < cfg.learning_starts:
            continue

        batch = buffer.sample(cfg.batch_size)

        with torch.no_grad():
            action = target_actor(batch.next_obs).value
            noise = (torch.randn_like(action) * cfg.target_noise).clamp(
                -cfg.target_noise_clip, cfg.target_noise_clip
            )
            next_action = (action + noise).clamp(-1.0, 1.0)
            next_q_values = torch.min(
                target_critic_1(batch.next_obs, next_action),
                target_critic_2(batch.next_obs, next_action),
            )

        q_values_1 = critic_1(batch.obs, batch.action.value)
        q_values_2 = critic_2(batch.obs, batch.action.value)
        critic_loss = compute_critic_loss(cfg.gamma, batch, q_values_1, next_q_values) \
                    + compute_critic_loss(cfg.gamma, batch, q_values_2, next_q_values)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        updates += 1

        if updates % cfg.policy_delay == 0:
            actor_loss = compute_actor_loss(critic_1(batch.obs, actor(batch.obs).value))
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            soft_update(actor, target_actor, cfg.tau)
            soft_update(critic_1, target_critic_1, cfg.tau)
            soft_update(critic_2, target_critic_2, cfg.tau)

        if result := evaluator.run_if_needed(collector.steps, actor):
            log_evaluation(log, result, actor, {"q1": critic_1, "q2": critic_2}, cfg)

    return log

# Command line

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--algo", choices=["ddpg", "td3"], required=True)
    parser.add_argument("--layernorm", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=DDPGConfig.max_steps)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    # One thread per run, so that several runs can share the CPU
    torch.set_num_threads(1)

    config_class = TD3Config if args.algo == "td3" else DDPGConfig
    cfg = config_class(seed=args.seed, max_steps=args.steps, layernorm=args.layernorm)
    name = f"{args.algo}{'-ln' if args.layernorm else ''}-s{args.seed}"

    # skip the runs that are already done
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.json"
    if path.exists():
        print(f"{name} already done ({path}), skipping", flush=True)
        return
    print(f"Running {name}: {cfg}", flush=True)

    start = time.time()
    log = run_td3(cfg) if args.algo == "td3" else run_ddpg(cfg)
    log["wall_time_min"] = (time.time() - start) / 60
    log["versions"] = {"torch": torch.__version__, "gymnasium": gym.__version__}

    path.write_text(json.dumps(log, indent=1))
    print(f"Done in {log['wall_time_min']:.1f} min, results in {path}", flush=True)


if __name__ == "__main__":
    main()