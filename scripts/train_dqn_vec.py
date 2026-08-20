"""Vectorized DQN training script."""

from __future__ import annotations

import argparse
import collections
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from curious_agent.env.vec_env import VecGridWorld
from curious_agent.agents.dqn_curious import DNQCuriousAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_dqn_vec.log"),
    ],
)
logger = logging.getLogger(__name__)


def ensure_training_log(log_path: str | Path = "training_dqn_vec.log") -> None:
    """Ensure the vectorized DQN log file exists."""
    resolved_path = Path(log_path).resolve()
    for configured_logger in (logger, logging.getLogger()):
        for handler in configured_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if Path(handler.baseFilename).resolve() == resolved_path:
                    return
    file_handler = logging.FileHandler(resolved_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)


def load_config(config_path: str) -> dict:
    """Load a YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_dqn_vec(config: dict, render: bool = False) -> None:
    """Train the vectorized DQN agent."""
    ensure_training_log()

    paths = config.get("paths", {})
    checkpoint_dir = Path(paths.get("checkpoints", "checkpoints/dqn_vec"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_config = config.get("training", {})
    num_episodes = training_config.get("num_episodes", 2000)
    max_steps = training_config.get("max_steps_per_episode", 200)
    log_interval = training_config.get("log_interval", 10)
    save_interval = training_config.get("save_interval", 100)
    num_envs = training_config.get("num_envs", 16)

    env_config = config.get("environment", {})
    grid_size = env_config.get("grid_size", 10)

    vec_env = VecGridWorld(
        num_envs=num_envs,
        grid_size=grid_size,
        config=config,
    )

    logger.info(f"VecGridWorld created: {vec_env}")

    agent = DNQCuriousAgent(
        state_dim=vec_env.state_dim,
        num_actions=vec_env.num_actions,
        config=config,
    )

    device = agent.device
    logger.info(f"Agent device: {device}")
    logger.info(f"Starting vectorized training: num_envs={num_envs}, num_episodes={num_episodes}")

    ep_rewards = np.zeros(num_envs, dtype=np.float64)
    ep_curiosity = np.zeros(num_envs, dtype=np.float64)
    ep_steps = np.zeros(num_envs, dtype=np.int64)

    completed_rewards: list[float] = []
    completed_curiosity: list[float] = []
    completed_lengths: list[int] = []

    zone_curiosity: dict[str, float] = collections.defaultdict(float)
    zone_visits: dict[str, int] = collections.defaultdict(int)

    states = vec_env.reset()

    global_steps = 0

    while len(completed_rewards) < num_episodes:
        states_t = torch.FloatTensor(states).to(device)

        if np.random.random() < agent.epsilon:
            actions = np.random.randint(0, vec_env.num_actions, size=(num_envs,))
        else:
            agent.q_network.eval()
            with torch.no_grad():
                q_values = agent.q_network(states_t)
            actions = q_values.argmax(dim=-1).cpu().numpy()

        actions_oh = torch.zeros(num_envs, vec_env.num_actions, device=device)
        actions_oh.scatter_(1, torch.LongTensor(actions).unsqueeze(1).to(device), 1.0)

        agent.confidence_net.eval()
        with torch.no_grad():
            o_c_before_t = agent.confidence_net(states_t, actions_oh)
        o_c_before = o_c_before_t.squeeze(1).cpu().numpy()

        next_states, ext_rewards, dones, infos = vec_env.step(actions)
        next_states_t = torch.FloatTensor(next_states).to(device)

        agent.world_model.eval()
        with torch.no_grad():
            predicted_next = agent.world_model(states_t, actions_oh)

        sq_err = (predicted_next - next_states_t) ** 2
        actual_error = sq_err.mean(dim=1).cpu().numpy()

        model_loss = agent.world_model.update(states_t, actions_oh, next_states_t)
        agent.model_losses.append(model_loss)

        actual_error_t = torch.FloatTensor(actual_error).unsqueeze(1).to(device)
        conf_loss = agent.confidence_net.update(states_t, actions_oh, actual_error_t)
        agent.confidence_losses.append(conf_loss)

        agent.confidence_net.eval()
        with torch.no_grad():
            o_c_after_t = agent.confidence_net(states_t, actions_oh)
        o_c_after = o_c_after_t.squeeze(1).cpu().numpy()

        raw_curiosity = o_c_before - o_c_after

        if agent.normalize_curiosity:
            batch_mean = float(np.abs(raw_curiosity).mean())
            agent._running_intrinsic = (
                0.99 * agent._running_intrinsic + 0.01 * batch_mean
            )
            curiosity_rewards = (
                raw_curiosity / (agent._running_intrinsic + 1e-8)
            ) * agent.target_intrinsic_scale
        else:
            curiosity_rewards = raw_curiosity

        total_rewards = ext_rewards + agent.beta * curiosity_rewards

        for i in range(num_envs):
            agent.replay_buffer.push(
                states[i],
                int(actions[i]),
                float(total_rewards[i]),
                next_states[i],
                bool(dones[i]),
            )
            zone = infos[i].get("zone_type", "unknown")
            zone_curiosity[zone] += float(curiosity_rewards[i])
            zone_visits[zone] += 1

        agent.update_q_network()
        agent.decay_epsilon()

        ep_rewards += total_rewards
        ep_curiosity += curiosity_rewards
        ep_steps += 1

        for i in range(num_envs):
            if dones[i] or ep_steps[i] >= max_steps:
                completed_rewards.append(float(ep_rewards[i]))
                completed_curiosity.append(float(ep_curiosity[i]))
                completed_lengths.append(int(ep_steps[i]))

                ep_rewards[i] = 0.0
                ep_curiosity[i] = 0.0
                ep_steps[i] = 0

                n_done = len(completed_rewards)

                if n_done % log_interval == 0:
                    window = completed_rewards[-log_interval:]
                    avg_r = sum(window) / len(window)
                    win_c = completed_curiosity[-log_interval:]
                    avg_c = sum(win_c) / len(win_c)
                    win_l = completed_lengths[-log_interval:]
                    avg_l = sum(win_l) / len(win_l)
                    stats = agent.get_statistics()

                    logger.info(
                        f"Episode {n_done}/{num_episodes}: "
                        f"Avg Reward={avg_r:.3f}, "
                        f"Avg Curiosity={avg_c:.3f}, "
                        f"Avg Length={avg_l:.1f}, "
                        f"Epsilon={stats['epsilon']:.3f}, "
                        f"Buffer Size={stats['buffer_size']}"
                    )

                    zone_order = ["static", "deterministic_a", "deterministic_b", "noisy", "dynamic"]
                    zone_parts = []
                    for z in zone_order:
                        visits = zone_visits.get(z, 0)
                        avg_zc = zone_curiosity.get(z, 0.0) / visits if visits > 0 else 0.0
                        zone_parts.append(f"{z}={avg_zc:.4f}")
                    logger.info(f"Zone Curiosity [{', '.join(zone_parts)}]")

                    zone_curiosity.clear()
                    zone_visits.clear()

                if n_done % save_interval == 0:
                    ckpt_path = checkpoint_dir / f"agent_episode_{n_done}.pt"
                    agent.save(str(ckpt_path))
                    logger.info(f"Checkpoint saved to {ckpt_path}")

                if n_done >= num_episodes:
                    break

        states = next_states
        global_steps += 1

    final_path = checkpoint_dir / "agent_final.pt"
    agent.save(str(final_path))
    logger.info(f"Final model saved to {final_path}")

    summary_window = min(100, len(completed_rewards))
    logger.info("\n" + "=" * 50)
    logger.info("Vectorized Training Complete!")
    logger.info(f"Total Completed Episodes : {len(completed_rewards)}")
    logger.info(f"num_envs                 : {num_envs}")
    logger.info(f"Final Epsilon            : {agent.epsilon:.3f}")
    logger.info(
        f"Average Reward  (last {summary_window}): "
        f"{sum(completed_rewards[-summary_window:]) / summary_window:.3f}"
    )
    logger.info(
        f"Average Curiosity (last {summary_window}): "
        f"{sum(completed_curiosity[-summary_window:]) / summary_window:.3f}"
    )
    logger.info("=" * 50)

    vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DQN Curious Agent with Vectorized Environments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dqn.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    config = load_config(args.config)
    train_dqn_vec(config)


if __name__ == "__main__":
    main()
