"""ICM DQN training script."""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curious_agent.env.grid_world import GridWorld
from curious_agent.agents.dqn_icm import ICMCuriousAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training_icm.log"),
    ],
)
logger = logging.getLogger(__name__)


def ensure_training_log(log_path: str | Path = "training_icm.log") -> logging.Handler | None:
    """Ensure the ICM log file exists."""
    resolved_path = Path(log_path).resolve()

    for configured_logger in (logger, logging.getLogger()):
        for handler in configured_logger.handlers:
            if not isinstance(handler, logging.FileHandler):
                continue
            if Path(handler.baseFilename).resolve() == resolved_path:
                return None

    file_handler = logging.FileHandler(resolved_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return file_handler


def load_config(config_path: str) -> dict:
    """Load a YAML config."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_dqn_icm(config: dict, render: bool = False) -> None:
    """Train the ICM DQN curious agent."""
    ensure_training_log()

    paths = config.get("paths", {})
    checkpoint_dir = Path(paths.get("checkpoints", "checkpoints/icm"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env_config = config.get("environment", {})
    grid_size = env_config.get("grid_size", 10)

    env = GridWorld(
        grid_size=grid_size,
        config=config,
        render=render,
    )

    if render:
        env.init_pygame()

    agent = ICMCuriousAgent(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
        config=config,
    )

    training_config = config.get("training", {})
    num_episodes = training_config.get("num_episodes", 2000)
    max_steps = training_config.get("max_steps_per_episode", 200)
    log_interval = training_config.get("log_interval", 10)
    save_interval = training_config.get("save_interval", 100)

    episode_rewards = []
    episode_curiosity_rewards = []
    episode_lengths = []

    logger.info(f"Starting training for {num_episodes} episodes")
    logger.info(f"Grid size: {grid_size}x{grid_size}")
    logger.info(f"Agent: {agent}")

    for episode in range(num_episodes):
        state = env.reset()

        total_reward = 0.0
        total_curiosity = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            action = agent.select_action(state)

            next_state, r_ext, done, info = env.step(action)

            r_curiosity = agent.compute_intrinsic_reward(state, action, next_state)

            agent.update_icm(state, action, next_state)

            beta = agent.beta
            r_total = r_ext + beta * r_curiosity

            agent.store_experience(state, action, r_total, next_state, done)

            agent.update_q_network()

            total_reward += r_total
            total_curiosity += r_curiosity

            state = next_state
            step += 1

            if render:
                env.render_frame()

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_curiosity_rewards.append(total_curiosity)
        episode_lengths.append(step)

        if (episode + 1) % log_interval == 0:
            avg_reward = sum(episode_rewards[-log_interval:]) / log_interval
            avg_curiosity = sum(episode_curiosity_rewards[-log_interval:]) / log_interval
            avg_length = sum(episode_lengths[-log_interval:]) / log_interval

            stats = agent.get_statistics()

            logger.info(
                f"Episode {episode + 1}/{num_episodes}: "
                f"Avg Reward={avg_reward:.3f}, "
                f"Avg Curiosity={avg_curiosity:.3f}, "
                f"Avg Length={avg_length:.1f}, "
                f"Epsilon={stats['epsilon']:.3f}"
            )

        if (episode + 1) % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"agent_episode_{episode + 1}.pt"
            agent.save(str(checkpoint_path))
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    final_path = checkpoint_dir / "agent_final.pt"
    agent.save(str(final_path))
    logger.info(f"Final model saved to {final_path}")

    logger.info("\n" + "=" * 50)
    logger.info("Training Complete!")
    logger.info(f"Total Episodes: {num_episodes}")
    logger.info(f"Final Epsilon: {agent.epsilon:.3f}")
    summary_window = min(100, len(episode_rewards))
    logger.info(
        f"Average Reward (last {summary_window}): "
        f"{sum(episode_rewards[-summary_window:]) / summary_window:.3f}"
    )
    logger.info(
        f"Average Curiosity (last {summary_window}): "
        f"{sum(episode_curiosity_rewards[-summary_window:]) / summary_window:.3f}"
    )
    logger.info("=" * 50)

    env.close()


def main():
    """Run the training entry point."""
    parser = argparse.ArgumentParser(description="Train ICM DQN Curious Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/icm.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with Pygame",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )

    args = parser.parse_args()

    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)
        import torch
        torch.manual_seed(args.seed)

    config = load_config(args.config)

    train_dqn_icm(config, render=args.render)


if __name__ == "__main__":
    main()