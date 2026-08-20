"""Vectorized GridWorld environment."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from curious_agent.env.grid_world import GridWorld


class VecGridWorld:
    """Vectorized wrapper around multiple GridWorld instances."""

    def __init__(
        self,
        num_envs: int,
        grid_size: int = 10,
        config: Dict[str, Any] | None = None,
    ) -> None:
        """Initialize the vectorized environment."""
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        self.num_envs = num_envs
        self.config = config or {}

        self.envs: List[GridWorld] = [
            GridWorld(grid_size=grid_size, config=config, render=False)
            for _ in range(num_envs)
        ]

        self.state_dim = self.envs[0].state_dim
        self.num_actions = self.envs[0].num_actions

    def reset(self) -> np.ndarray:
        """Reset all environments and return batched states."""
        states = np.stack([env.reset() for env in self.envs], axis=0)
        return states.astype(np.float32)

    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all environments for the corresponding actions."""
        if len(actions) != self.num_envs:
            raise ValueError(
                f"actions length {len(actions)} != num_envs {self.num_envs}"
            )

        next_states = np.empty((self.num_envs, self.state_dim), dtype=np.float32)
        rewards = np.empty((self.num_envs,), dtype=np.float32)
        dones = np.empty((self.num_envs,), dtype=bool)
        infos: List[Dict[str, Any]] = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, info = env.step(int(action))

            if done:
                next_state = env.reset()

            next_states[i] = next_state
            rewards[i] = reward
            dones[i] = done
            infos.append(info)

        return next_states, rewards, dones, infos

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()

    def __repr__(self) -> str:
        return (
            f"VecGridWorld("
            f"num_envs={self.num_envs}, "
            f"state_dim={self.state_dim}, "
            f"num_actions={self.num_actions})"
        )
