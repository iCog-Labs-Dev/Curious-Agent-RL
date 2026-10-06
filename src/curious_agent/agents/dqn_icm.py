"""ICM curious agent."""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from curious_agent.models.icm import IntrinsicCuriosityModule
from curious_agent.models.q_network import QNetwork, DuelingQNetwork
from curious_agent.utils.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class ICMCuriousAgent:
    """ICM DQN agent."""

    def __init__(
        self,
        state_dim: int = 2,
        num_actions: int = 4,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.config = config or {}

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        agent_config = self.config.get("agent", {})
        self.beta = agent_config.get("beta", 1.0)
        self.gamma = agent_config.get("gamma", 0.99)
        self.epsilon = agent_config.get("epsilon", 1.0)
        self.epsilon_min = agent_config.get("epsilon_min", 0.01)
        self.epsilon_decay = agent_config.get("epsilon_decay", 0.9995)

        self.normalize_curiosity = agent_config.get("normalize_curiosity", True)
        self.target_intrinsic_scale = agent_config.get("target_intrinsic_scale", 1.0)
        self._running_intrinsic = 1e-3

        networks_config = self.config.get("networks", {})

        icm_config = networks_config.get("icm", {})
        self.icm = IntrinsicCuriosityModule(
            state_dim=state_dim,
            num_actions=num_actions,
            feature_dim=icm_config.get("feature_dim", 32),
            hidden_dims=icm_config.get("hidden_dims", [64]),
            learning_rate=icm_config.get("learning_rate", 0.001),
            eta=agent_config.get("eta", 1.0),
            inverse_weight=agent_config.get("inverse_weight", 0.2),
        ).to(self.device)

        q_config = networks_config.get("q_network", {})
        use_dueling = q_config.get("use_dueling", False)

        if use_dueling:
            self.q_network = DuelingQNetwork(
                state_dim=state_dim,
                num_actions=num_actions,
                hidden_dims=q_config.get("hidden_dims", [128, 128]),
                value_hidden_dim=q_config.get("value_hidden_dim", 64),
                advantage_hidden_dim=q_config.get("advantage_hidden_dim", 64),
                learning_rate=q_config.get("learning_rate", 0.001),
            ).to(self.device)
        else:
            self.q_network = QNetwork(
                state_dim=state_dim,
                num_actions=num_actions,
                hidden_dims=q_config.get("hidden_dims", [128, 128]),
                learning_rate=q_config.get("learning_rate", 0.001),
            ).to(self.device)

        self.target_q_network = self._create_target_network()
        self.target_update_frequency = self.config.get("training", {}).get(
            "target_update_frequency", 100
        )
        self.tau = self.config.get("training", {}).get("tau", 0.005)

        training_config = self.config.get("training", {})
        self.replay_buffer = ReplayBuffer(
            capacity=training_config.get("buffer_size", 10000)
        )
        self.batch_size = training_config.get("batch_size", 64)
        self.min_buffer_size = training_config.get("min_buffer_size", 1000)

        self.step_count = 0

        self.curiosity_rewards: List[float] = []
        self.external_rewards: List[float] = []
        self.q_losses: List[float] = []

        logger.info(
            f"ICMCuriousAgent initialized: state_dim={state_dim}, "
            f"num_actions={num_actions}, device={self.device}"
        )
        logger.info(f"ICM: {self.icm}")
        logger.info(f"Q-Network: {self.q_network}")

    def _create_target_network(self) -> nn.Module:
        """Create the target network."""
        target = copy.deepcopy(self.q_network).to(self.device)
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        return target

    def _update_target_network(self) -> None:
        """Update the target network."""
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _encode_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor."""
        return torch.FloatTensor(state).to(self.device)

    def _encode_action(self, action: int) -> torch.Tensor:
        """Convert action to one-hot."""
        action_one_hot = torch.zeros(self.num_actions).to(self.device)
        action_one_hot[action] = 1.0
        return action_one_hot

    def select_action(self, state: np.ndarray) -> int:
        """Select an action with epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
            logger.debug(f"Random action: {action}")
        else:
            state_tensor = self._encode_state(state)
            action = self.q_network.get_action(state_tensor, epsilon=0.0)
            logger.debug(f"Greedy action: {action}")
        return action

    def compute_intrinsic_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """Compute the intrinsic reward."""
        raw_reward = self.icm.intrinsic_reward(state, action, next_state)
        self.curiosity_rewards.append(raw_reward)

        if not self.normalize_curiosity:
            return raw_reward

        self._running_intrinsic = (
            0.99 * self._running_intrinsic + 0.01 * abs(raw_reward)
        )
        normalized = (
            raw_reward / (self._running_intrinsic + 1e-8)
        ) * self.target_intrinsic_scale
        return normalized

    def update_icm(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> Tuple[float, float]:
        """Update the ICM on one transition."""
        return self.icm.update(state, action, next_state)

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_q_network(self) -> Optional[float]:
        """Update the Q-network."""
        if not self.replay_buffer.is_ready(self.min_buffer_size):
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, device=self.device
        )

        loss = self.q_network.update(
            states,
            actions,
            rewards,
            next_states,
            dones,
            gamma=self.gamma,
            target_network=self.target_q_network,
        )

        self.q_losses.append(loss)

        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self._update_target_network()
            logger.debug("Target network updated")

        return loss

    def decay_epsilon(self) -> None:
        """Decay the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        logger.debug(f"Epsilon decayed to: {self.epsilon:.4f}")

    def get_statistics(self) -> Dict:
        """Return agent statistics."""
        icm_stats = {
            "avg_intrinsic_reward": (
                np.mean(self.curiosity_rewards) if self.curiosity_rewards else 0.0
            ),
            "avg_inverse_loss": (
                np.mean(self.icm.inverse_losses[-100:])
                if self.icm.inverse_losses
                else 0.0
            ),
            "avg_forward_loss": (
                np.mean(self.icm.forward_losses[-100:])
                if self.icm.forward_losses
                else 0.0
            ),
        }
        return {
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "buffer_size": len(self.replay_buffer),
            "avg_curiosity_reward": (
                np.mean(self.curiosity_rewards) if self.curiosity_rewards else 0.0
            ),
            "avg_external_reward": (
                np.mean(self.external_rewards) if self.external_rewards else 0.0
            ),
            "avg_q_loss": (
                np.mean(self.q_losses[-100:]) if self.q_losses else 0.0
            ),
            **icm_stats,
        }

    def reset_statistics(self) -> None:
        """Reset episode statistics."""
        self.curiosity_rewards = []
        self.external_rewards = []

    def save(self, path: str) -> None:
        """Save model checkpoints."""
        torch.save(
            {
                "icm": self.icm.state_dict(),
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            },
            path,
        )
        logger.info(f"ICMCuriousAgent saved to {path}")

    def load(self, path: str) -> None:
        """Load model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.icm.load_state_dict(checkpoint["icm"])
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]
        logger.info(f"ICMCuriousAgent loaded from {path}")

    def __repr__(self) -> str:
        return (
            f"ICMCuriousAgent(state_dim={self.state_dim}, "
            f"num_actions={self.num_actions}, "
            f"epsilon={self.epsilon:.3f}, "
            f"device={self.device})"
        )