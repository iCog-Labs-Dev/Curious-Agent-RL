"""
DQN Curious Agent - Phase 2 Implementation

This module implements Schmidhuber's curious agent using neural networks
for the world model (M), confidence network (C), and Q-function (Q).

Key differences from tabular version:
1. M and C are neural networks trained online
2. Q-network uses experience replay for stable training
3. Curiosity reward is computed per-step using C's before/after predictions
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from curious_agent.models.world_model import WorldModel
from curious_agent.models.confidence_net import ConfidenceNetwork
from curious_agent.models.q_network import QNetwork, DuelingQNetwork
from curious_agent.utils.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class DNQCuriousAgent:
    """
    DQN implementation of Schmidhuber's curious agent.
    
    Uses neural networks for:
    - M: World model (predicts next state)
    - C: Confidence network (predicts M's error)
    - Q: Q-function (evaluates state-action pairs)
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        num_actions: int = 4,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the DQN curious agent.
        
        Args:
            state_dim: Dimension of state space
            num_actions: Number of discrete actions
            config: Configuration dictionary
            device: Device to use for computations
        """
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.config = config or {}
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Hyperparameters
        agent_config = self.config.get("agent", {})
        self.beta = agent_config.get("beta", 1.0)
        self.gamma = agent_config.get("gamma", 0.99)
        self.epsilon = agent_config.get("epsilon", 1.0)
        self.epsilon_min = agent_config.get("epsilon_min", 0.01)
        self.epsilon_decay = agent_config.get("epsilon_decay", 0.9995)

        # Curiosity normalization.
        # Normalize raw curiosity to keep beta comparable across agents.
        self.normalize_curiosity = agent_config.get("normalize_curiosity", True)
        self.target_intrinsic_scale = agent_config.get("target_intrinsic_scale", 1.0)
        self._running_intrinsic = 1e-3
        
        # Network configuration
        networks_config = self.config.get("networks", {})
        
        # World Model M
        wm_config = networks_config.get("world_model", {})
        self.world_model = WorldModel(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dims=wm_config.get("hidden_dims", [64, 64]),
            learning_rate=wm_config.get("learning_rate", 0.001),
        ).to(self.device)
        
        # Confidence Network C
        cn_config = networks_config.get("confidence_net", {})
        self.confidence_net = ConfidenceNetwork(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_dims=cn_config.get("hidden_dims", [32, 32]),
            learning_rate=cn_config.get("learning_rate", 0.0005),
        ).to(self.device)
        
        # Q-Network
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
        
        # Target network for stability
        self.target_q_network = self._create_target_network()
        self.target_update_frequency = self.config.get("training", {}).get(
            "target_update_frequency", 100
        )
        self.tau = self.config.get("training", {}).get("tau", 0.005)
        
        # Replay buffer
        training_config = self.config.get("training", {})
        self.replay_buffer = ReplayBuffer(
            capacity=training_config.get("buffer_size", 10000)
        )
        self.batch_size = training_config.get("batch_size", 64)
        self.min_buffer_size = training_config.get("min_buffer_size", 1000)
        
        # Step counter
        self.step_count = 0
        
        # Statistics
        self.curiosity_rewards: List[float] = []
        self.external_rewards: List[float] = []
        self.model_losses: List[float] = []
        self.confidence_losses: List[float] = []
        self.q_losses: List[float] = []
        
        logger.info(
            f"DNQCuriousAgent initialized: state_dim={state_dim}, "
            f"num_actions={num_actions}, device={self.device}"
        )
        logger.info(f"World Model: {self.world_model}")
        logger.info(f"Confidence Net: {self.confidence_net}")
        logger.info(f"Q-Network: {self.q_network}")
    
    def _create_target_network(self) -> nn.Module:
        """Create target network as a copy of Q-network."""
        target = copy.deepcopy(self.q_network).to(self.device)
        target.eval()
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        return target
    
    def _update_target_network(self) -> None:
        """Soft update target network."""
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
        """Convert action to one-hot tensor."""
        action_one_hot = torch.zeros(self.num_actions).to(self.device)
        action_one_hot[action] = 1.0
        return action_one_hot
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
        
        Returns:
            Selected action index
        """
        if np.random.random() < self.epsilon:
            # Random exploration
            action = np.random.randint(0, self.num_actions)
            logger.debug(f"Random action: {action}")
        else:
            # Greedy action selection
            state_tensor = self._encode_state(state)
            action = self.q_network.get_action(state_tensor, epsilon=0.0)
            logger.debug(f"Greedy action: {action}")
        
        return action
    
    def get_confidence_before(
        self, state: np.ndarray, action: int
    ) -> float:
        """
        Get C's predicted error BEFORE taking action.
        
        Args:
            state: Current state
            action: Action to take
        
        Returns:
            Predicted error
        """
        state_tensor = self._encode_state(state)
        return self.confidence_net.predict(state_tensor, action)
    
    def update_model(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """
        Update world model M on a transition.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Actual next state
        
        Returns:
            Model loss
        """
        state_tensor = self._encode_state(state).unsqueeze(0)
        action_tensor = self._encode_action(action).unsqueeze(0)
        next_state_tensor = self._encode_state(next_state).unsqueeze(0)
        
        # Update model
        loss = self.world_model.update(state_tensor, action_tensor, next_state_tensor)
        self.model_losses.append(loss)
        
        return loss
    
    def update_confidence(
        self,
        state: np.ndarray,
        action: int,
        actual_error: float,
    ) -> float:
        """
        Update confidence network C.
        
        Args:
            state: Current state
            action: Action taken
            actual_error: Actual error from model update
        
        Returns:
            Confidence loss
        """
        state_tensor = self._encode_state(state).unsqueeze(0)
        action_tensor = self._encode_action(action).unsqueeze(0)
        actual_error_tensor = torch.tensor([[actual_error]]).to(self.device)
        
        # Update confidence
        loss = self.confidence_net.update(state_tensor, action_tensor, actual_error_tensor)
        self.confidence_losses.append(loss)
        
        return loss
    
    def compute_curiosity_reward(
        self,
        o_c_before: float,
        o_c_after: float,
    ) -> float:
        """
        Compute curiosity reward.
        
        r_curiosity = o_C_before - o_C_after
        
        Args:
            o_c_before: C's prediction before model update
            o_c_after: C's prediction after model update
        
        Returns:
            Curiosity reward
        """
        raw_curiosity = o_c_before - o_c_after
        if not self.normalize_curiosity:
            return raw_curiosity

        self._running_intrinsic = (
            0.99 * self._running_intrinsic + 0.01 * abs(raw_curiosity)
        )
        normalized = (
            raw_curiosity / (self._running_intrinsic + 1e-8)
        ) * self.target_intrinsic_scale
        return normalized
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Total reward (external + curiosity)
            next_state: Next state
            done: Whether episode is done
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_q_network(self) -> Optional[float]:
        """
        Update Q-network using mini-batch from replay buffer.
        
        Returns:
            Q-network loss, or None if not ready
        """
        if not self.replay_buffer.is_ready(self.min_buffer_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size, device=self.device
        )
        
        # Update Q-network
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
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self._update_target_network()
            logger.debug("Target network updated")
        
        return loss
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        logger.debug(f"Epsilon decayed to: {self.epsilon:.4f}")
    
    def get_statistics(self) -> Dict:
        """Get agent statistics."""
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
            "avg_model_loss": (
                np.mean(self.model_losses[-100:]) if self.model_losses else 0.0
            ),
            "avg_confidence_loss": (
                np.mean(self.confidence_losses[-100:])
                if self.confidence_losses
                else 0.0
            ),
            "avg_q_loss": (
                np.mean(self.q_losses[-100:]) if self.q_losses else 0.0
            ),
        }
    
    def reset_statistics(self) -> None:
        """Reset episode statistics."""
        self.curiosity_rewards = []
        self.external_rewards = []
    
    def save(self, path: str) -> None:
        """Save all model checkpoints."""
        torch.save(
            {
                "world_model": self.world_model.state_dict(),
                "confidence_net": self.confidence_net.state_dict(),
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            },
            path,
        )
        logger.info(f"DNQCuriousAgent saved to {path}")
    
    def load(self, path: str) -> None:
        """Load all model checkpoints."""
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.confidence_net.load_state_dict(checkpoint["confidence_net"])
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.epsilon = checkpoint["epsilon"]
        self.step_count = checkpoint["step_count"]
        logger.info(f"DNQCuriousAgent loaded from {path}")
    
    def __repr__(self) -> str:
        return (
            f"DNQCuriousAgent("
            f"state_dim={self.state_dim}, "
            f"num_actions={self.num_actions}, "
            f"epsilon={self.epsilon:.3f}, "
            f"device={self.device})"
        )
