"""ICM implementation."""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class FeatureEncoder(nn.Module):
    """Feature encoder."""

    def __init__(
        self,
        state_dim: int = 2,
        feature_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.feature_dim = feature_dim

        if hidden_dims is None:
            hidden_dims = [64]

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, feature_dim))

        self.network = nn.Sequential(*layers)

        logger.info(
            f"FeatureEncoder initialized: input={state_dim}, "
            f"hidden={hidden_dims}, output={feature_dim}"
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Map states to feature vectors."""
        return self.network(state)

    def __repr__(self) -> str:
        return (
            f"FeatureEncoder(state_dim={self.state_dim}, "
            f"feature_dim={self.feature_dim}, "
            f"params={sum(p.numel() for p in self.parameters())})"
        )


class InverseModel(nn.Module):
    """Inverse dynamics model."""

    def __init__(
        self,
        feature_dim: int = 32,
        num_actions: int = 4,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions

        if hidden_dims is None:
            hidden_dims = [64]

        input_dim = 2 * feature_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_actions))

        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

        logger.info(
            f"InverseModel initialized: input={input_dim}, "
            f"hidden={hidden_dims}, output={num_actions}"
        )

    def forward(self, features_current: torch.Tensor, features_next: torch.Tensor) -> torch.Tensor:
        """Predict action logits from two feature vectors."""
        x = torch.cat([features_current, features_next], dim=-1)
        return self.network(x)

    def predict_action(self, features_current: torch.Tensor, features_next: torch.Tensor) -> int:
        """Predict one action index."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(features_current, features_next)
        return int(torch.argmax(logits, dim=-1).item())

    def __repr__(self) -> str:
        return (
            f"InverseModel(feature_dim={self.feature_dim}, "
            f"num_actions={self.num_actions}, "
            f"params={sum(p.numel() for p in self.parameters())})"
        )


class ForwardModel(nn.Module):
    """Forward dynamics model."""

    def __init__(
        self,
        feature_dim: int = 32,
        num_actions: int = 4,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_actions = num_actions

        if hidden_dims is None:
            hidden_dims = [64]

        input_dim = feature_dim + num_actions
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, feature_dim))

        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

        logger.info(
            f"ForwardModel initialized: input={input_dim}, "
            f"hidden={hidden_dims}, output={feature_dim}"
        )

    def forward(self, features_current: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict the next feature vector."""
        x = torch.cat([features_current, action], dim=-1)
        return self.network(x)

    def __repr__(self) -> str:
        return (
            f"ForwardModel(feature_dim={self.feature_dim}, "
            f"num_actions={self.num_actions}, "
            f"params={sum(p.numel() for p in self.parameters())})"
        )


class IntrinsicCuriosityModule(nn.Module):
    """Combined ICM module."""

    def __init__(
        self,
        state_dim: int = 2,
        num_actions: int = 4,
        feature_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        eta: float = 1.0,
        inverse_weight: float = 0.2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.eta = eta
        self.inverse_weight = inverse_weight

        if hidden_dims is None:
            hidden_dims = [64]

        self.encoder = FeatureEncoder(state_dim, feature_dim, hidden_dims)
        self.inverse_model = InverseModel(feature_dim, num_actions, hidden_dims)
        self.forward_model = ForwardModel(feature_dim, num_actions, hidden_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.inverse_losses: List[float] = []
        self.forward_losses: List[float] = []
        self.intrinsic_rewards: List[float] = []

        logger.info(
            f"IntrinsicCuriosityModule initialized: state_dim={state_dim}, "
            f"feature_dim={feature_dim}, eta={eta}, "
            f"inverse_weight={inverse_weight}"
        )

    def features(self, state: torch.Tensor) -> torch.Tensor:
        """Encode a state into features."""
        return self.encoder(state)

    def _encode_state(self, state: np.ndarray) -> torch.Tensor:
        """Convert numpy state data to a tensor."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return torch.FloatTensor(state).to(self.get_device())

    def _one_hot(self, action: int) -> torch.Tensor:
        """Return a one-hot action tensor."""
        one_hot = torch.zeros(1, self.num_actions, device=self.get_device())
        one_hot[0, action] = 1.0
        return one_hot

    def intrinsic_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """Compute the curiosity reward for one transition."""
        self.eval()
        with torch.no_grad():
            state_t = self._encode_state(state)
            next_state_t = self._encode_state(next_state)
            action_t = self._one_hot(action)

            phi_current = self.encoder(state_t)
            phi_next = self.encoder(next_state_t)
            phi_hat_next = self.forward_model(phi_current, action_t)

            error = torch.mean((phi_hat_next - phi_next) ** 2)
            reward = (self.eta / 2.0) * error.item()

        self.intrinsic_rewards.append(reward)
        return reward

    def update(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> Tuple[float, float]:
        """Update the ICM models on one transition."""
        self.train()

        state_t = self._encode_state(state)
        next_state_t = self._encode_state(next_state)
        action_t = self._one_hot(action)
        action_idx = torch.tensor([action], device=self.get_device())

        phi_current = self.encoder(state_t)
        phi_next = self.encoder(next_state_t)

        action_logits = self.inverse_model(phi_current, phi_next)
        inverse_loss = self.inverse_model.loss_fn(action_logits, action_idx)

        phi_hat_next = self.forward_model(phi_current, action_t)
        forward_loss = self.forward_model.loss_fn(phi_hat_next, phi_next)

        total_loss = (1.0 - self.inverse_weight) * inverse_loss + self.inverse_weight * forward_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        inverse_value = float(inverse_loss.item())
        forward_value = float(forward_loss.item())
        self.inverse_losses.append(inverse_value)
        self.forward_losses.append(forward_value)

        logger.debug(
            f"ICM update: inverse_loss={inverse_value:.6f}, "
            f"forward_loss={forward_value:.6f}"
        )

        return inverse_value, forward_value

    def intrinsic_reward_batch(
        self,
        states_t: torch.Tensor,
        actions_oh: torch.Tensor,
        next_states_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute curiosity rewards for a batch."""
        self.eval()
        with torch.no_grad():
            phi_current = self.encoder(states_t)
            phi_next = self.encoder(next_states_t)
            phi_hat_next = self.forward_model(phi_current, actions_oh)
            per_sample_error = ((phi_hat_next - phi_next) ** 2).mean(dim=1)
            rewards = (self.eta / 2.0) * per_sample_error
        return rewards

    def update_batch(
        self,
        states_t: torch.Tensor,
        actions_oh: torch.Tensor,
        actions_idx: torch.Tensor,
        next_states_t: torch.Tensor,
    ) -> tuple[float, float]:
        """Update the ICM models on a batch."""
        self.train()

        phi_current = self.encoder(states_t)
        phi_next = self.encoder(next_states_t)

        action_logits = self.inverse_model(phi_current, phi_next)
        inverse_loss = self.inverse_model.loss_fn(action_logits, actions_idx)

        phi_hat_next = self.forward_model(phi_current, actions_oh)
        forward_loss = self.forward_model.loss_fn(phi_hat_next, phi_next)

        total_loss = (1.0 - self.inverse_weight) * inverse_loss + self.inverse_weight * forward_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        inv_val = float(inverse_loss.item())
        fwd_val = float(forward_loss.item())
        self.inverse_losses.append(inv_val)
        self.forward_losses.append(fwd_val)
        return inv_val, fwd_val

    def compute_forward_error(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
    ) -> float:
        """Return the raw forward-prediction error."""
        self.eval()
        with torch.no_grad():
            state_t = self._encode_state(state)
            next_state_t = self._encode_state(next_state)
            action_t = self._one_hot(action)

            phi_current = self.encoder(state_t)
            phi_next = self.encoder(next_state_t)
            phi_hat_next = self.forward_model(phi_current, action_t)

            error = float(torch.mean((phi_hat_next - phi_next) ** 2).item())
        return error

    def get_device(self) -> torch.device:
        """Get the model device."""
        return next(self.parameters()).device

    def save(self, path: str) -> None:
        """Persist the ICM model state."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "inverse_losses": self.inverse_losses,
                "forward_losses": self.forward_losses,
            },
            path,
        )
        logger.info(f"IntrinsicCuriosityModule saved to {path}")

    def load(self, path: str) -> None:
        """Load the ICM model state."""
        checkpoint = torch.load(path, map_location=self.get_device())
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.inverse_losses = checkpoint.get("inverse_losses", [])
        self.forward_losses = checkpoint.get("forward_losses", [])
        logger.info(f"IntrinsicCuriosityModule loaded from {path}")

    def __repr__(self) -> str:
        return (
            f"IntrinsicCuriosityModule(state_dim={self.state_dim}, "
            f"feature_dim={self.feature_dim}, "
            f"params={sum(p.numel() for p in self.parameters())})"
        )
