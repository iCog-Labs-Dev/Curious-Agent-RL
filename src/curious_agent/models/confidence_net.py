"""
Confidence Network (C) Predicts Model Error

This network learns to predict how much error the world model (M) will make
for a given state-action pair. The key insight is that curiosity reward is
based on improvement in prediction ability:

r_curiosity = o_C_before - o_C_after

When C's error prediction drops after M updates, the agent gets positive
reward (it learned something!).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class ConfidenceNetwork(nn.Module):
    """
    Neural network that predicts the world model's error.
    
    Architecture:
    - Input: state (2D) + action one-hot (4D) = 6D
    - Hidden: MLP with configurable layers
    - Output: predicted error (scalar)
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        num_actions: int = 4,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.0005,
    ):
        """
        Initialize the confidence network.
        
        Args:
            state_dim: Dimension of state space
            num_actions: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate (slower than world model)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Input dimension: state + one-hot action
        input_dim = state_dim + num_actions
        
        # Default hidden dimensions (smaller than world model)
        if hidden_dims is None:
            hidden_dims = [32, 32]
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output in [0, 1] range
        
        self.network = nn.Sequential(*layers)
        
        # Optimizer (slower learning rate)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Statistics
        self.training_losses: List[float] = []
        self.predictions: List[float] = []
        
        logger.info(
            f"ConfidenceNetwork initialized: input={input_dim}, "
            f"hidden={hidden_dims}, output=1"
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action one-hot tensor [batch_size, num_actions]
        
        Returns:
            Predicted error [batch_size, 1]
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Forward pass
        predicted_error = self.network(x)
        
        return predicted_error
    
    def predict(self, state: torch.Tensor, action: int) -> float:
        """
        Predict error for a single state-action pair.
        
        Args:
            state: State tensor [state_dim]
            action: Action index
        
        Returns:
            Predicted error (scalar)
        """
        # Add batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Create one-hot action on the same device as the state
        action_one_hot = torch.zeros(
            1, self.num_actions, device=state.device
        )
        action_one_hot[0, action] = 1.0
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            predicted_error = self.forward(state, action_one_hot)
        
        error_value = predicted_error.item()
        self.predictions.append(error_value)
        
        return error_value
    
    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        actual_error: torch.Tensor,
    ) -> float:
        """
        Update the model on actual error.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action one-hot [batch_size, num_actions]
            actual_error: Actual error from world model [batch_size, 1]
        
        Returns:
            Loss value
        """
        self.train()
        
        # Forward pass
        predicted_error = self.forward(state, action)
        
        # Compute loss
        loss = self.loss_fn(predicted_error, actual_error)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        logger.debug(f"ConfidenceNetwork update: loss={loss_value:.6f}")
        
        return loss_value
    
    def get_prediction_before_and_after(
        self,
        state: torch.Tensor,
        action: int,
        update_fn,
    ) -> Tuple[float, float]:
        """
        Get predictions before and after an update.
        
        This is the key method for computing curiosity reward.
        
        Args:
            state: State tensor
            action: Action index
            update_fn: Function that performs the update
        
        Returns:
            Tuple of (prediction_before, prediction_after)
        """
        # Get prediction before
        pred_before = self.predict(state, action)
        
        # Perform update
        update_fn()
        
        # Get prediction after (same input!)
        pred_after = self.predict(state, action)
        
        return pred_before, pred_after
    
    def get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'predictions': self.predictions,
        }, path)
        logger.info(f"ConfidenceNetwork saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.get_device())
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        self.predictions = checkpoint.get('predictions', [])
        logger.info(f"ConfidenceNetwork loaded from {path}")
    
    def __repr__(self) -> str:
        return (
            f"ConfidenceNetwork("
            f"state_dim={self.state_dim}, "
            f"num_actions={self.num_actions}, "
            f"params={sum(p.numel() for p in self.parameters())})"
        )