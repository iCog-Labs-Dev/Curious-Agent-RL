"""
World Model (M) Network

Predicts next state from current state and action.
This is the agent's internal model of the environment dynamics.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class WorldModel(nn.Module):
    """
    Neural network that predicts next state from (state, action).
    
    Architecture:
    - Input: state (2D) + action one-hot (4D) = 6D
    - Hidden: MLP with configurable layers
    - Output: predicted next state (2D)
    """
    
    def __init__(
        self,
        state_dim: int = 2,
        num_actions: int = 4,
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.001,
    ):
        """
        Initialize the world model.
        
        Args:
            state_dim: Dimension of state space (default: 2 for row, col)
            num_actions: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Input dimension: state + one-hot action
        input_dim = state_dim + num_actions
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [64, 64]
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, state_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Statistics
        self.training_losses: List[float] = []
        
        logger.info(
            f"WorldModel initialized: input={input_dim}, "
            f"hidden={hidden_dims}, output={state_dim}"
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
            action: Action one-hot tensor [batch_size, num_actions]
        
        Returns:
            Predicted next state [batch_size, state_dim]
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)
        
        # Forward pass
        predicted_next_state = self.network(x)
        
        return predicted_next_state
    
    def predict(self, state: torch.Tensor, action: int) -> torch.Tensor:
        """
        Predict next state for a single state-action pair.
        
        Args:
            state: State tensor [state_dim]
            action: Action index
        
        Returns:
            Predicted next state [state_dim]
        """
        # Add batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Create one-hot action on the same device as state
        action_one_hot = torch.zeros(1, self.num_actions, device=state.device)
        action_one_hot[0, action] = 1.0
        
        # Forward pass
        with torch.no_grad():
            predicted = self.forward(state, action_one_hot)
        
        return predicted.squeeze(0)
    
    def update(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> float:
        """
        Update the model on a transition.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action one-hot [batch_size, num_actions]
            next_state: Actual next state [batch_size, state_dim]
        
        Returns:
            Loss value
        """
        self.train()
        
        # Forward pass
        predicted_next_state = self.forward(state, action)
        
        # Compute loss
        loss = self.loss_fn(predicted_next_state, next_state)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        logger.debug(f"WorldModel update: loss={loss_value:.6f}")
        
        return loss_value
    
    def compute_error(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> float:
        """
        Compute prediction error without updating.
        
        Args:
            state: Current state
            action: Action one-hot
            next_state: Actual next state
        
        Returns:
            MSE error
        """
        self.eval()
        
        with torch.no_grad():
            predicted_next_state = self.forward(state, action)
            error = self.loss_fn(predicted_next_state, next_state)
        
        return error.item()
    
    def get_device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
        }, path)
        logger.info(f"WorldModel saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.get_device())
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        logger.info(f"WorldModel loaded from {path}")
    
    def __repr__(self) -> str:
        return (
            f"WorldModel("
            f"state_dim={self.state_dim}, "
            f"num_actions={self.num_actions}, "
            f"params={sum(p.numel() for p in self.parameters())})"
        )