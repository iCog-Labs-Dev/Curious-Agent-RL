"""
Tests for the Intrinsic Curiosity Module (ICM) — Pathak et al. 2017
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curious_agent.models.icm import (
    FeatureEncoder,
    ForwardModel,
    InverseModel,
    IntrinsicCuriosityModule,
)
from curious_agent.agents.dqn_icm import ICMCuriousAgent


class TestFeatureEncoder:
    """Test cases for FeatureEncoder."""

    def test_initialization(self):
        """Test model initialization."""
        model = FeatureEncoder(state_dim=2, feature_dim=32, hidden_dims=[64])

        assert model.state_dim == 2
        assert model.feature_dim == 32

    def test_forward_pass(self):
        """Test forward pass."""
        model = FeatureEncoder(state_dim=2, feature_dim=32)

        state = torch.randn(1, 2)
        output = model(state)

        assert output.shape == (1, 32)

    def test_feature_dim_output(self):
        """Test that the output dimension matches feature_dim."""
        model = FeatureEncoder(state_dim=2, feature_dim=16)
        state = torch.randn(5, 2)
        assert model(state).shape == (5, 16)


class TestInverseModel:
    """Test cases for InverseModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = InverseModel(feature_dim=32, num_actions=4, hidden_dims=[64])

        assert model.feature_dim == 32
        assert model.num_actions == 4

    def test_forward_pass(self):
        """Test forward pass."""
        model = InverseModel(feature_dim=32, num_actions=4)

        features_current = torch.randn(1, 32)
        features_next = torch.randn(1, 32)

        output = model(features_current, features_next)

        assert output.shape == (1, 4)

    def test_predict_action(self):
        """Test single action prediction."""
        model = InverseModel(feature_dim=32, num_actions=4)

        features_current = torch.randn(1, 32)
        features_next = torch.randn(1, 32)

        action = model.predict_action(features_current, features_next)

        assert 0 <= action < 4


class TestForwardModel:
    """Test cases for ForwardModel."""

    def test_initialization(self):
        """Test model initialization."""
        model = ForwardModel(feature_dim=32, num_actions=4, hidden_dims=[64])

        assert model.feature_dim == 32
        assert model.num_actions == 4

    def test_forward_pass(self):
        """Test forward pass."""
        model = ForwardModel(feature_dim=32, num_actions=4)

        features_current = torch.randn(1, 32)
        action = torch.zeros(1, 4)
        action[0, 0] = 1.0

        output = model(features_current, action)

        assert output.shape == (1, 32)


class TestIntrinsicCuriosityModule:
    """Test cases for the full ICM module."""

    def test_initialization(self):
        """Test module initialization."""
        icm = IntrinsicCuriosityModule(
            state_dim=2,
            num_actions=4,
            feature_dim=32,
            hidden_dims=[64],
            learning_rate=0.001,
        )

        assert icm.state_dim == 2
        assert icm.num_actions == 4
        assert icm.feature_dim == 32
        assert hasattr(icm, "encoder")
        assert hasattr(icm, "inverse_model")
        assert hasattr(icm, "forward_model")
        assert hasattr(icm, "optimizer")

    def test_intrinsic_reward_non_negative(self):
        """Test that the curiosity reward is non-negative."""
        icm = IntrinsicCuriosityModule(state_dim=2, num_actions=4)

        state = np.array([0.0, 0.0], dtype=np.float32)
        next_state = np.array([1.0, 0.0], dtype=np.float32)

        reward = icm.intrinsic_reward(state, 0, next_state)

        assert reward >= 0.0

    def test_intrinsic_reward_formula(self):
        """Test the reward equals (eta/2) * forward prediction error."""
        icm = IntrinsicCuriosityModule(
            state_dim=2, num_actions=4, feature_dim=16, eta=1.0
        )

        state = np.array([0.0, 0.0], dtype=np.float32)
        next_state = np.array([1.0, 0.0], dtype=np.float32)

        reward = icm.intrinsic_reward(state, 0, next_state)
        error = icm.compute_forward_error(state, 0, next_state)

        assert reward == pytest.approx(0.5 * error, rel=1e-3)

    def test_update_returns_losses(self):
        """Test that update returns (inverse_loss, forward_loss)."""
        icm = IntrinsicCuriosityModule(state_dim=2, num_actions=4, learning_rate=0.01)

        state = np.array([0.0, 0.0], dtype=np.float32)
        next_state = np.array([1.0, 0.0], dtype=np.float32)

        inverse_loss, forward_loss = icm.update(state, 0, next_state)

        assert isinstance(inverse_loss, float)
        assert isinstance(forward_loss, float)
        assert inverse_loss >= 0.0
        assert forward_loss >= 0.0

    def test_forward_error_decreases_with_training(self):
        """Boredom analog: forward error drops after repeated updates on a
        deterministic transition, so the curiosity reward shrinks."""
        icm = IntrinsicCuriosityModule(
            state_dim=2,
            num_actions=4,
            feature_dim=16,
            hidden_dims=[32],
            learning_rate=0.01,
        )

        state = np.array([0.0, 0.0], dtype=np.float32)
        next_state = np.array([1.0, 0.0], dtype=np.float32)

        # Evaluate the forward model on a copy before training begins.
        error_before = icm.compute_forward_error(state, 0, next_state)

        for _ in range(200):
            icm.update(state, 0, next_state)

        error_after = icm.compute_forward_error(state, 0, next_state)

        # The model should now predict this deterministic transition better.
        assert error_after < error_before

    def test_inverse_model_learns_action(self):
        """The inverse model should learn to predict the action that connects
        two states once the ICM is trained on a consistent transition."""
        icm = IntrinsicCuriosityModule(
            state_dim=2,
            num_actions=4,
            feature_dim=16,
            hidden_dims=[32],
            learning_rate=0.01,
        )

        state = np.array([0.0, 0.0], dtype=np.float32)
        next_state = np.array([1.0, 0.0], dtype=np.float32)
        action = 1  # DOWN in the GridWorld action space.

        for _ in range(200):
            icm.update(state, action, next_state)

        # Predict the action from the learned feature space.
        icm.eval()
        with torch.no_grad():
            features_current = icm.encoder(
                icm._encode_state(state)
            )
            features_next = icm.encoder(
                icm._encode_state(next_state)
            )
            predicted = icm.inverse_model.predict_action(
                features_current, features_next
            )

        assert predicted == action


class TestICMCuriousAgent:
    """Test cases for the ICM DQN agent."""

    def test_has_icm_but_no_world_model(self):
        """The ICM agent must have the ICM module and no Schmidhuber modules."""
        agent = ICMCuriousAgent(
            state_dim=2,
            num_actions=4,
            config={"networks": {"q_network": {"hidden_dims": [8]}}},
            device=torch.device("cpu"),
        )

        assert hasattr(agent, "icm")
        assert not hasattr(agent, "world_model")
        assert not hasattr(agent, "confidence_net")

    def test_select_action(self):
        """Test action selection."""
        agent = ICMCuriousAgent(
            state_dim=2,
            num_actions=4,
            config={"agent": {"epsilon": 0.1}},
            device=torch.device("cpu"),
        )

        state = np.array([0.0, 0.0], dtype=np.float32)
        action = agent.select_action(state)

        assert 0 <= action < 4

    def test_full_transition_cycle(self):
        """Test the full per-step pipeline: reward, ICM update, Q update."""
        agent = ICMCuriousAgent(
            state_dim=2,
            num_actions=4,
            config={
                "agent": {"epsilon": 0.1, "beta": 1.0},
                "networks": {"q_network": {"hidden_dims": [8]}},
                "training": {
                    "batch_size": 4,
                    "buffer_size": 1000,
                    "min_buffer_size": 4,
                    "target_update_frequency": 50,
                },
            },
            device=torch.device("cpu"),
        )

        state = np.array([0.0, 0.0], dtype=np.float32)
        action = agent.select_action(state)
        next_state = np.array([1.0, 0.0], dtype=np.float32)

        reward = agent.compute_intrinsic_reward(state, action, next_state)
        agent.update_icm(state, action, next_state)

        r_total = 0.5 + agent.beta * reward
        agent.store_experience(state, action, r_total, next_state, done=False)

        # Seed the buffer so the Q update can sample.
        for _ in range(8):
            agent.store_experience(state, action, r_total, next_state, done=False)

        loss = agent.update_q_network()

        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_checkpoint_round_trip(self, tmp_path):
        """Save and load the agent checkpoint."""
        agent = ICMCuriousAgent(
            state_dim=2,
            num_actions=4,
            config={"networks": {"q_network": {"hidden_dims": [8]}}},
            device=torch.device("cpu"),
        )

        checkpoint_path = tmp_path / "icm.pt"
        agent.save(str(checkpoint_path))

        loaded = ICMCuriousAgent(
            state_dim=2,
            num_actions=4,
            config={"networks": {"q_network": {"hidden_dims": [8]}}},
            device=torch.device("cpu"),
        )
        loaded.load(str(checkpoint_path))

        assert loaded.epsilon == agent.epsilon
        assert loaded.step_count == agent.step_count

    def test_curiosity_normalization_on_by_default(self):
        """Normalization must be enabled by default and scale the raw signal."""
        agent = ICMCuriousAgent(
            state_dim=2,
            num_actions=4,
            config={
                "agent": {"target_intrinsic_scale": 1.0},
                "networks": {"q_network": {"hidden_dims": [8]}},
            },
            device=torch.device("cpu"),
        )

        assert agent.normalize_curiosity is True
        assert agent.target_intrinsic_scale == 1.0

        # Raw ICM curiosity is unbounded; a large raw reward must be scaled
        # toward the target scale once the running mean adapts.
        state = np.array([0.0, 0.0], dtype=np.float32)
        next_state = np.array([1.0, 0.0], dtype=np.float32)
        action = 0

        raw = agent.icm.intrinsic_reward(state, action, next_state)
        agent._running_intrinsic = raw  # simulate an adapted running mean
        normalized = agent.compute_intrinsic_reward(state, action, next_state)

        assert normalized == pytest.approx(agent.target_intrinsic_scale, rel=1e-2)

    def test_curiosity_normalization_can_be_disabled(self):
        """Disabling normalization must return the raw curiosity signal."""
        agent = ICMCuriousAgent(
            state_dim=2,
            num_actions=4,
            config={
                "agent": {"normalize_curiosity": False},
                "networks": {"q_network": {"hidden_dims": [8]}},
            },
            device=torch.device("cpu"),
        )

        assert agent.normalize_curiosity is False
        state = np.array([0.0, 0.0], dtype=np.float32)
        next_state = np.array([1.0, 0.0], dtype=np.float32)

        reward = agent.compute_intrinsic_reward(state, 0, next_state)
        raw = agent.icm.compute_forward_error(state, 0, next_state)
        assert reward == pytest.approx(0.5 * raw, rel=1e-3)
        # The running mean must not be updated when disabled.
        assert agent._running_intrinsic == 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])