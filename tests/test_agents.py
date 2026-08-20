"""
Tests for Curious Agents
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curious_agent.agents.tabular_curious import TabularCuriousAgent
from curious_agent.agents.dqn_curious import DNQCuriousAgent
from curious_agent.agents.dqn import DQNAgent


class TestTabularCuriousAgent:
    """Test cases for TabularCuriousAgent."""
    
    def test_initialization(self):
        """Test agent initialization."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"alpha_m": 0.2, "alpha_c": 0.05}},
        )
        
        assert agent.num_states == 100
        assert agent.num_actions == 4
        assert agent.M.shape == (100, 4)
        assert agent.C.shape == (100, 4)
        assert agent.Q.shape == (100, 4)
    
    def test_action_selection(self):
        """Test action selection."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"epsilon": 0.0}},
        )
        
        action = agent.select_action(0)
        assert 0 <= action < 4
    
    def test_confidence_before(self):
        """Test getting confidence prediction."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
        )
        
        pred = agent.get_confidence_before(0, 0)
        assert isinstance(pred, float)
        assert 0 <= pred <= 1
    
    def test_model_update(self):
        """Test model update."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
        )
        
        # Initial prediction is 0
        assert agent.M[0, 0] == 0
        
        # Update model
        error = agent.update_model(0, 0, 5)
        
        # Should have error (predicted 0, actual 5)
        assert error == 1.0
        # Model should be updated
        assert agent.M[0, 0] == 5
        
        # Update again with same transition
        error = agent.update_model(0, 0, 5)
        
        # Should have no error (predicted 5, actual 5)
        assert error == 0.0
    
    def test_confidence_update(self):
        """Test confidence update."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"alpha_c": 0.1}},
        )
        
        # Initial prediction
        initial_pred = agent.C[0, 0]
        
        # Update with error
        new_pred = agent.update_confidence(0, 0, 1.0)
        
        # Should move towards actual error
        assert new_pred != initial_pred
        assert 0 <= new_pred <= 1
    
    def test_curiosity_reward(self):
        """Test curiosity reward calculation."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
        )
        
        # Positive reward when prediction improves
        r = agent.compute_curiosity_reward(0.5, 0.3)
        assert r == 0.2
        
        # Negative reward when prediction worsens
        r = agent.compute_curiosity_reward(0.3, 0.5)
        assert r == -0.2
        
        # Zero reward when no change
        r = agent.compute_curiosity_reward(0.5, 0.5)
        assert r == 0.0
    
    def test_q_update(self):
        """Test Q-learning update."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"alpha_q": 0.1, "gamma": 0.99}},
        )
        
        # Initial Q-value
        initial_q = agent.Q[0, 0]
        
        # Update Q
        td_error = agent.update_q(0, 0, 1.0, 5, done=False)
        
        # Q-value should change
        assert agent.Q[0, 0] != initial_q
    
    def test_epsilon_decay(self):
        """Test epsilon decay."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"epsilon": 1.0, "epsilon_min": 0.01, "epsilon_decay": 0.99}},
        )
        
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        
        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= 0.01
    
    def test_action_probs(self):
        """Test action probability calculation."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"epsilon": 0.1}},
        )
        
        probs = agent.get_action_probs(0)
        
        assert len(probs) == 4
        assert abs(sum(probs) - 1.0) < 1e-6
    
    def test_statistics(self):
        """Test statistics collection."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
        )
        
        stats = agent.get_statistics()
        
        assert "epsilon" in stats
        assert "avg_curiosity_reward" in stats
        assert "avg_external_reward" in stats


class TestCuriosityMechanism:
    """Test the core curiosity mechanism."""
    
    def test_curiosity_drives_exploration(self):
        """Test that curiosity reward encourages exploration."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"beta": 1.0}},
        )
        
        # Simulate learning a new transition
        state_idx = 0
        action = 0
        next_state_idx = 5
        
        # The first observation reveals that the model is wrong.
        initial_error = agent.update_model(state_idx, action, next_state_idx)
        agent.update_confidence(state_idx, action, initial_error)

        # On the next observation, the learned transition is predicted correctly.
        o_c_before = agent.get_confidence_before(state_idx, action)
        improved_error = agent.update_model(state_idx, action, next_state_idx)
        agent.update_confidence(state_idx, action, improved_error)
        o_c_after = agent.get_confidence_before(state_idx, action)
        
        # Compute curiosity reward
        r_curiosity = agent.compute_curiosity_reward(o_c_before, o_c_after)
        
        # Should be positive (improvement)
        assert r_curiosity > 0
    
    def test_boredom_in_known_zones(self):
        """Test that agent gets bored in known zones."""
        agent = TabularCuriousAgent(
            num_states=100,
            num_actions=4,
            config={"agent": {"beta": 1.0}},
        )
        
        state_idx = 0
        action = 0
        next_state_idx = 5
        
        # Learn the transition perfectly
        agent.update_model(state_idx, action, next_state_idx)
        agent.update_confidence(state_idx, action, 0.0)  # No error
        
        # Get confidence before (should be low)
        o_c_before = agent.get_confidence_before(state_idx, action)
        
        # Update again with same transition
        agent.update_model(state_idx, action, next_state_idx)
        agent.update_confidence(state_idx, action, 0.0)
        o_c_after = agent.get_confidence_before(state_idx, action)
        
        # Curiosity reward should be small or zero
        r_curiosity = agent.compute_curiosity_reward(o_c_before, o_c_after)
        assert r_curiosity <= 0.01


class TestDNQCuriousAgent:
    """Test cases for the neural curious agent."""

    def test_dueling_target_network_matches_online_architecture(self):
        """The target must preserve every configured hidden layer."""
        agent = DNQCuriousAgent(
            state_dim=2,
            num_actions=4,
            config={
                "networks": {
                    "q_network": {
                        "use_dueling": True,
                        "hidden_dims": [16, 8],
                        "value_hidden_dim": 4,
                        "advantage_hidden_dim": 6,
                    }
                }
            },
            device=torch.device("cpu"),
        )

        online_parameters = dict(agent.q_network.named_parameters())
        target_parameters = dict(agent.target_q_network.named_parameters())

        assert online_parameters.keys() == target_parameters.keys()
        for name, online_parameter in online_parameters.items():
            assert torch.equal(online_parameter, target_parameters[name])
            assert not target_parameters[name].requires_grad

    def test_curiosity_normalization_on_by_default(self):
        """Normalization must be enabled by default and scale the raw signal."""
        agent = DNQCuriousAgent(
            state_dim=2,
            num_actions=4,
            config={"agent": {"target_intrinsic_scale": 1.0}},
            device=torch.device("cpu"),
        )

        assert agent.normalize_curiosity is True
        assert agent.target_intrinsic_scale == 1.0

        # The raw confidence change is tiny (sigmoid-bounded). Once the running
        # mean adapts to that magnitude, normalization brings the signal toward
        # the target scale.
        o_c_before = 0.60
        o_c_after = 0.55  # raw curiosity = 0.05
        agent._running_intrinsic = 0.05  # adapted running mean
        normalized = agent.compute_curiosity_reward(o_c_before, o_c_after)

        assert normalized == pytest.approx(
            agent.target_intrinsic_scale * (o_c_before - o_c_after) / 0.05,
            rel=1e-3,
        )
        assert normalized == pytest.approx(1.0, rel=1e-3)

    def test_curiosity_normalization_can_be_disabled(self):
        """Disabling normalization must return the raw curiosity signal."""
        agent = DNQCuriousAgent(
            state_dim=2,
            num_actions=4,
            config={"agent": {"normalize_curiosity": False}},
            device=torch.device("cpu"),
        )

        assert agent.normalize_curiosity is False
        reward = agent.compute_curiosity_reward(0.6, 0.55)
        assert reward == pytest.approx(0.05)
        # The running mean must not be updated when disabled.
        assert agent._running_intrinsic == 1e-3


class TestVanillaDQNAgent:
    """Tests for the external-reward-only DQN ablation."""

    def test_has_no_curiosity_components(self):
        agent = DQNAgent(
            config={"networks": {"q_network": {"hidden_dims": [8]}}},
            device=torch.device("cpu"),
        )

        assert not hasattr(agent, "world_model")
        assert not hasattr(agent, "confidence_net")
        assert not hasattr(agent, "beta")

    def test_stores_external_reward_without_shaping(self):
        agent = DQNAgent(device=torch.device("cpu"))
        state = np.array([0.0, 0.0], dtype=np.float32)
        next_state = np.array([0.0, 1.0], dtype=np.float32)

        agent.store_experience(state, 3, 0.75, next_state, False)

        assert agent.replay_buffer.buffer[0].reward == 0.75

    def test_bootstraps_from_target_network(self):
        agent = DQNAgent(
            config={
                "agent": {"gamma": 0.5},
                "networks": {
                    "q_network": {
                        "hidden_dims": [],
                        "learning_rate": 0.001,
                        "use_dueling": False,
                    }
                },
                "training": {
                    "batch_size": 1,
                    "min_buffer_size": 1,
                    "target_update_frequency": 100,
                },
            },
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            agent.q_network.network[0].weight.zero_()
            agent.q_network.network[0].bias.zero_()
            agent.target_q_network.network[0].weight.zero_()
            agent.target_q_network.network[0].bias.fill_(1.0)

        state = np.array([0.0, 0.0], dtype=np.float32)
        agent.store_experience(state, 0, 0.0, state, False)

        assert agent.update_q_network() == pytest.approx(0.25)

    def test_checkpoint_contains_only_dqn_state(self, tmp_path):
        agent = DQNAgent(device=torch.device("cpu"))
        checkpoint_path = tmp_path / "vanilla.pt"

        agent.save(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        assert checkpoint["agent_type"] == "vanilla_dqn"
        assert "q_network" in checkpoint
        assert "world_model" not in checkpoint
        assert "confidence_net" not in checkpoint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
