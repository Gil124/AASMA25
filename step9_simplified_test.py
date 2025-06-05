#!/usr/bin/env python3
"""
Simplified RL Agent Test - Focus on Core Components
================================================

This script tests our RL components without relying on catanatron_gym,
using a simple custom environment wrapper instead.
"""

import gymnasium
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# Import our custom modules
from step3_rl_agent_architecture import MaskedDQNAgent, MaskedPPOAgent
from step4_training_utilities import TrainingLogger, CatanatronRewardShaper


class SimpleCatanatronEnv:
    """Simple wrapper that mimics the Catanatron environment structure."""
    
    def __init__(self):
        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=95.0, shape=(614,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(290)
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self, seed=None):
        self.step_count = 0
        # Create random observation
        obs = np.random.uniform(0.0, 25.0, size=(614,)).astype(np.float32)
        
        # Create action mask with some random valid actions
        action_mask = np.zeros(290, dtype=bool)
        valid_actions = np.random.choice(290, size=np.random.randint(10, 50), replace=False)
        action_mask[valid_actions] = True
        
        info = {
            'action_mask': action_mask,
            'valid_actions': valid_actions,
            'step_count': self.step_count
        }
        
        return obs, info
    
    def step(self, action):
        self.step_count += 1
        
        # Random next observation
        obs = np.random.uniform(0.0, 25.0, size=(614,)).astype(np.float32)
        
        # Random reward (slightly positive to encourage learning)
        reward = np.random.uniform(-0.1, 0.5)
        
        # Terminal condition
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        # Create new action mask
        action_mask = np.zeros(290, dtype=bool)
        valid_actions = np.random.choice(290, size=np.random.randint(10, 50), replace=False)
        action_mask[valid_actions] = True
        
        info = {
            'action_mask': action_mask,
            'valid_actions': valid_actions,
            'step_count': self.step_count
        }
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        pass


def test_environment():
    """Test our simple environment."""
    print("=== Testing Simple Catanatron Environment ===")
    
    try:
        env = SimpleCatanatronEnv()
        print(f"✓ Environment created successfully")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Valid actions: {len(info['valid_actions'])}")
        
        # Test step
        action = np.random.choice(info['valid_actions'])
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  - Reward: {reward:.3f}")
        print(f"  - Terminated: {terminated}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agents():
    """Test DQN and PPO agents."""
    print("\n=== Testing RL Agents ===")
    
    try:
        obs_dim = 614
        action_dim = 290
        
        # Test DQN agent
        dqn_agent = MaskedDQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=1e-4,
            buffer_size=1000,
            batch_size=32
        )
        print(f"✓ DQN Agent created")
        
        # Test forward pass
        obs = torch.randn(1, obs_dim)
        action_mask = torch.ones(1, action_dim, dtype=torch.bool)
        q_values = dqn_agent.q_network(obs)
        action = dqn_agent.select_action(obs, action_mask)
        print(f"✓ DQN forward pass successful (action: {action})")
        
        # Test PPO agent  
        ppo_agent = MaskedPPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=3e-4
        )
        print(f"✓ PPO Agent created")
        
        # Test forward pass
        action_probs = ppo_agent.actor(obs)
        value = ppo_agent.critic(obs)
        action = ppo_agent.select_action(obs, action_mask)
        print(f"✓ PPO forward pass successful (action: {action})")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_components():
    """Test training utilities."""
    print("\n=== Testing Training Components ===")
    
    try:
        # Test logger
        logger = TrainingLogger("test_run")
        logger.log_scalar("test_reward", 0.5, 1)
        logger.log_scalar("test_loss", 0.1, 1)
        print("✓ Training logger working")
        
        # Test reward shaper
        reward_shaper = CatanatronRewardShaper()
        print("✓ Reward shaper created")
        
        # Test some reward shaping
        game_state = {}  # Mock game state
        action = 0
        base_reward = 0.1
        shaped_reward = reward_shaper.shape_reward(game_state, action, base_reward)
        print(f"✓ Reward shaping working (shaped: {shaped_reward})")
        
        return True
        
    except Exception as e:
        print(f"✗ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training_loop():
    """Test a minimal training loop."""
    print("\n=== Testing Mini Training Loop ===")
    
    try:
        # Create environment and agent
        env = SimpleCatanatronEnv()
        agent = MaskedDQNAgent(
            obs_dim=614,
            action_dim=290,
            hidden_dim=64,
            learning_rate=1e-3,
            buffer_size=1000,
            batch_size=16
        )
        
        print("✓ Environment and agent created")
        
        # Training loop
        total_reward = 0
        episode_rewards = []
        
        for episode in range(5):  # Very short training
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(20):  # Short episodes
                # Convert to tensors
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_mask = torch.BoolTensor(info['action_mask']).unsqueeze(0)
                
                # Select action
                action = agent.select_action(obs_tensor, action_mask)
                
                # Take step
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                
                # Store experience
                agent.replay_buffer.push(
                    obs, action, reward, next_obs, terminated, info['action_mask']
                )
                
                # Train if enough samples
                if len(agent.replay_buffer) > agent.batch_size:
                    # Create dummy next action mask for training
                    next_action_mask = next_info['action_mask']
                    loss = agent.train_step(next_action_mask)
                    
                episode_reward += reward
                obs = next_obs
                info = next_info
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            total_reward += episode_reward
            print(f"  Episode {episode + 1}: reward = {episode_reward:.3f}")
        
        avg_reward = total_reward / len(episode_rewards)
        print(f"✓ Mini training completed")
        print(f"  - Average reward: {avg_reward:.3f}")
        print(f"  - Total episodes: {len(episode_rewards)}")
        print(f"  - Buffer size: {len(agent.replay_buffer)}")
        
        # Test model saving
        model_path = Path("test_model.pth")
        torch.save({
            'q_network': agent.q_network.state_dict(),
            'target_network': agent.target_network.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
        }, model_path)
        print(f"✓ Model saved to {model_path}")
        
        # Clean up
        model_path.unlink(missing_ok=True)
        env.close()
        
        return True
        
    except Exception as e:
        print(f"✗ Mini training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_simplified_tests():
    """Run all simplified tests."""
    print("🚀 Starting Simplified RL Agent Test Suite")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("Agents", test_agents),
        ("Training Components", test_training_components),
        ("Mini Training Loop", test_mini_training_loop)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {test_name}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All core RL components working correctly!")
        print("\nNext steps:")
        print("  1. Integrate with actual Catanatron environment")
        print("  2. Run longer training sessions")
        print("  3. Tune hyperparameters")
        print("  4. Compare DQN vs PPO performance")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_simplified_tests()
    sys.exit(0 if success else 1)
