#!/usr/bin/env python3
"""
Simple Training Test for Catanatron RL Agent
===========================================

This script runs a minimal training session to test that everything works.
Phase 4 of the RL agent development.
"""

import gym
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
from step5_rl_player import CatanatronRLPlayer
from step6_training_pipeline import CatanatronTrainer

def test_basic_functionality():
    """Test basic functionality of our RL components."""
    print("=== Testing Basic RL Functionality ===")
    
    # Create a simple environment to test
    try:
        env = gymnasium.make('catanatron_gym:catanatron-v1')
        print(f"✓ Environment created successfully")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")
        
        # Test observation and action spaces
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Info keys: {list(info.keys())}")
        
        # Test action masking
        if 'action_mask' in info:
            valid_actions = np.where(info['action_mask'])[0]
            print(f"  - Valid actions available: {len(valid_actions)}")
        else:
            print("  - No action mask found in info")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_agent_creation():
    """Test creating DQN and PPO agents."""
    print("\n=== Testing Agent Creation ===")
    
    try:
        # Test DQN agent
        obs_dim = 614  # From environment testing
        action_dim = 290
        
        dqn_agent = MaskedDQNAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=1e-4,
            buffer_size=10000,
            batch_size=32
        )
        print(f"✓ DQN Agent created successfully")
        print(f"  - Network parameters: {sum(p.numel() for p in dqn_agent.q_network.parameters())}")
        
        # Test PPO agent
        ppo_agent = MaskedPPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=128,
            learning_rate=3e-4
        )
        print(f"✓ PPO Agent created successfully")
        print(f"  - Actor parameters: {sum(p.numel() for p in ppo_agent.actor.parameters())}")
        print(f"  - Critic parameters: {sum(p.numel() for p in ppo_agent.critic.parameters())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Agent creation failed: {e}")
        return False

def test_training_utilities():
    """Test training utilities."""
    print("\n=== Testing Training Utilities ===")
    
    try:
        # Test logger
        logger = TrainingLogger("test_run")
        logger.log_scalar("test_reward", 0.5, 1)
        logger.log_scalar("test_loss", 0.1, 1)
        print("✓ Training logger working")
        
        # Test reward shaper
        reward_shaper = CatanatronRewardShaper()
        print("✓ Reward shaper created")
        
        return True
        
    except Exception as e:
        print(f"✗ Training utilities test failed: {e}")
        return False

def test_mini_training():
    """Run a very short training session to test the full pipeline."""
    print("\n=== Testing Mini Training Session ===")
    
    try:
        # Create trainer with minimal configuration
        config = {
            'agent_type': 'dqn',
            'total_timesteps': 100,  # Very short for testing
            'eval_freq': 50,
            'save_freq': 100,
            'log_freq': 10,
            'batch_size': 16,
            'buffer_size': 1000,
            'learning_rate': 1e-4,
            'hidden_dim': 64,  # Smaller network for speed
            'exploration_fraction': 0.5,
            'exploration_final_eps': 0.1,
            'target_update_freq': 20
        }
        
        trainer = CatanatronTrainer(config, run_name="mini_test")
        print("✓ Trainer created successfully")
        
        # Run mini training
        print("  Running mini training session...")
        final_metrics = trainer.train()
        
        print(f"✓ Mini training completed!")
        print(f"  - Final average reward: {final_metrics.get('avg_reward', 'N/A')}")
        print(f"  - Total episodes: {final_metrics.get('total_episodes', 'N/A')}")
        
        # Test model saving/loading
        model_path = trainer.save_model("mini_test_model")
        print(f"✓ Model saved to: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Mini training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests in sequence."""
    print("🚀 Starting RL Agent Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Agent Creation", test_agent_creation),
        ("Training Utilities", test_training_utilities),
        ("Mini Training", test_mini_training)
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
        print("🎉 All tests passed! Your RL agent setup is working correctly.")
        print("\nNext steps:")
        print("  1. Run longer training sessions")
        print("  2. Experiment with hyperparameters")
        print("  3. Compare DQN vs PPO performance")
        print("  4. Add more sophisticated reward shaping")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    # Make sure we can import catanatron
    try:
        import catanatron_gym
        print("✓ Catanatron gym environment available")
    except ImportError as e:
        print(f"✗ Cannot import catanatron_gym: {e}")
        print("Please make sure Catanatron is installed:")
        print("  pip install catanatron[gym]")
        sys.exit(1)
    
    # Run the test suite
    success = run_all_tests()
    sys.exit(0 if success else 1)
