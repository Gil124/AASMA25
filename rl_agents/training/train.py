"""
Step 10: Working Training Script for Catanatron RL Agent

This script provides a complete, working implementation that trains an RL agent
using all the components developed in previous steps. It includes proper error
handling, logging, and configuration management.
"""

import os
import sys
import time
import traceback
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import json

# Add current directory to path for catanatron imports
sys.path.append('.')

# Register the catanatron gym environment
import catanatron.gym

# Import our custom components
from ..core.agents import MaskedDQNAgent, MaskedPPOAgent
from .utilities import TrainingLogger, CatanatronRewardShaper
from rl_agents.core.player import CatanatronRLPlayer
from rl_agents.training.pipeline import CatanatronTrainer


def setup_logging(log_dir: str) -> logging.Logger:
    """Setup logging for the training process."""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('catanatron_rl_training')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class CatanatronRLTrainingConfig:
    """Configuration class for RL training parameters."""
    
    def __init__(self):
        # Environment settings
        self.env_name = "catanatron/Catanatron-v0"
        self.num_players = 4
        self.invalid_action_reward = -1.0
        
        # Training settings
        self.agent_type = "DQN"  # "DQN" or "PPO"
        self.total_timesteps = 100000
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.buffer_size = 10000
        self.learning_starts = 1000
        self.train_freq = 4
        self.target_update_freq = 1000
        self.exploration_fraction = 0.1
        self.exploration_initial_eps = 1.0
        self.exploration_final_eps = 0.05
        
        # PPO specific settings
        self.ppo_epochs = 10
        self.clip_range = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        
        # Network architecture
        self.hidden_dim = 256
        self.num_layers = 3
        
        # Evaluation settings
        self.eval_freq = 5000
        self.eval_episodes = 10
        
        # Logging and checkpointing
        self.log_freq = 1000
        self.checkpoint_freq = 10000
        self.save_replay_buffer = True
        
        # Output directories
        self.output_dir = "training_outputs"
        self.models_dir = "models"
        self.logs_dir = "logs"
        self.tensorboard_dir = "tensorboard"
    
    def create_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.output_dir,
            os.path.join(self.output_dir, self.models_dir),
            os.path.join(self.output_dir, self.logs_dir),
            os.path.join(self.output_dir, self.tensorboard_dir)
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_config(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = {
            key: value for key, value in self.__dict__.items()
            if not callable(value) and not key.startswith('_')
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self, filepath: str):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


def test_environment_setup(logger: logging.Logger) -> bool:
    """Test that the Catanatron environment is properly set up."""
    try:
        logger.info("Testing environment setup...")
        
        # Create environment
        env = gym.make("catanatron/Catanatron-v0")
        
        # Test reset
        obs, info = env.reset()
        logger.info(f"Environment reset successful. Observation shape: {obs.shape}")
        
        # Test action space
        action_space_size = env.action_space.n
        logger.info(f"Action space size: {action_space_size}")
        
        # Test valid actions
        if hasattr(env, 'get_valid_actions'):
            valid_actions = env.get_valid_actions()
            logger.info(f"Valid actions count: {len(valid_actions)}")
        else:
            logger.warning("Environment doesn't have get_valid_actions method")
        
        # Test a few random steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"Step {i+1}: Action {action}, Reward {reward}")
            
            if terminated or truncated:
                obs, info = env.reset()
                logger.info("Episode ended, environment reset")
        
        env.close()
        logger.info("Environment test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Environment test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def create_agent(config: CatanatronRLTrainingConfig, 
                 obs_space: gym.Space, 
                 action_space: gym.Space,
                 logger: logging.Logger) -> Optional[object]:
    """Create the appropriate RL agent based on configuration."""
    try:
        logger.info(f"Creating {config.agent_type} agent...")
        
        agent_type_upper = config.agent_type.upper()
        
        if agent_type_upper == "DQN":
            agent = MaskedDQNAgent(
                obs_dim=obs_space.shape[0],
                action_dim=action_space.n,
                lr=config.learning_rate,
                gamma=0.99,
                epsilon=config.exploration_initial_eps
            )
        elif agent_type_upper == "PPO":
            agent = MaskedPPOAgent(
                obs_dim=obs_space.shape[0],
                action_dim=action_space.n,
                lr=config.learning_rate,
                gamma=0.99,
                clip_ratio=config.clip_range
            )
        else:
            raise ValueError(f"Unknown agent type: {config.agent_type}")
        
        logger.info(f"{config.agent_type} agent created successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def train_agent(config: CatanatronRLTrainingConfig, logger: logging.Logger) -> bool:
    """Main training function."""
    try:
        logger.info("Starting RL agent training...")
        
        # Create directories
        config.create_directories()
        
        # Save configuration
        config_path = os.path.join(config.output_dir, "training_config.json")
        config.save_config(config_path)
        logger.info(f"Configuration saved to {config_path}")
        
        # Create environment
        logger.info("Creating training environment...")
        env = gym.make(config.env_name)
        
        # Create agent
        agent = create_agent(config, env.observation_space, env.action_space, logger)
        if agent is None:
            return False        # Create trainer
        logger.info("Creating trainer...")
        trainer = CatanatronTrainer(
            agent_type=config.agent_type,
            log_dir=config.output_dir,
            model_dir=os.path.join(config.output_dir, "models")
        )
        
        # Start training
        logger.info(f"Starting training for {config.total_timesteps} timesteps...")
        
        training_start_time = time.time()
        
        # Estimate episodes from timesteps (assuming ~300 steps per episode)
        estimated_episodes = max(1, config.total_timesteps // 300)
          # Train the agent
        trainer.train(total_episodes=estimated_episodes)
        
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time
        
        logger.info(f"Training completed in {training_duration:.2f} seconds")
        logger.info(f"Final model saved to {trainer.model_dir}")
        
        # Clean up
        env.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def quick_evaluation(config: CatanatronRLTrainingConfig, 
                    model_path: str, 
                    logger: logging.Logger) -> bool:
    """Run a quick evaluation of the trained model."""
    try:
        logger.info("Running quick evaluation...")
        
        # Create environment
        env = gym.make(config.env_name)
          # Load trained agent
        if config.agent_type == "DQN":
            agent = MaskedDQNAgent(
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                lr=config.learning_rate
            )
        else:
            agent = MaskedPPOAgent(
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                lr=config.learning_rate
            )
        
        # Load model weights if they exist
        if os.path.exists(model_path):
            agent.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
            return False
        
        # Run evaluation episodes
        total_rewards = []
        episode_lengths = []
        
        for episode in range(config.eval_episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Get valid actions
                if hasattr(env, 'get_valid_actions'):
                    valid_actions = env.get_valid_actions()
                    action_mask = np.zeros(env.action_space.n, dtype=bool)
                    action_mask[valid_actions] = True
                else:
                    action_mask = np.ones(env.action_space.n, dtype=bool)
                
                # Get action from agent
                action = agent.act(obs, action_mask, training=False)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            logger.info(f"Evaluation episode {episode + 1}: Reward = {episode_reward}, Length = {episode_length}")
        
        # Calculate statistics
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        mean_length = np.mean(episode_lengths)
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"  Mean Episode Length: {mean_length:.2f}")
        
        env.close()
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


def main(agent_type: str = "dqn", num_episodes: int = 1000, 
         config_path: str = None, output_dir: str = "training_outputs",
         device: str = "cpu", seed: int = 42):
    """
    Main training function that can be called programmatically.
    
    Args:
        agent_type: Type of agent to train ("dqn" or "ppo")
        num_episodes: Number of training episodes
        config_path: Path to configuration file
        output_dir: Directory to save outputs
        device: Device to use for training
        seed: Random seed for reproducibility
    
    Returns:
        bool: True if training completed successfully, False otherwise
    """
    print("=" * 60)
    print("Catanatron RL Agent Training Pipeline")
    print("=" * 60)
    
    try:
        # Create and configure training config
        config = CatanatronRLTrainingConfig()
        
        # Override with provided parameters
        config.agent_type = agent_type
        config.num_episodes = num_episodes
        config.output_dir = output_dir
        config.device = device
        config.seed = seed
        
        # Load config from file if provided
        if config_path and os.path.exists(config_path):
            config.load_config(config_path)
        
        # Setup logging
        logger = setup_logging(os.path.join(config.output_dir, config.logs_dir))
        
        logger.info("Starting Catanatron RL training pipeline")
        logger.info(f"Configuration: Agent={config.agent_type}, Episodes={config.num_episodes}")
        
        # Test environment setup
        if not test_environment_setup(logger):
            logger.error("Environment test failed. Exiting.")
            return False
        
        # Train agent
        if not train_agent(config, logger):
            logger.error("Training failed. Exiting.")
            return False
        
        # Quick evaluation
        model_path = os.path.join(config.output_dir, config.models_dir, "final_model.pth")
        if not quick_evaluation(config, model_path, logger):
            logger.warning("Evaluation failed, but training was successful.")
        
        logger.info("Training pipeline completed successfully!")
        print("\n" + "=" * 60)
        print("Training completed! Check the logs and output directories for results.")
        print(f"Output directory: {config.output_dir}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)