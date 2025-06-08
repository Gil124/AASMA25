"""
Step 6: Complete Training Pipeline
This module implements the complete training pipeline for RL agents in Catanatron.
"""
import numpy as np
import torch
import gymnasium as gym
import catanatron.gym
import time
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict
import matplotlib.pyplot as plt

# Import our components
from ..core.agents import MaskedDQNAgent, MaskedPPOAgent, get_agent_config
from .utilities import (
    ReplayBuffer, PPOBuffer, TrainingLogger, 
    create_training_schedule, get_epsilon, get_learning_rate,
    CatanatronRewardShaper
)
from ..core.player import CatanatronRLPlayer, CatanatronRLEnvironmentWrapper

# Import Catanatron components
from catanatron.models.player import Color
from catanatron.models.player import RandomPlayer
from catanatron.players.value import ValueFunctionPlayer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CatanatronTrainer:
    """Complete training pipeline for Catanatron RL agents"""
    
    def __init__(self, 
                 agent_type: str = "dqn",
                 config: Optional[Dict[str, Any]] = None,
                 log_dir: str = "training_logs",
                 model_dir: str = "models",
                 device: str = "cpu"):
        
        self.agent_type = agent_type
        self.config = config or get_agent_config(agent_type)
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.device = device
          # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.env = None
        self.agent = None
        self.buffer = None
        self.logger = TrainingLogger(log_dir, f"catanatron_{agent_type}")
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0
    
    def setup_environment(self, env_config: Dict[str, Any] = None):
        """Setup the training environment with 2-player games and curriculum learning"""
        logger.info("Setting up Catanatron 2-player environment with heuristic opponents...")
        
        # Import heuristic players for training
        from catanatron.players.minimax import AlphaBetaPlayer
        from catanatron.players.weighted_random import WeightedRandomPlayer
        from catanatron.players.mcts import MCTSPlayer
        from catanatron.players.playouts import GreedyPlayoutsPlayer
        
        # Curriculum learning: start with easier opponents, progress to harder ones
        training_phase = self.config.get("training_phase", "aplabeta")
        
        if training_phase == "basic":
            # Phase 1: Train against random and weighted random
            opponent_class = RandomPlayer if self.episode < 1000 else WeightedRandomPlayer
        elif training_phase == "intermediate":
            # Phase 2: Train against greedy and basic MCTS
            opponent_class = GreedyPlayoutsPlayer if self.episode < 2000 else MCTSPlayer
        else:  # advanced
            # Phase 3: Train against AlphaBeta (the champion)
            opponent_class = AlphaBetaPlayer
          # 2-player configuration (RL agent vs single opponent)
        # RL agent is Color.BLUE (p0), opponent should be Color.RED
        default_config = {
            "enemies": [
                opponent_class(Color.RED, prunning_improved=True) if opponent_class == AlphaBetaPlayer 
                else opponent_class(Color.RED)
            ]
        }
        
        if env_config:
            default_config.update(env_config)
        
        self.env = CatanatronRLEnvironmentWrapper(default_config)
        logger.info(f"2-player environment created with {opponent_class.__name__} opponent")
        logger.info(f"obs_dim={self.env.obs_dim}, action_dim={self.env.action_dim}")
        
        return self.env.obs_dim, self.env.action_dim
    
    def setup_agent(self, obs_dim: int, action_dim: int):
        """Setup the RL agent"""
        logger.info(f"Setting up {self.agent_type.upper()} agent...")
        
        agent_type_lower = self.agent_type.lower()
        if agent_type_lower == "dqn":
            self.agent = MaskedDQNAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=self.config.get("lr", 3e-4),
                gamma=self.config.get("gamma", 0.99),
                epsilon=self.config.get("epsilon", 0.1),
                device=self.device
            )
            self.buffer = ReplayBuffer(capacity=self.config.get("replay_buffer_size", 100000))
            
        elif agent_type_lower == "ppo":
            self.agent = MaskedPPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=self.config.get("lr", 3e-4),
                gamma=self.config.get("gamma", 0.99),
                clip_ratio=self.config.get("clip_ratio", 0.2),
                device=self.device
            )
            self.buffer = PPOBuffer(capacity=self.config.get("batch_size", 64) * 20)
        
        logger.info(f"Agent created with {sum(p.numel() for p in self._get_agent_parameters())} parameters")
    
    def _get_agent_parameters(self):
        """Get agent parameters for counting"""
        agent_type_lower = self.agent_type.lower()
        if agent_type_lower == "dqn":
            return self.agent.q_network.parameters()
        elif agent_type_lower == "ppo":
            return self.agent.network.parameters()
        return []
    
    def train_dqn(self, total_episodes: int = 10000):
        """Train DQN agent"""
        logger.info(f"Starting DQN training for {total_episodes} episodes...")
        
        # Training schedule
        schedule = create_training_schedule(total_episodes)
        
        # Training loop
        for episode in range(total_episodes):
            self.episode = episode
            episode_reward = 0
            episode_length = 0
            episode_losses = []
            
            # Update epsilon
            current_epsilon = get_epsilon(episode, schedule)
            self.agent.epsilon = current_epsilon
            
            # Reset environment
            observation, info = self.env.reset()
            done = False
            
            while not done:
                # Get valid actions
                valid_actions = info.get('valid_actions', list(range(self.env.action_dim)))
                
                # Select action
                action = self.agent.select_action(observation, valid_actions)
                
                # Take step
                next_observation, reward, terminated, truncated, next_info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                next_valid_actions = next_info.get('valid_actions', list(range(self.env.action_dim)))
                self.buffer.push(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=done,
                    valid_actions=valid_actions,
                    next_valid_actions=next_valid_actions
                )
                
                # Update agent
                if len(self.buffer) >= self.config.get("batch_size", 32):
                    batch = self.buffer.sample(self.config.get("batch_size", 32))
                    loss_info = self.agent.update(batch)
                    episode_losses.append(loss_info.get("loss", 0))
                
                # Update state
                observation = next_observation
                info = next_info
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1
            
            # Log episode
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            self.logger.log_episode(episode, episode_reward, episode_length, episode_reward > 0)
            self.logger.log_training_step(avg_loss, self.agent.optimizer.param_groups[0]['lr'], current_epsilon)
            
            # Save model periodically
            if episode > 0 and episode % schedule.get("save_frequency", 1000) == 0:
                self._save_checkpoint(episode)
            
            # Evaluate periodically
            if episode > 0 and episode % schedule.get("evaluation_frequency", 500) == 0:
                self._evaluate_agent(num_games=50)
        
        logger.info("DQN training completed!")
    
    def train_ppo(self, total_episodes: int = 10000):
        """Train PPO agent"""
        logger.info(f"Starting PPO training for {total_episodes} episodes...")
        
        # Training schedule
        schedule = create_training_schedule(total_episodes)
        update_frequency = self.config.get("batch_size", 64)
        
        # Training loop
        for episode in range(total_episodes):
            self.episode = episode
            episode_reward = 0
            episode_length = 0
            
            # Reset environment
            observation, info = self.env.reset()
            done = False
            
            while not done:
                # Get valid actions
                valid_actions = info.get('valid_actions', list(range(self.env.action_dim)))
                
                # Select action
                action, log_prob, value = self.agent.select_action(observation, valid_actions)
                
                # Take step
                next_observation, reward, terminated, truncated, next_info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience
                self.buffer.push(
                    observation=observation,
                    action=action,
                    reward=reward,
                    value=value,
                    log_prob=log_prob,
                    done=done,
                    valid_actions=valid_actions
                )
                
                # Update state
                observation = next_observation
                info = next_info
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1
            
            # Update agent when buffer is full
            if len(self.buffer) >= update_frequency:
                batch = self.buffer.get_batch(
                    gamma=self.config.get("gamma", 0.99),
                    gae_lambda=self.config.get("gae_lambda", 0.95)
                )
                
                # Multiple epochs of updates
                n_epochs = self.config.get("n_epochs", 10)
                losses = []
                for _ in range(n_epochs):
                    loss_info = self.agent.update(batch)
                    losses.append(loss_info)
                
                # Clear buffer
                self.buffer.clear()
                
                # Log training metrics
                avg_loss = np.mean([l.get("total_loss", 0) for l in losses])
                self.logger.log_training_step(avg_loss, self.agent.optimizer.param_groups[0]['lr'])
            
            # Log episode
            self.logger.log_episode(episode, episode_reward, episode_length, episode_reward > 0)
            
            # Save model periodically
            if episode > 0 and episode % schedule.get("save_frequency", 1000) == 0:
                self._save_checkpoint(episode)
            
            # Evaluate periodically
            if episode > 0 and episode % schedule.get("evaluation_frequency", 500) == 0:
                self._evaluate_agent(num_games=50)
        
        logger.info("PPO training completed!")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        model_path = os.path.join(self.model_dir, f"{self.agent_type}_episode_{episode}.pt")
        self.agent.save(model_path)
        
        # Save buffer
        buffer_path = os.path.join(self.model_dir, f"buffer_episode_{episode}.pkl")
        if hasattr(self.buffer, 'save'):
            self.buffer.save(buffer_path)
        
        # Save metrics
        self.logger.save_metrics()
        
        logger.info(f"Checkpoint saved at episode {episode}")
    
    def _evaluate_agent(self, num_games: int = 100) -> Dict[str, float]:
        """Evaluate agent performance"""
        logger.info(f"Evaluating agent over {num_games} games...")
        
        # Temporarily disable exploration
        original_epsilon = getattr(self.agent, 'epsilon', None)
        if original_epsilon is not None:
            self.agent.epsilon = 0.0
        
        wins = 0
        total_rewards = []
        game_lengths = []
        
        for game in range(num_games):
            observation, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            while not done:
                valid_actions = info.get('valid_actions', list(range(self.env.action_dim)))
                
                agent_type_lower = self.agent_type.lower()
                if agent_type_lower == "dqn":
                    action = self.agent.select_action(observation, valid_actions)
                elif agent_type_lower == "ppo":
                    action, _, _ = self.agent.select_action(observation, valid_actions)
                
                observation, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            if episode_reward > 0:  # Assuming positive reward means win
                wins += 1
            
            total_rewards.append(episode_reward)
            game_lengths.append(episode_length)
        
        # Restore original epsilon
        if original_epsilon is not None:
            self.agent.epsilon = original_epsilon
        
        # Calculate metrics
        win_rate = wins / num_games
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(game_lengths)
        
        evaluation_results = {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_game_length": avg_length,
            "total_games": num_games
        }
        
        logger.info(f"Evaluation results: Win rate={win_rate:.3f}, "
                   f"Avg reward={avg_reward:.2f}, Avg length={avg_length:.1f}")
          # Save best model
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            best_model_path = os.path.join(self.model_dir, f"best_{self.agent_type}_model.pt")
            self.agent.save(best_model_path)
            logger.info(f"New best model saved with win rate {win_rate:.3f}")
        
        return evaluation_results
    
    def train(self, total_episodes: int = 10000):
        """Main training function"""
        logger.info(f"Starting training pipeline for {self.agent_type.upper()}...")
        
        # Setup
        obs_dim, action_dim = self.setup_environment()
        self.setup_agent(obs_dim, action_dim)
        
        # Train based on agent type
        agent_type_lower = self.agent_type.lower()
        if agent_type_lower == "dqn":
            self.train_dqn(total_episodes)
        elif agent_type_lower == "ppo":
            self.train_ppo(total_episodes)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
        
        # Final evaluation
        final_results = self._evaluate_agent(num_games=200)
        logger.info(f"Final evaluation: {final_results}")
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, f"final_{self.agent_type}_model.pt")
        self.agent.save(final_model_path)
        
        # Close environment
        self.env.close()
        
        return final_results
    
    def save_model(self, path: str):
        """Save the current model"""
        if self.agent is not None:
            self.agent.save(path)
            logger.info(f"Model saved to {path}")
        else:
            logger.warning("No agent to save")

def create_trainer_config(agent_type: str = "dqn") -> Dict[str, Any]:
    """Create training configuration"""
    base_config = get_agent_config(agent_type)
    
    # Add training-specific configs
    base_config.update({
        "total_episodes": 20000,
        "evaluation_frequency": 1000,
        "save_frequency": 2000,
        "log_frequency": 100
    })
    
    return base_config

def visualize_training_progress(log_dir: str, experiment_name: str):
    """Create training progress visualizations"""
    import pickle
    
    metrics_path = os.path.join(log_dir, f"{experiment_name}_metrics.pkl")
    
    if not os.path.exists(metrics_path):
        logger.warning(f"Metrics file not found: {metrics_path}")
        return
    
    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(metrics['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    
    # Win rate
    axes[0, 1].plot(metrics['win_rate'])
    axes[0, 1].set_title('Win Rate (Rolling 100 episodes)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Win Rate')
    
    # Episode lengths
    axes[1, 0].plot(metrics['episode_lengths'])
    axes[1, 0].set_title('Episode Lengths')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Length')
    
    # Loss
    if metrics['loss']:
        axes[1, 1].plot(metrics['loss'])
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    plot_path = os.path.join(log_dir, f"{experiment_name}_training_progress.png")
    plt.savefig(plot_path)
    logger.info(f"Training progress plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    print("=== Catanatron RL Training Pipeline ===")
    
    # Example usage
    config = create_trainer_config("dqn")
    print(f"Training configuration: {config}")
    
    # Create trainer
    trainer = CatanatronTrainer(
        agent_type="dqn",
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Trainer created successfully!")
    print(f"Using device: {trainer.device}")
    
    # To actually run training, uncomment the following:
    # results = trainer.train(total_episodes=1000)
    # print(f"Training completed with results: {results}")
    
    # # Visualize results
    # visualize_training_progress(trainer.log_dir, f"catanatron_{trainer.agent_type}")
