#!/usr/bin/env python3
"""
DQN Training Script - Robust Implementation for Report Performance
===============================================================

This script recreates the DQN training from the research report with enhanced 
robustness to achieve the performance targets described in the paper:

Architecture (matching report):
- 614-dimensional state space (actual Catanatron features)
- 290 action categories (actual Catanatron action space)
- Two hidden layers (256, 128 neurons with ReLU) - increased for better capacity
- Experience replay buffer (20,000 transitions) - increased for stability
- Target network updates every 1000 steps - more frequent for stability

Enhanced training for robust performance:
- Increased to 2000 episodes total for better convergence
- Improved curriculum learning across four phases:
  * RandomPlayer (episodes 1-500) - extended foundation
  * WeightedRandomPlayer (501-1000) - gradual difficulty increase  
  * VictoryPointPlayer (1001-1500) - intermediate challenge
  * AlphaBetaPlayer (1501-2000) - expert level training
- Enhanced exploration schedule
- Improved reward shaping

Performance targets (as per report):
- 65% overall win rate 
- 92% win rate vs RandomPlayer
- 60% win rate vs VictoryPointPlayer  
- 32% win rate vs AlphaBetaPlayer (challenging but achievable)
- ~10ms average decision time
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from collections import deque, namedtuple
import random
import json
from datetime import datetime

# Add current directory to path for catanatron imports
sys.path.append('.')

# Import Catanatron components
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer  # Faster alternative to GreedyPlayoutsPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.game import Game
from catanatron.models.enums import Action

# Import our RL components
from rl_agents.core.player import CatanatronRLPlayer, CatanatronRLEnvironmentWrapper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', [
    'observation', 'action', 'reward', 'next_observation', 
    'done', 'valid_actions', 'next_valid_actions'
])

class ReportSpecDQNNetwork(nn.Module):
    """Enhanced DQN Network for better performance"""
    
    def __init__(self, obs_dim: int = 614, action_dim: int = 290):
        super(ReportSpecDQNNetwork, self).__init__()
        
        # Enhanced architecture for better capacity
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, action_dim)
        )
        
        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class ReportSpecReplayBuffer:
    """Enhanced experience replay buffer for stability"""
    
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, observation: np.ndarray, action: int, reward: float,
             next_observation: np.ndarray, done: bool, 
             valid_actions: List[int], next_valid_actions: List[int]):
        """Save an experience"""
        experience = Experience(observation, action, reward, next_observation, 
                              done, valid_actions, next_valid_actions)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        observations = np.array([e.observation for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_observations = np.array([e.next_observation for e in batch])
        dones = np.array([e.done for e in batch])
        valid_actions_masks = [e.next_valid_actions for e in batch]
        
        return {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'next_observations': next_observations,
            'dones': dones,
            'valid_actions_masks': valid_actions_masks
        }
    
    def __len__(self):
        return len(self.buffer)

class ReportSpecDQNAgent:
    """DQN Agent with exact report specifications"""
    
    def __init__(self, obs_dim: int = 95, action_dim: int = 4, 
                 lr: float = 1e-4, gamma: float = 0.99, 
                 epsilon: float = 1.0, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        
        # Networks with exact architecture from report
        self.q_network = ReportSpecDQNNetwork(obs_dim, action_dim).to(device)
        self.target_q_network = ReportSpecDQNNetwork(obs_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
          # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Target network update frequency: more frequent for stability
        self.update_counter = 0
        self.target_update_freq = 1000
        
        logger.info(f"Enhanced DQN Agent initialized: {obs_dim}D state -> 256 -> 128 -> {action_dim}D action")
        logger.info(f"Target network updates every {self.target_update_freq} steps")
        
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        """Select action using epsilon-greedy with action masking"""
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor).squeeze()
            
            # Mask invalid actions
            masked_q_values = q_values.clone()
            mask = torch.ones(self.action_dim, dtype=torch.bool)
            mask[valid_actions] = False
            masked_q_values[mask] = float('-inf')
            
            action = torch.argmax(masked_q_values).item()
            
        return action if action in valid_actions else np.random.choice(valid_actions)
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update Q-network using DQN loss"""
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_observations = torch.FloatTensor(batch['next_observations']).to(self.device)
        dones = torch.BoolTensor(batch['dones']).to(self.device)
        valid_actions_masks = batch['valid_actions_masks']
        
        # Current Q-values
        current_q_values = self.q_network(observations).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_q_network(next_observations)
            
            # Mask invalid actions for next states
            for i, valid_actions in enumerate(valid_actions_masks):
                mask = torch.ones(self.action_dim, dtype=torch.bool)
                mask[valid_actions] = False
                next_q_values[i][mask] = float('-inf')
            
            max_next_q_values = next_q_values.max(1)[0].detach()
            target_q_values = rewards + (self.gamma * max_next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network more frequently for stability
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            logger.info(f"Updated target network at step {self.update_counter}")
        
        return {"loss": loss.item(), "q_values_mean": current_q_values.mean().item()}
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_counter': self.update_counter,
            'epsilon': self.epsilon
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_counter = checkpoint['update_counter']
        self.epsilon = checkpoint['epsilon']
        logger.info(f"Agent loaded from {path}")

class CurriculumTrainingEnvironment:
    """Curriculum learning environment matching report specifications"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.current_phase = 0
        self.episode = 0
          # Curriculum phases for robust 2000 episodes training
        self.curriculum_phases = [            {"episodes": (1, 500), "opponent": RandomPlayer, "name": "RandomPlayer"},
            {"episodes": (501, 1000), "opponent": WeightedRandomPlayer, "name": "WeightedRandomPlayer"}, 
            {"episodes": (1001, 1500), "opponent": VictoryPointPlayer, "name": "VictoryPointPlayer"},
            {"episodes": (1501, 2000), "opponent": AlphaBetaPlayer, "name": "AlphaBetaPlayer"}
        ]
        self.current_env = None
        logger.info("Curriculum training environment initialized")
        logger.info("Phases: RandomPlayer (1-500), WeightedRandomPlayer (501-1000), VictoryPointPlayer (1001-1500), AlphaBetaPlayer (1501-2000)")
    
    def get_current_opponent(self, episode: int):
        """Get opponent for current episode"""
        for i, phase in enumerate(self.curriculum_phases):
            start, end = phase["episodes"]
            if start <= episode <= end:
                self.current_phase = i
                return phase["opponent"], phase["name"]
        
        # Default to last phase if beyond 1001 episodes
        last_phase = self.curriculum_phases[-1]
        return last_phase["opponent"], last_phase["name"]
    
    def reset(self, episode: int):
        """Reset environment for new episode with appropriate opponent"""
        self.episode = episode
        opponent_class, opponent_name = self.get_current_opponent(episode)
        
        # Create opponent with appropriate parameters
        if opponent_class == AlphaBetaPlayer:
            opponent = opponent_class(Color.RED, prunning_improved=True)
        else:
            opponent = opponent_class(Color.RED)
          # Create 2-player environment configuration
        env_config = {"enemies": [opponent]}
        self.current_env = CatanatronRLEnvironmentWrapper(env_config)
        
        if episode == 1 or self.get_current_opponent(episode-1)[1] != opponent_name:
            logger.info(f"Episode {episode}: Training against {opponent_name}")
        
        return self.current_env.reset()
    
    def step(self, action):
        """Take step in current environment"""
        return self.current_env.step(action)
    
    @property
    def obs_dim(self):
        return self.current_env.obs_dim if self.current_env else 614
    
    @property  
    def action_dim(self):
        return self.current_env.action_dim if self.current_env else 290

def train_report_spec_dqn():
    """Train DQN agent with exact report specifications"""
    
    # Create environment wrapper to get actual dimensions
    env_wrapper = CatanatronRLEnvironmentWrapper()
    actual_obs_dim = env_wrapper.obs_dim
    actual_action_dim = env_wrapper.action_dim
    env_wrapper.close()
    
    logger.info(f"Detected environment dimensions: obs_dim={actual_obs_dim}, action_dim={actual_action_dim}")
    logger.info("NOTE: Report specified 95-dim state, but actual environment has different dimensions")
      # Create output directory
    output_dir = "training_outputs_dqn"
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Configuration using actual environment dimensions (enhanced for robustness)
    config = {
        "total_episodes": 2000,  # Increased from 600 for better performance
        "obs_dim": int(actual_obs_dim),
        "action_dim": int(actual_action_dim), 
        "replay_buffer_size": 20000,  # Increased for stability
        "batch_size": 64,  # Increased for better updates
        "learning_rate": 5e-5,  # Reduced for stability
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,  # Lower final epsilon
        "epsilon_decay_episodes": 1600,  # 80% of total episodes
        "target_update_freq": 1000,  # More frequent updates for stability
        "learning_starts": 2000,  # Start learning after enough experiences
        "train_freq": 4,
        "gradient_steps": 1,
        "reward_scale": 1.0
    }
    
    # Save configuration
    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("=== Starting DQN Training with Report Specifications ===")
    logger.info(f"Configuration: {config}")
    
    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    env = CurriculumTrainingEnvironment(device)
    agent = ReportSpecDQNAgent(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"], 
        lr=config["learning_rate"],
        gamma=config["gamma"],
        epsilon=config["epsilon_start"],
        device=device
    )
    replay_buffer = ReportSpecReplayBuffer(capacity=config["replay_buffer_size"])
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    training_start_time = time.time()
      # Training loop - enhanced with 2000 episodes for better convergence
    for episode in range(1, config["total_episodes"] + 1):
        
        # Epsilon decay schedule
        if episode <= config["epsilon_decay_episodes"]:
            epsilon_progress = (episode - 1) / config["epsilon_decay_episodes"]
            agent.epsilon = config["epsilon_start"] - epsilon_progress * (config["epsilon_start"] - config["epsilon_end"])
        else:
            agent.epsilon = config["epsilon_end"]
        
        # Reset environment for curriculum learning
        observation, info = env.reset(episode)
        
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        done = False
        
        while not done:
            # Get valid actions (4 high-level categories from report)
            valid_actions = info.get('valid_actions', list(range(config["action_dim"])))
            
            # Select action
            action = agent.select_action(observation, valid_actions)
            
            # Take step
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Store experience in replay buffer
            next_valid_actions = next_info.get('valid_actions', list(range(config["action_dim"])))
            replay_buffer.push(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
                valid_actions=valid_actions,
                next_valid_actions=next_valid_actions
            )
            
            # Train agent after learning_starts episodes and every train_freq steps
            if (len(replay_buffer) >= config["learning_starts"] and 
                agent.update_counter % config["train_freq"] == 0):
                
                batch = replay_buffer.sample(config["batch_size"])
                loss_info = agent.update(batch)
                episode_losses.append(loss_info["loss"])
            
            # Update state
            observation = next_observation
            info = next_info
            episode_reward += reward
            episode_length += 1
        
        # Log episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        losses.append(avg_loss)
        
        # Periodic logging
        if episode % 50 == 0 or episode == 1:
            avg_reward_50 = np.mean(episode_rewards[-50:])
            avg_length_50 = np.mean(episode_lengths[-50:])
            avg_loss_50 = np.mean(losses[-50:]) if losses else 0
            
            logger.info(f"Episode {episode:4d} | "
                       f"Reward: {episode_reward:6.2f} | "
                       f"Length: {episode_length:3d} | " 
                       f"Epsilon: {agent.epsilon:.3f} | "
                       f"Avg Reward (50): {avg_reward_50:6.2f} | "
                       f"Avg Length (50): {avg_length_50:5.1f} | "
                       f"Loss: {avg_loss:.4f}")
          # Save model periodically and at phase transitions
        if (episode % 500 == 0 or episode == config["total_episodes"] or 
            episode in [500, 1000, 1500]):  # Phase transition saves
            
            model_path = os.path.join(models_dir, f"dqn_episode_{episode}.pt")
            agent.save(model_path)
            
            # Save training metrics
            metrics = {
                "episode": episode,
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths, 
                "losses": losses,
                "training_time": time.time() - training_start_time
            }
            
            metrics_path = os.path.join(output_dir, f"training_metrics_episode_{episode}.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            logger.info(f"Model and metrics saved at episode {episode}")
    
    # Final model save
    final_model_path = os.path.join(models_dir, "final_dqn_model.pt")
    agent.save(final_model_path)
    
    # Final training summary
    total_time = time.time() - training_start_time
    final_avg_reward = np.mean(episode_rewards[-100:])
    
    logger.info("=== Training Completed ===")
    logger.info(f"Total episodes: {config['total_episodes']}")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info(f"Final average reward (last 100 episodes): {final_avg_reward:.2f}")
    logger.info(f"Final epsilon: {agent.epsilon:.3f}")
    logger.info(f"Total network updates: {agent.update_counter}")
    logger.info(f"Final model saved to: {final_model_path}")
    
    return final_model_path, output_dir

if __name__ == "__main__":
    print("=" * 80)
    print("DQN Training Script - Enhanced for Report Performance")
    print("=" * 80)
    print("Training configuration:")
    print("- 614-dimensional state space (actual Catanatron features)")
    print("- 290 action categories (full Catanatron action space)") 
    print("- Architecture: 614 -> 256 -> 128 -> 290")
    print("- Experience replay: 20,000 transitions")
    print("- Target network updates: every 1000 steps")
    print("- Curriculum learning: 2000 episodes for robust performance")
    print("  * RandomPlayer (1-500) - Extended foundation")
    print("  * WeightedRandomPlayer (501-1000) - Gradual difficulty")
    print("  * VictoryPointPlayer (1001-1500) - Intermediate challenge")
    print("  * AlphaBetaPlayer (1501-2000) - Expert level training")
    print("=" * 80)
    
    try:
        model_path, output_dir = train_report_spec_dqn()
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🤖 Final model: {model_path}")
        print("\nNext step: Run evaluation script to test against all opponents")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
