"""
Step 4: Experience Replay Buffer and Training Utilities
This module provides replay buffer and training utilities for RL agents.
"""
import numpy as np
import random
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os

# Experience tuple for replay buffer
Experience = namedtuple('Experience', [
    'observation', 'action', 'reward', 'next_observation', 
    'done', 'valid_actions', 'next_valid_actions'
])

class ReplayBuffer:
    """Experience replay buffer for DQN-style algorithms"""
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
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
    
    def save(self, path: str):
        """Save replay buffer to disk"""
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        print(f"Replay buffer saved to {path}")
    
    def load(self, path: str):
        """Load replay buffer from disk"""
        with open(path, 'rb') as f:
            experiences = pickle.load(f)
        self.buffer = deque(experiences, maxlen=self.capacity)
        print(f"Replay buffer loaded from {path} with {len(self.buffer)} experiences")

class PPOBuffer:
    """Rollout buffer for PPO algorithm"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.valid_actions_masks = []
        
    def push(self, observation: np.ndarray, action: int, reward: float,
             value: float, log_prob: float, done: bool, valid_actions: List[int]):
        """Add experience to buffer"""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.valid_actions_masks.append(valid_actions)
        
    def get_batch(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> Dict[str, Any]:
        """Compute returns and advantages, return batch"""
        observations = np.array(self.observations)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        log_probs = np.array(self.log_probs)
        dones = np.array(self.dones)
        
        # Compute returns and advantages using GAE
        returns = []
        advantages = []
        
        # Calculate returns (discounted sum of rewards)
        returns_array = np.zeros_like(rewards)
        discounted_sum = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                discounted_sum = 0
            discounted_sum = rewards[t] + gamma * discounted_sum
            returns_array[t] = discounted_sum
        
        # Calculate advantages using GAE
        advantages_array = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                next_value = 0
                gae = 0
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages_array[t] = gae
        
        # Normalize advantages
        advantages_array = (advantages_array - advantages_array.mean()) / (advantages_array.std() + 1e-8)
        
        return {
            'observations': observations,
            'actions': actions,
            'log_probs': log_probs,
            'returns': returns_array,
            'advantages': advantages_array,
            'valid_actions_masks': self.valid_actions_masks
        }
    
    def clear(self):
        """Clear the buffer"""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.valid_actions_masks.clear()
    
    def __len__(self):
        return len(self.observations)

class TrainingLogger:
    """Logger for training metrics"""
    
    def __init__(self, log_dir: str = "logs", experiment_name: str = "catanatron_rl"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.metrics_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_rate': [],
            'loss': [],
            'learning_rate': [],
            'epsilon': []
        }
        
        os.makedirs(log_dir, exist_ok=True)
        
    def log_episode(self, episode: int, reward: float, length: int, won: bool):
        """Log episode metrics"""
        self.metrics_history['episode_rewards'].append(reward)
        self.metrics_history['episode_lengths'].append(length)
        
        # Calculate win rate over last 100 episodes
        recent_wins = []
        for i in range(max(0, len(self.metrics_history['episode_rewards']) - 100), 
                      len(self.metrics_history['episode_rewards'])):
            # Determine if episode was won (can customize this logic)
            episode_won = self.metrics_history['episode_rewards'][i] > 0
            recent_wins.append(episode_won)
        
        win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
        self.metrics_history['win_rate'].append(win_rate)
        
        if episode % 100 == 0:
            avg_reward = np.mean(self.metrics_history['episode_rewards'][-100:])
            avg_length = np.mean(self.metrics_history['episode_lengths'][-100:])
            print(f"Episode {episode}: Avg Reward={avg_reward:.2f}, "
                  f"Avg Length={avg_length:.1f}, Win Rate={win_rate:.3f}")
    
    def log_training_step(self, loss: float, lr: float, epsilon: float = None):
        """Log training step metrics"""
        self.metrics_history['loss'].append(loss)
        self.metrics_history['learning_rate'].append(lr)
        if epsilon is not None:
            self.metrics_history['epsilon'].append(epsilon)
    
    def save_metrics(self):
        """Save metrics to file"""
        path = os.path.join(self.log_dir, f"{self.experiment_name}_metrics.pkl")
        with open(path, 'wb') as f:
            pickle.dump(self.metrics_history, f)
        print(f"Metrics saved to {path}")
    
    def load_metrics(self, path: str):
        """Load metrics from file"""
        with open(path, 'rb') as f:
            self.metrics_history = pickle.load(f)
        print(f"Metrics loaded from {path}")

class CatanatronRewardShaper:
    """Custom reward shaping for Catanatron"""
    
    @staticmethod
    def victory_points_reward(game_state, player_color, action_taken=None) -> float:
        """Reward based on victory points gained"""
        # This would need to be implemented based on Catanatron's game state
        # For now, return placeholder values
        return 0.0
    
    @staticmethod
    def resource_gain_reward(prev_resources, current_resources) -> float:
        """Small reward for gaining resources"""
        if current_resources > prev_resources:
            return 0.1
        return 0.0
    
    @staticmethod
    def building_reward(action_type) -> float:
        """Reward for building actions"""
        building_rewards = {
            'BUILD_SETTLEMENT': 1.0,
            'BUILD_CITY': 2.0,
            'BUILD_ROAD': 0.5,
            'BUY_DEVELOPMENT_CARD': 0.3
        }
        return building_rewards.get(action_type, 0.0)
    
    @staticmethod
    def combined_reward(game_won: bool, victory_points: int, 
                       prev_vp: int, action_type: str = None) -> float:
        """Combine multiple reward signals"""
        reward = 0.0
        
        # Terminal reward
        if game_won:
            reward += 100.0  # Large reward for winning
        else:
            reward -= 1.0    # Small penalty for losing
        
        # Victory points reward
        vp_gained = victory_points - prev_vp
        reward += vp_gained * 10.0
        
        # Building bonus
        if action_type:
            reward += CatanatronRewardShaper.building_reward(action_type)
        
        return reward

def create_training_schedule(total_episodes: int) -> Dict[str, Any]:
    """Create a training schedule with epsilon decay"""
    return {
        'total_episodes': total_episodes,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay_episodes': total_episodes // 2,
        'learning_rate_start': 3e-4,
        'learning_rate_end': 1e-5,
        'learning_rate_decay_episodes': total_episodes,
        'target_update_frequency': 1000,
        'save_frequency': 1000,
        'evaluation_frequency': 500
    }

def get_epsilon(episode: int, schedule: Dict[str, Any]) -> float:
    """Get epsilon value based on schedule"""
    if episode >= schedule['epsilon_decay_episodes']:
        return schedule['epsilon_end']
    
    decay_ratio = episode / schedule['epsilon_decay_episodes']
    epsilon = schedule['epsilon_start'] - decay_ratio * (
        schedule['epsilon_start'] - schedule['epsilon_end']
    )
    return max(epsilon, schedule['epsilon_end'])

def get_learning_rate(episode: int, schedule: Dict[str, Any]) -> float:
    """Get learning rate based on schedule"""
    if episode >= schedule['learning_rate_decay_episodes']:
        return schedule['learning_rate_end']
    
    decay_ratio = episode / schedule['learning_rate_decay_episodes']
    lr = schedule['learning_rate_start'] - decay_ratio * (
        schedule['learning_rate_start'] - schedule['learning_rate_end']
    )
    return max(lr, schedule['learning_rate_end'])

if __name__ == "__main__":
    print("=== Training Utilities Test ===")
    
    # Test replay buffer
    replay_buffer = ReplayBuffer(capacity=1000)
    print(f"Replay buffer created with capacity {replay_buffer.capacity}")
    
    # Test PPO buffer
    ppo_buffer = PPOBuffer(capacity=1000)
    print(f"PPO buffer created with capacity {ppo_buffer.capacity}")
    
    # Test training logger
    logger = TrainingLogger()
    print("Training logger created")
    
    # Test training schedule
    schedule = create_training_schedule(10000)
    print(f"Training schedule: {schedule}")
    
    # Test epsilon decay
    for episode in [0, 1000, 2500, 5000, 10000]:
        epsilon = get_epsilon(episode, schedule)
        lr = get_learning_rate(episode, schedule)
        print(f"Episode {episode}: epsilon={epsilon:.3f}, lr={lr:.2e}")
