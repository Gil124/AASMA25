"""
Step 5: Custom Catanatron RL Player Implementation
This module implements a custom RL player that integrates with Catanatron's player system.
"""
import numpy as np
import torch
from typing import List, Dict, Any, Optional
import gymnasium as gym
import logging
import time

from catanatron.models.player import Player, Color
from catanatron.models.enums import Action, ActionType
from catanatron.game import Game

# Import our RL components
from .agents import MaskedDQNAgent, MaskedPPOAgent
from ..training.utilities import CatanatronRewardShaper

logger = logging.getLogger(__name__)

class CatanatronRLPlayer(Player):
    """RL Player that integrates with Catanatron game system"""
    
    def __init__(self, color: Color, agent_type: str = "dqn", 
                 model_path: Optional[str] = None, 
                 training_mode: bool = False,
                 device: str = "cpu"):
        super().__init__(color)
        
        self.agent_type = agent_type
        self.model_path = model_path
        self.training_mode = training_mode
        self.device = device
        
        # Decision time tracking for performance evaluation
        self.decision_times = []
        
        # These will be set when environment is available
        self.agent = None
        self.obs_dim = None
        self.action_dim = None
          # For training
        self.last_observation = None
        self.last_action = None
        self.last_valid_actions = None
        
    def initialize_agent(self, obs_dim: int, action_dim: int):
        """Initialize the RL agent with proper dimensions"""
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        if self.agent_type == "dqn":
            self.agent = MaskedDQNAgent(obs_dim, action_dim, device=self.device)
        elif self.agent_type == "ppo":
            self.agent = MaskedPPOAgent(obs_dim, action_dim, device=self.device)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
        
        # Load pre-trained model if provided
        if self.model_path:
            self.agent.load(self.model_path)
            logger.info(f"Loaded pre-trained model from {self.model_path}")
    
    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        """Decide which action to take given the current game state"""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call initialize_agent() first.")
        
        # Start timing for decision latency measurement
        start_time = time.time()
        
        # Convert game state to observation
        observation = self._game_to_observation(game)
        
        # Convert actions to indices
        valid_action_indices = self._actions_to_indices(playable_actions)
        
        # Store for training
        if self.training_mode:
            self.last_observation = observation
            self.last_valid_actions = valid_action_indices
        
        # Get action from agent
        if self.agent_type == "dqn":
            action_index = self.agent.select_action(observation, valid_action_indices)
        elif self.agent_type == "ppo":
            action_index, log_prob, value = self.agent.select_action(observation, valid_action_indices)
            # Store additional info for PPO training
            if self.training_mode:
                self.last_log_prob = log_prob
                self.last_value = value
        
        # Store last action for training
        if self.training_mode:
            self.last_action = action_index
        
        # Convert back to Action object
        selected_action = self._index_to_action(action_index, playable_actions)
        
        # Record decision time in milliseconds (for comparison with 14.12ms Minimax benchmark)
        decision_time_ms = (time.time() - start_time) * 1000
        self.decision_times.append(decision_time_ms)
        
        return selected_action
    
    def average_decision_time(self):
        """Get average decision time in milliseconds"""
        if not self.decision_times:
            return 0.0
        return sum(self.decision_times) / len(self.decision_times)
    
    def reset_decision_times(self):
        """Reset decision time tracking"""
        self.decision_times = []
    
    def _game_to_observation(self, game: Game) -> np.ndarray:
        """Convert game state to observation vector"""
        # This is a placeholder - you'll need to implement based on Catanatron's feature extraction
        # The actual implementation should use Catanatron's feature creation functions
        
        try:
            from catanatron.features import create_sample_vector
            from catanatron.features import get_feature_ordering
            
            # Get features for this player's perspective
            feature_vector = create_sample_vector(game, self.color)
            return np.array(feature_vector, dtype=np.float32)
            
        except ImportError:
            # Fallback: create dummy observation
            logger.warning("Could not import Catanatron features, using dummy observation")
            return np.random.random(614).astype(np.float32)
    
    def _actions_to_indices(self, actions: List[Action]) -> List[int]:
        """Convert Action objects to indices"""
        # This is a placeholder - you'll need to implement based on Catanatron's action space
        try:
            from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY
            
            indices = []
            for action in actions:
                # Normalize action for lookup
                normalized_action = self._normalize_action(action)
                key = (normalized_action.action_type, normalized_action.value)
                
                if key in ACTIONS_ARRAY:
                    indices.append(ACTIONS_ARRAY.index(key))
                else:
                    logger.warning(f"Action not found in action space: {action}")
            
            return indices
        except ImportError:
            # Fallback: return dummy indices
            logger.warning("Could not import action array, using dummy indices")
            return list(range(len(actions)))
    
    def _normalize_action(self, action: Action) -> Action:
        """Normalize action for consistent representation"""
        # Based on the existing reinforcement.py implementation
        normalized = action
        if normalized.action_type == ActionType.ROLL:
            return Action(action.color, action.action_type, None)
        elif normalized.action_type == ActionType.MOVE_ROBBER:
            return Action(action.color, action.action_type, action.value[0])
        elif normalized.action_type == ActionType.BUILD_ROAD:
            return Action(action.color, action.action_type, tuple(sorted(action.value)))
        elif normalized.action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return Action(action.color, action.action_type, None)
        elif normalized.action_type == ActionType.DISCARD:
            return Action(action.color, action.action_type, None)
        
        return normalized
    
    def _index_to_action(self, index: int, valid_actions: List[Action]) -> Action:
        """Convert action index back to Action object"""
        try:
            from catanatron.gym.envs.catanatron_env import ACTIONS_ARRAY
            
            if index < len(ACTIONS_ARRAY):
                action_type, value = ACTIONS_ARRAY[index]
                # Find corresponding action in valid_actions
                for action in valid_actions:
                    normalized = self._normalize_action(action)
                    if (normalized.action_type, normalized.value) == (action_type, value):
                        return action
            
            # Fallback: return first valid action
            logger.warning(f"Could not find action for index {index}, using first valid action")
            return valid_actions[0]
            
        except ImportError:
            # Fallback: use index directly
            if index < len(valid_actions):
                return valid_actions[index]
            else:
                return valid_actions[0]
    
    def get_training_data(self, reward: float, next_observation: np.ndarray, 
                         next_valid_actions: List[int], done: bool) -> Dict[str, Any]:
        """Get training data for the last step (for use in training loop)"""
        if not self.training_mode or self.last_observation is None:
            return {}
        
        data = {
            'observation': self.last_observation,
            'action': self.last_action,
            'reward': reward,
            'next_observation': next_observation,
            'next_valid_actions': next_valid_actions,
            'done': done
        }
        
        # Add PPO-specific data
        if self.agent_type == "ppo" and hasattr(self, 'last_log_prob'):
            data['log_prob'] = self.last_log_prob
            data['value'] = self.last_value
        
        return data
    
    def update_agent(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update the agent with a batch of experiences"""
        if self.agent is None or not self.training_mode:
            return {}
        
        return self.agent.update(batch)
    
    def save_model(self, path: str):
        """Save the agent model"""
        if self.agent is not None:
            self.agent.save(path)
    
    def load_model(self, path: str):
        """Load the agent model"""
        if self.agent is not None:
            self.agent.load(path)
    
    def set_training_mode(self, training: bool):
        """Set training mode"""
        self.training_mode = training
    
    def set_epsilon(self, epsilon: float):
        """Set exploration epsilon (for DQN)"""
        if self.agent_type == "dqn" and self.agent is not None:
            self.agent.epsilon = epsilon
    
    def __repr__(self):
        return f"CatanatronRLPlayer({self.color}, {self.agent_type})"

class CatanatronRLEnvironmentWrapper:
    """Wrapper to integrate RL player with Catanatron environment"""
    
    def __init__(self, env_config: Dict[str, Any] = None):
        import catanatron.gym
        
        self.env_config = env_config or {}
        # Pass the config as the config parameter, not as kwargs
        self.env = gym.make("catanatron/Catanatron-v0", config=self.env_config)
        
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # For reward shaping
        self.reward_shaper = CatanatronRewardShaper()
        self.last_victory_points = 0
        
    def reset(self):
        """Reset environment and return initial observation"""
        observation, info = self.env.reset()
        self.last_victory_points = 0  # Reset VP tracking
        return observation, info
    
    def step(self, action: int):
        """Take a step in the environment"""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward shaping
        shaped_reward = self._shape_reward(reward, observation, terminated)
        
        return observation, shaped_reward, terminated, truncated, info
    
    def _shape_reward(self, base_reward: float, observation: np.ndarray, done: bool) -> float:
        """Apply reward shaping to improve learning"""
        # Extract victory points from observation (this depends on Catanatron's feature structure)
        # For now, use base reward
        shaped_reward = base_reward
        
        # Add small rewards for game progress
        if not done:
            shaped_reward += 0.01  # Small reward for staying in game
        
        return shaped_reward
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def get_valid_actions(self) -> List[int]:
        """Get valid actions from the environment"""
        return self.env.unwrapped.get_valid_actions()

# Factory function to create different types of RL players
def create_rl_player(color: Color, agent_type: str = "dqn", 
                    model_path: Optional[str] = None,
                    training_mode: bool = False,
                    **kwargs) -> CatanatronRLPlayer:
    """Factory function to create RL players"""
    
    player = CatanatronRLPlayer(
        color=color,
        agent_type=agent_type,
        model_path=model_path,
        training_mode=training_mode,
        **kwargs
    )
    
    return player

if __name__ == "__main__":
    print("=== Catanatron RL Player Test ===")
      # Test player creation
    from catanatron.models.player import Color
    
    # Create DQN player
    dqn_player = create_rl_player(Color.RED, agent_type="dqn", training_mode=True)
    print(f"Created DQN player: {dqn_player}")
    
    # Create PPO player
    ppo_player = create_rl_player(Color.BLUE, agent_type="ppo", training_mode=True)
    print(f"Created PPO player: {ppo_player}")
    
    # Test environment wrapper
    try:
        env_wrapper = CatanatronRLEnvironmentWrapper()
        print(f"Environment wrapper created:")
        print(f"  Observation dim: {env_wrapper.obs_dim}")
        print(f"  Action dim: {env_wrapper.action_dim}")
        
        # Initialize players with environment dimensions
        dqn_player.initialize_agent(env_wrapper.obs_dim, env_wrapper.action_dim)
        ppo_player.initialize_agent(env_wrapper.obs_dim, env_wrapper.action_dim)
        
        print("Players initialized successfully!")
        
        env_wrapper.close()
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        print("This is expected if Catanatron gym is not properly installed")
