"""
Step 3: Define RL Agent Architecture Options
This module provides different RL agent architectures suitable for Catanatron.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CatanatronRLAgent(ABC):
    """Base class for all Catanatron RL agents"""
    
    def __init__(self, obs_dim: int, action_dim: int, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
    @abstractmethod
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        """Select an action given observation and valid actions"""
        pass
    
    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update agent with a batch of experience"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent to file"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent from file"""
        pass

class DQNNetwork(nn.Module):
    """Deep Q-Network for Catanatron"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [256, 128]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MaskedDQNAgent(CatanatronRLAgent):
    """DQN Agent with action masking for invalid actions"""
    
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4, 
                 gamma: float = 0.99, epsilon: float = 0.1, device: str = "cpu"):
        super().__init__(obs_dim, action_dim, device)
        
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Networks
        self.q_network = DQNNetwork(obs_dim, action_dim).to(self.device)
        self.target_q_network = DQNNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        self.update_counter = 0
        self.target_update_freq = 1000  # Update target network every 1000 steps
        
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> int:
        """Select action using epsilon-greedy with action masking"""
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(obs_tensor).squeeze()
            
            # Mask invalid actions by setting their Q-values to negative infinity
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
        valid_actions_masks = batch['valid_actions_masks']  # List of lists
        
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
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
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
            'epsilon': self.epsilon        }, path)
        logger.info(f"Agent saved to {path}")
    
    def _convert_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert state dict keys from 'layers.*' to 'network.*' format"""
        converted_dict = {}
        for key, value in state_dict.items():
            if key.startswith('layers.'):
                # Convert 'layers.X.weight' to 'network.X.weight'
                new_key = key.replace('layers.', 'network.')
                converted_dict[new_key] = value
                logger.debug(f"Converted key: {key} -> {new_key}")
            else:
                converted_dict[key] = value
        return converted_dict

    def load(self, path: str):
        """Load agent state with compatibility for different model formats"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'q_network_state_dict' in checkpoint:
            # Standard RL agents format
            q_state_dict = checkpoint['q_network_state_dict']
            target_state_dict = checkpoint['target_q_network_state_dict']
            
            # Check if we need to convert keys from 'layers.*' to 'network.*'
            if any(key.startswith('layers.') for key in q_state_dict.keys()):
                logger.info("Converting state dict keys from 'layers.*' to 'network.*' format")
                q_state_dict = self._convert_state_dict_keys(q_state_dict)
                target_state_dict = self._convert_state_dict_keys(target_state_dict)
            
            self.q_network.load_state_dict(q_state_dict)
            self.target_q_network.load_state_dict(target_state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.update_counter = checkpoint.get('update_counter', 0)
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            
        else:
            # Handle other formats (e.g., just the model state dict)
            logger.warning("Unknown checkpoint format, attempting direct load")
            self.q_network.load_state_dict(checkpoint)
            self.target_q_network.load_state_dict(checkpoint)
        
        logger.info(f"Agent loaded from {path}")

class PPONetwork(nn.Module):
    """Policy and Value networks for PPO"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int] = [512, 256]):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        shared_layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*shared_layers)
        
        # Policy head
        self.policy_head = nn.Linear(prev_dim, action_dim)
        
        # Value head
        self.value_head = nn.Linear(prev_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shared_features = self.shared_net(x)
        logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return logits, value.squeeze()

class MaskedPPOAgent(CatanatronRLAgent):
    """PPO Agent with action masking"""
    
    def __init__(self, obs_dim: int, action_dim: int, lr: float = 3e-4,
                 gamma: float = 0.99, clip_ratio: float = 0.2, device: str = "cpu"):
        super().__init__(obs_dim, action_dim, device)
        
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        
        self.network = PPONetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
    def select_action(self, observation: np.ndarray, valid_actions: List[int]) -> Tuple[int, float, float]:
        """Select action and return action, log_prob, value"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            logits, value = self.network(obs_tensor)
            
            # Mask invalid actions
            masked_logits = logits.clone()
            mask = torch.ones(self.action_dim, dtype=torch.bool)
            mask[valid_actions] = False
            masked_logits[0][mask] = float('-inf')
            
            # Sample action
            probs = F.softmax(masked_logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
        return action.item(), log_prob.item(), value.item()
    
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update using PPO loss"""
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)
        valid_actions_masks = batch['valid_actions_masks']
        
        # Get current policy and value
        logits, values = self.network(observations)
        
        # Mask invalid actions
        for i, valid_actions in enumerate(valid_actions_masks):
            mask = torch.ones(self.action_dim, dtype=torch.bool)
            mask[valid_actions] = False
            logits[i][mask] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(probs)
        new_log_probs = action_dist.log_prob(actions)
        
        # PPO policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy = action_dist.entropy().mean()
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item()
        }
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"PPO Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"PPO Agent loaded from {path}")

def get_agent_config(agent_type: str) -> Dict[str, Any]:
    """Get default configuration for different agent types"""
    configs = {
        "dqn": {
            "lr": 3e-4,
            "gamma": 0.99,
            "epsilon": 0.1,
            "hidden_dims": [512, 256, 128],
            "target_update_freq": 1000,
            "batch_size": 32,
            "replay_buffer_size": 100000
        },
        "ppo": {
            "lr": 3e-4,
            "gamma": 0.99,
            "clip_ratio": 0.2,
            "hidden_dims": [512, 256],
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95
        }
    }
    return configs.get(agent_type, {})

if __name__ == "__main__":
    # Example usage
    print("=== RL Agent Architecture Options ===")
    
    # Example dimensions (will be determined from actual environment)
    obs_dim = 614  # From catanatron observation space
    action_dim = 290  # Correct action space size from training
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create DQN agent
    dqn_agent = MaskedDQNAgent(obs_dim, action_dim)
    print(f"DQN Agent created with {sum(p.numel() for p in dqn_agent.q_network.parameters())} parameters")
    
    # Create PPO agent
    ppo_agent = MaskedPPOAgent(obs_dim, action_dim)
    print(f"PPO Agent created with {sum(p.numel() for p in ppo_agent.network.parameters())} parameters")
    
    print("\nConfigurations available:")
    for agent_type in ["dqn", "ppo"]:
        config = get_agent_config(agent_type)
        print(f"{agent_type.upper()}: {config}")
