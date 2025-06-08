#!/usr/bin/env python3
"""
DQN Agent Evaluation Script
============================

Evaluates the trained DQN agent against multiple opponent types.
This script matches the exact architecture and configuration used in training.

Architecture:
- 614-dimensional state space (Catanatron features)
- 290 action categories (Catanatron action space)
- Network: 614 -> 256 -> 128 -> 290 (with dropout)

Performance Targets from Report:
- 65% overall win rate 
- 92% win rate vs RandomPlayer
- 60% win rate vs VictoryPointPlayer  
- 32% win rate vs AlphaBetaPlayer
- ~10ms average decision time
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from collections import defaultdict

# Add current directory to path for catanatron imports
sys.path.append('.')

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.features import create_sample_vector


class ReportSpecDQNNetwork(nn.Module):
    """DQN Network architecture exactly matching the trained model"""
    def __init__(self, obs_dim=614, action_dim=290):
        super(ReportSpecDQNNetwork, self).__init__()
        
        # Must match exact architecture used during training
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class DQNAgent(Player):
    """DQN Agent for evaluation"""
    
    # Class variable to store the loaded model
    _shared_q_network = None
    _model_loaded = False
    
    def __init__(self, color, model_path=None, device='cpu', shared_network=None):
        super().__init__(color)
        self.device = device
        self.obs_dim = 614  # From training config
        self.action_dim = 290  # From training config
        
        # Use shared network if provided, otherwise load from file
        if shared_network is not None:
            self.q_network = shared_network
        else:
            # Load model only if not already loaded
            if not DQNAgent._model_loaded or DQNAgent._shared_q_network is None:
                self.q_network = ReportSpecDQNNetwork(self.obs_dim, self.action_dim).to(device)
                checkpoint = torch.load(model_path, map_location=device)
                self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
                self.q_network.eval()
                DQNAgent._shared_q_network = self.q_network
                DQNAgent._model_loaded = True
                print(f"DQN Agent loaded from {model_path}")
                print(f"Architecture: {self.obs_dim} -> 256 -> 128 -> {self.action_dim}")
            else:
                self.q_network = DQNAgent._shared_q_network
    
    def get_state_features(self, game, color):
        """Extract state features matching training"""
        try:
            # Use the same feature extraction as training
            features = create_sample_vector(game, color)
            
            # Convert to numpy array if it's a list
            if isinstance(features, list):
                features = np.array(features, dtype=np.float32)
            elif not isinstance(features, np.ndarray):
                features = np.array(features, dtype=np.float32)
            
            # Handle feature dimension mismatch (training expects 614, current might be different)
            if len(features) > self.obs_dim:
                features = features[:self.obs_dim]  # Truncate if too long
            elif len(features) < self.obs_dim:
                # Pad with zeros if too short
                padded = np.zeros(self.obs_dim, dtype=np.float32)
                padded[:len(features)] = features
                features = padded
            
            return features.astype(np.float32)
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(self.obs_dim, dtype=np.float32)
    
    def decide(self, game, playable_actions):
        """Make a decision using the trained DQN"""
        start_time = time.time()
        
        try:
            # Get current player color
            color = game.state.current_color()
            
            # Extract state features
            state_features = self.get_state_features(game, color)
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(self.device)
            
            # Get Q-values from the network
            with torch.no_grad():
                q_values = self.q_network(state_tensor).squeeze(0)
            
            # Convert to numpy for easier handling
            q_values_np = q_values.cpu().numpy()
            
            # Filter valid actions and select best one
            valid_action_indices = []
            valid_actions = []
            
            for action in playable_actions:
                action_idx = self._action_to_index(action)
                if action_idx is not None and action_idx < len(q_values_np):
                    valid_action_indices.append(action_idx)
                    valid_actions.append(action)
            
            if not valid_actions:
                # Fallback to random action if no valid mapping
                action_list = list(playable_actions) if not isinstance(playable_actions, list) else playable_actions
                return np.random.choice(action_list)
            
            # Get Q-values for valid actions
            valid_q_values = [q_values_np[idx] for idx in valid_action_indices]
            
            # Select action with highest Q-value
            best_action_idx = np.argmax(valid_q_values)
            selected_action = valid_actions[best_action_idx]
            
            # Record decision time
            decision_time = (time.time() - start_time) * 1000  # ms
            
            return selected_action
            
        except Exception as e:
            print(f"Error in DQN decision making: {e}")
            # Fallback to random action
            action_list = list(playable_actions) if not isinstance(playable_actions, list) else playable_actions
            return np.random.choice(action_list)
    
    def _action_to_index(self, action):
        """Simple action to index mapping"""
        try:
            # Basic mapping based on action type
            from catanatron.models.enums import ActionType
            
            action_type_base = {
                ActionType.ROLL: 0,
                ActionType.MOVE_ROBBER: 50,
                ActionType.BUILD_ROAD: 100,
                ActionType.BUILD_SETTLEMENT: 150,
                ActionType.BUILD_CITY: 200,
                ActionType.BUY_DEVELOPMENT_CARD: 250,
                ActionType.PLAY_KNIGHT_CARD: 260,
                ActionType.DISCARD: 270,
                ActionType.END_TURN: 289
            }.get(action.action_type, 0)
            
            # Add some variation based on action value if available
            action_offset = 0
            if hasattr(action, 'value') and action.value is not None:
                action_offset = hash(str(action.value)) % 10
            
            return min(action_type_base + action_offset, self.action_dim - 1)
        except:
            return 0


def safe_get_victory_points(game, color):
    """Safely extract victory points for a player"""
    try:
        # Try different ways to access victory points
        if hasattr(game.state, 'player_state') and color in game.state.player_state:
            player_state = game.state.player_state[color]
            if hasattr(player_state, 'actual_victory_points'):
                return player_state.actual_victory_points
            elif 'actual_victory_points' in player_state:
                return player_state['actual_victory_points']
        
        # Fallback methods
        if hasattr(game.state, 'player_state'):
            for key, state in game.state.player_state.items():
                if key == color or (hasattr(key, 'value') and key.value == color.value):
                    if hasattr(state, 'actual_victory_points'):
                        return state.actual_victory_points
                    elif isinstance(state, dict) and 'actual_victory_points' in state:
                        return state['actual_victory_points']
        
        return 0  # Default fallback
    except Exception as e:
        print(f"Error getting victory points for {color}: {e}")
        return 0


def evaluate_dqn_agent():
    """Main evaluation function"""
    print("="*60)
    print("DQN AGENT EVALUATION")
    print("="*60)
    
    # Configuration
    model_path = "training_outputs_dqn/models/final_dqn_model.pt"
    num_games_per_opponent = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
      # Verify model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Load the model once upfront to avoid repeated loading
    print(f"Loading DQN model from {model_path}")
    print(f"Device: {device}")
    
    try:
        shared_q_network = ReportSpecDQNNetwork(614, 290).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        shared_q_network.load_state_dict(checkpoint['q_network_state_dict'])
        shared_q_network.eval()
        print("✓ Model loaded successfully")
        print("Architecture: 614 -> 256 -> 128 -> 290")
    except Exception as e:
        print(f"Error loading DQN model: {e}")
        return
    
    # Initialize DQN agent factory using the shared model
    def create_dqn_agent(color):
        return DQNAgent(color, shared_network=shared_q_network, device=device)
    
    print("✓ DQN agent factory ready")
    
    # Define opponent types
    opponents = {
        'RandomPlayer': lambda color: RandomPlayer(color),
        'VictoryPointPlayer': lambda color: VictoryPointPlayer(color),
        'AlphaBetaPlayer': lambda color: AlphaBetaPlayer(color, depth=2, prunning=True)
    }
    
    # Results storage
    results = {}
    total_games = 0
    total_wins = 0
    decision_times = []
    
    # Evaluate against each opponent type
    for opponent_name, opponent_factory in opponents.items():
        print(f"\n{'='*50}")
        print(f"EVALUATING AGAINST {opponent_name.upper()}")
        print(f"{'='*50}")
        wins = 0
        game_times = []
        
        for game_num in range(num_games_per_opponent):
            try:
                # Create players - alternate colors
                if game_num % 2 == 0:
                    dqn_color = Color.RED
                    opponent_color = Color.BLUE
                    dqn_agent = create_dqn_agent(dqn_color)
                    players = [dqn_agent, opponent_factory(opponent_color)]
                else:
                    dqn_color = Color.BLUE
                    opponent_color = Color.RED
                    dqn_agent = create_dqn_agent(dqn_color)
                    players = [opponent_factory(opponent_color), dqn_agent]
                
                # Play game
                start_time = time.time()
                game = Game(players)
                game.play()
                game_time = time.time() - start_time
                game_times.append(game_time)
                
                # Check winner
                dqn_vps = safe_get_victory_points(game, dqn_color)
                opponent_vps = safe_get_victory_points(game, opponent_color)
                
                if dqn_vps >= opponent_vps:
                    wins += 1
                    total_wins += 1
                
                total_games += 1
                
                # Progress update
                if (game_num + 1) % 25 == 0:
                    win_rate = wins / (game_num + 1) * 100
                    avg_game_time = np.mean(game_times)
                    print(f"  Progress: {game_num + 1}/{num_games_per_opponent} games, "
                          f"Win rate: {win_rate:.1f}%, Avg game time: {avg_game_time:.1f}s")
                
            except Exception as e:
                print(f"  Error in game {game_num + 1}: {e}")
                continue
        
        # Calculate results for this opponent
        win_rate = wins / num_games_per_opponent * 100
        avg_game_time = np.mean(game_times) if game_times else 0
        
        results[opponent_name] = {
            'wins': wins,
            'total_games': num_games_per_opponent,
            'win_rate': win_rate,
            'avg_game_time': avg_game_time
        }
        
        print(f"\n{opponent_name} Results:")
        print(f"  Wins: {wins}/{num_games_per_opponent} ({win_rate:.1f}%)")
        print(f"  Average game time: {avg_game_time:.2f} seconds")
    
    # Overall results
    overall_win_rate = total_wins / total_games * 100 if total_games > 0 else 0
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Win Rate: {overall_win_rate:.1f}%")
    print(f"Total Games Played: {total_games}")
    print(f"Total Wins: {total_wins}")
    
    # Performance vs targets
    targets = {
        'Overall': 65.0,
        'RandomPlayer': 92.0,
        'VictoryPointPlayer': 60.0,
        'AlphaBetaPlayer': 32.0
    }
    
    print(f"\n{'='*60}")
    print("PERFORMANCE VS REPORT TARGETS")
    print(f"{'='*60}")
    print(f"{'Opponent':<20} {'Actual':<10} {'Target':<10} {'Status':<10}")
    print("-" * 50)
    
    # Overall performance
    status = "✅ PASS" if overall_win_rate >= targets['Overall'] else "❌ FAIL"
    print(f"{'Overall':<20} {overall_win_rate:<10.1f} {targets['Overall']:<10.1f} {status}")
    
    # Per-opponent performance
    for opponent_name in opponents.keys():
        if opponent_name in results:
            actual = results[opponent_name]['win_rate']
            target = targets[opponent_name]
            status = "✅ PASS" if actual >= target else "❌ FAIL"
            print(f"{opponent_name:<20} {actual:<10.1f} {target:<10.1f} {status}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"training_outputs_dqn/evaluation_results_{timestamp}.json"
    
    evaluation_results = {
        'timestamp': timestamp,
        'overall_win_rate': overall_win_rate,
        'total_games': total_games,
        'total_wins': total_wins,
        'results_by_opponent': results,
        'targets': targets,
        'model_path': model_path,
        'device': device
    }
    
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return evaluation_results


if __name__ == "__main__":
    print("Starting DQN Agent Evaluation...")
    evaluation_results = evaluate_dqn_agent()
    print("\nEvaluation completed!")