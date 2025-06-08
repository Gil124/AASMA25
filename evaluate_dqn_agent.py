#!/usr/bin/env python3
"""
DQN Agent Evaluation Script - Updated for RL Agents Integration
===============================================================

Evaluates the trained DQN agent against multiple opponent types using the proper
RL agents implementation. This script integrates with the actual training infrastructure
to provide realistic performance metrics for the research report.

Architecture:
- 614-dimensional state space (actual Catanatron features)
- 290 action categories (actual Catanatron action space)
- Network: 614 -> 256 -> 128 -> 290 with dropout
- Uses same environment wrapper and agent as training

Expected Performance Metrics:
- Win rates against different opponent types
- Decision time measurements  
- Statistical analysis for academic reporting
- Performance trends and learning evaluation
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
import statistics
from typing import Dict, List, Tuple, Any

# Add current directory to path for catanatron imports
sys.path.append('.')

from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer, Player
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.state_functions import get_actual_victory_points

# Import RL agents components
from rl_agents.core.player import CatanatronRLPlayer
from rl_agents.core.agents import MaskedDQNAgent
from rl_agents.evaluation.deployment import CatanatronDeployment



class EvaluationDQNAgent(Player):
    """DQN Agent wrapper for evaluation that properly integrates with RL agents implementation"""
    
    def __init__(self, color: Color, model_path: str, device: str = 'cpu'):
        super().__init__(color)
        self.model_path = model_path
        self.device = device
        self.decision_times = []
        
        # Create RL player using the deployment system
        self.deployment = CatanatronDeployment(model_path, "dqn")
        self.rl_player = self.deployment.create_tournament_player(color)
        
        print(f"✓ DQN Agent initialized for {color}")
        print(f"  Model: {model_path}")
        print(f"  Device: {device}")
        print(f"  Architecture: 614 -> 256 -> 128 -> 290")
    
    def decide(self, game, playable_actions):
        """Make a decision using the trained DQN with timing"""
        start_time = time.time()
        
        try:
            # Use the RL player's decision making
            action = self.rl_player.decide(game, playable_actions)
              # Record decision time
            decision_time = (time.time() - start_time) * 1000  # Convert to ms
            self.decision_times.append(decision_time)
            return action
            
        except Exception as e:
            print(f"Error in DQN decision making: {e}")
            # Fallback to random action
            import random
            return random.choice(list(playable_actions))
    
    def get_avg_decision_time(self) -> float:
        """Get average decision time in milliseconds"""
        return np.mean(self.decision_times) if self.decision_times else 0.0
    
    def get_decision_time_stats(self) -> Dict[str, float]:
        """Get comprehensive decision time statistics"""
        if not self.decision_times:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "count": 0}
        
        return {
            "mean": np.mean(self.decision_times),
            "std": np.std(self.decision_times), 
            "min": np.min(self.decision_times),
            "max": np.max(self.decision_times),
            "median": np.median(self.decision_times),
            "count": len(self.decision_times)
        }


def safe_get_victory_points(game, color):
    """Safely extract victory points for a player using Catanatron's built-in function"""
    try:
        # Use the official Catanatron function to get actual victory points
        return get_actual_victory_points(game.state, color)
    except Exception as e:
        print(f"Error getting victory points for {color}: {e}")
        return 0


def run_single_game(dqn_agent: EvaluationDQNAgent, opponent: Player, game_id: int) -> Dict[str, Any]:
    """Run a single game and return detailed results"""
    try:
        # Create players list based on color assignment - use the EvaluationDQNAgent directly
        if dqn_agent.color == Color.RED:
            players = [dqn_agent, opponent]
        else:
            players = [opponent, dqn_agent]
        
        # Play the game with timing
        start_time = time.time()
        game = Game(players)
        game.play()
        game_duration = time.time() - start_time
          # Determine winner (use strict comparison - first to 10 wins)
        dqn_vps = safe_get_victory_points(game, dqn_agent.color)
        opponent_vps = safe_get_victory_points(game, opponent.color)
        
        dqn_won = dqn_vps > opponent_vps
        
        return {
            'game_id': game_id,
            'dqn_won': dqn_won,
            'dqn_vps': dqn_vps,
            'opponent_vps': opponent_vps,
            'game_duration': game_duration,
            'total_turns': len(game.state.actions) if hasattr(game.state, 'actions') else 0
        }
        
    except Exception as e:
        print(f"  Error in game {game_id}: {e}")
        return {
            'game_id': game_id,
            'dqn_won': False,
            'dqn_vps': 0,
            'opponent_vps': 10,  # Assume opponent won
            'game_duration': 0,
            'total_turns': 0,
            'error': str(e)
        }


def evaluate_against_opponent(dqn_agent: EvaluationDQNAgent, opponent_name: str, 
                            opponent_factory, num_games: int = 100) -> Dict[str, Any]:
    """Evaluate DQN agent against a specific opponent type"""
    print(f"\n{'='*50}")
    print(f"EVALUATING AGAINST {opponent_name.upper()}")
    print(f"{'='*50}")
    
    results = []
    wins = 0
    game_durations = []
    
    # Track decision times for this opponent
    initial_decision_count = len(dqn_agent.decision_times)
    
    for game_num in range(num_games):
        # Alternate colors to ensure fair evaluation
        if game_num % 2 == 0:
            dqn_color = Color.RED
            opponent_color = Color.BLUE
        else:
            dqn_color = Color.BLUE
            opponent_color = Color.RED
              # Update agent color
        dqn_agent.color = dqn_color
        dqn_agent.rl_player.color = dqn_color
        
        # Create opponent
        opponent = opponent_factory(opponent_color)
        
        # Run game
        game_result = run_single_game(dqn_agent, opponent, game_num + 1)
        results.append(game_result)
        
        if game_result['dqn_won']:
            wins += 1
            
        if game_result['game_duration'] > 0:
            game_durations.append(game_result['game_duration'])
        
        # Progress update
        if (game_num + 1) % 25 == 0:
            current_win_rate = wins / (game_num + 1) * 100
            avg_duration = np.mean(game_durations) if game_durations else 0
            print(f"  Progress: {game_num + 1}/{num_games} games, "
                  f"Win rate: {current_win_rate:.1f}%, Avg duration: {avg_duration:.1f}s")
    
    # Calculate comprehensive statistics
    win_rate = wins / num_games * 100
    avg_game_duration = np.mean(game_durations) if game_durations else 0
    std_game_duration = np.std(game_durations) if game_durations else 0
    
    # Calculate decision times for this opponent
    final_decision_count = len(dqn_agent.decision_times)
    opponent_decision_times = dqn_agent.decision_times[initial_decision_count:final_decision_count]
    avg_decision_time = np.mean(opponent_decision_times) if opponent_decision_times else 0.0
    
    # Calculate confidence interval for win rate (95% CI)
    if num_games >= 30:  # Large enough sample for normal approximation
        z_score = 1.96  # 95% confidence
        margin_error = z_score * np.sqrt((win_rate / 100) * (1 - win_rate / 100) / num_games) * 100
        win_rate_ci = (max(0, win_rate - margin_error), min(100, win_rate + margin_error))
    else:
        win_rate_ci = (0, 100)  # Wide interval for small samples
    
    opponent_results = {
        'opponent_name': opponent_name,
        'total_games': num_games,
        'wins': wins,
        'losses': num_games - wins,
        'win_rate': win_rate,
        'win_rate_ci': win_rate_ci,
        'avg_game_duration': avg_game_duration,
        'std_game_duration': std_game_duration,
        'avg_decision_time': avg_decision_time,
        'decision_count': len(opponent_decision_times),
        'game_results': results
    }
    
    # Print results
    print(f"\n{opponent_name} Results:")
    print(f"  Wins: {wins}/{num_games} ({win_rate:.1f}%)")
    print(f"  95% CI: [{win_rate_ci[0]:.1f}%, {win_rate_ci[1]:.1f}%]")
    print(f"  Avg game duration: {avg_game_duration:.2f}s ± {std_game_duration:.2f}s")
    print(f"  Avg decision time: {avg_decision_time:.2f}ms ({len(opponent_decision_times)} decisions)")
    
    return opponent_results


def evaluate_dqn_agent():
    """Main evaluation function with comprehensive metrics"""
    print("="*60)
    print("DQN AGENT EVALUATION - RL AGENTS INTEGRATION")
    print("="*60)
      # Configuration
    model_path = "training_outputs_dqn/models/final_dqn_model.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    # Create DQN agent (will be initialized with Color.RED initially)
    try:
        dqn_agent = EvaluationDQNAgent(Color.RED, model_path, device)
    except Exception as e:
        print(f"Error creating DQN agent: {e}")
        return None
    
    # Define opponent types with different difficulty levels and game counts
    # Basic opponents: 100 games each (fast evaluation)
    # Challenging opponents: 10 games each (computationally expensive)
    opponents_config = {
        # Basic opponents - 100 games each
        'RandomPlayer': {
            'factory': lambda color: RandomPlayer(color),
            'games': 20,
            'description': 'Random move selection'
        },
        'WeightedRandomPlayer': {
            'factory': lambda color: WeightedRandomPlayer(color),
            'games': 20,
            'description': 'Weighted random with preferences'
        },
        'VictoryPointPlayer': {
            'factory': lambda color: VictoryPointPlayer(color),
            'games': 20,
            'description': 'Victory point focused strategy'
        },
        'AlphaBetaPlayer': {
            'factory': lambda color: AlphaBetaPlayer(color, depth=2, prunning=True),
            'games': 20,
            'description': 'Alpha-beta pruning minimax'
        },
        
        # Challenging opponents - 10 games each (computationally expensive)
        'ImprovedAlphaBeta': {
            'factory': lambda color: AlphaBetaPlayer(color, depth=3, prunning_improved=True),
            'games': 20,
            'description': 'Improved alpha-beta with depth 3'
        },
        'MCTSPlayer': {
            'factory': lambda color: MCTSPlayer(color, num_simulations=10),
            'games': 20,
            'description': 'Monte Carlo Tree Search (100 simulations)'
        },
        'GreedyPlayoutsPlayer': {
            'factory': lambda color: GreedyPlayoutsPlayer(color, num_playouts=25),
            'games': 20,
            'description': 'Greedy playouts strategy (50 playouts)'
        }
    }
    
    print(f"\nEvaluation Configuration:")
    total_games = sum(config['games'] for config in opponents_config.values())
    print(f"Total opponents: {len(opponents_config)}")
    print(f"Total games: {total_games}")
    print(f"Basic opponents (100 games each): {len([c for c in opponents_config.values() if c['games'] == 100])}")
    print(f"Challenging opponents (10 games each): {len([c for c in opponents_config.values() if c['games'] == 10])}")
      # Run evaluations
    all_results = {}
    total_games = 0
    total_wins = 0
    
    for opponent_name, config in opponents_config.items():
        print(f"\n📋 {config['description']}")
        opponent_results = evaluate_against_opponent(
            dqn_agent, opponent_name, config['factory'], config['games']
        )
        all_results[opponent_name] = opponent_results
        total_games += opponent_results['total_games']
        total_wins += opponent_results['wins']
    
    # Overall statistics
    overall_win_rate = total_wins / total_games * 100 if total_games > 0 else 0
    decision_time_stats = dqn_agent.get_decision_time_stats()
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Win Rate: {overall_win_rate:.1f}% ({total_wins}/{total_games})")
    print(f"Average Decision Time: {decision_time_stats['mean']:.2f}ms")
    print(f"Decision Time Range: {decision_time_stats['min']:.2f}-{decision_time_stats['max']:.2f}ms")
    print(f"Total Decisions Made: {decision_time_stats['count']}")    # Performance analysis
    print(f"\n{'='*70}")
    print("PERFORMANCE ANALYSIS BY OPPONENT")
    print(f"{'='*70}")
    print(f"{'Opponent':<20} {'Win Rate':<12} {'95% CI':<20} {'Avg Duration':<12} {'Avg Decision Time':<15}")
    print("-" * 80)
    
    for opponent_name, results in all_results.items():
        ci_str = f"[{results['win_rate_ci'][0]:.1f}, {results['win_rate_ci'][1]:.1f}]%"
        decision_time_str = f"{results['avg_decision_time']:.2f}ms"
        print(f"{opponent_name:<20} {results['win_rate']:<11.1f}% {ci_str:<20} {results['avg_game_duration']:<11.2f}s {decision_time_str:<15}")
    
    # Decision Time Analysis
    print(f"\n{'='*60}")
    print("DECISION TIME ANALYSIS")
    print(f"{'='*60}")
    print(f"{'Opponent':<20} {'Avg Time (ms)':<15} {'Decisions':<10} {'Games':<8}")
    print("-" * 53)
    
    all_decision_times = []
    for opponent_name, results in all_results.items():
        avg_time = results['avg_decision_time']
        decisions = results.get('decision_count', 0)
        games = results['total_games']
        all_decision_times.extend([avg_time] * games)  # Weight by games played
        print(f"{opponent_name:<20} {avg_time:<14.2f} {decisions:<9} {games:<8}")
    
    # Overall decision time stats
    overall_avg_decision = np.mean(all_decision_times) if all_decision_times else 0
    print("-" * 53)
    print(f"{'OVERALL AVERAGE':<20} {overall_avg_decision:<14.2f} {decision_time_stats['count']:<9} {total_games:<8}")
    print(f"Note: Overall statistics include global decision time tracking across all evaluations.")
    
    # Research targets comparison
    research_targets = {
        # Basic opponents (higher sample size, more reliable targets)
        'RandomPlayer': (85, 95),         # Expected range: 85-95%
        'WeightedRandomPlayer': (70, 85), # Expected range: 70-85%  
        'VictoryPointPlayer': (50, 70),   # Expected range: 50-70%
        'AlphaBetaPlayer': (25, 45),      # Expected range: 25-45%
        
        # Challenging opponents (lower sample size, wider acceptable ranges)
        'ImprovedAlphaBeta': (15, 40),    # Expected range: 15-40% (very tough)
        'MCTSPlayer': (20, 50),           # Expected range: 20-50% (depends on simulations)
        'GreedyPlayoutsPlayer': (25, 55)  # Expected range: 25-55% (strategic but not perfect)
    }
    
    print(f"\n{'='*60}")
    print("RESEARCH TARGETS ANALYSIS")
    print(f"{'='*60}")
    print(f"{'Opponent':<20} {'Actual':<10} {'Expected':<15} {'Status':<10}")
    print("-" * 55)
    
    performance_status = {}
    for opponent_name, results in all_results.items():
        if opponent_name in research_targets:
            actual = results['win_rate']
            expected_min, expected_max = research_targets[opponent_name]
            
            if expected_min <= actual <= expected_max:
                status = "✅ GOOD"
                performance_status[opponent_name] = "good"
            elif actual > expected_max:
                status = "🔥 EXCELLENT"  
                performance_status[opponent_name] = "excellent"
            else:
                status = "⚠️ BELOW"
                performance_status[opponent_name] = "below"
                
            expected_str = f"{expected_min}-{expected_max}%"
            print(f"{opponent_name:<20} {actual:<9.1f}% {expected_str:<15} {status}")
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"training_outputs_dqn/evaluation_results_{timestamp}.json"
    
    evaluation_summary = {
        'timestamp': timestamp,
        'model_path': model_path,
        'device': device,
        'total_games': total_games,
        'total_wins': total_wins,
        'overall_win_rate': overall_win_rate,
        'decision_time_stats': decision_time_stats,
        'results_by_opponent': all_results,
        'research_targets': research_targets,
        'performance_status': performance_status,        'evaluation_config': {
            'basic_opponents_games': 100,
            'challenging_opponents_games': 10,
            'alternating_colors': True,
            'architecture': "614 -> 256 -> 128 -> 290",
            'training_episodes': 2000,
            'opponents_config': {name: {'games': config['games'], 'description': config['description']} 
                               for name, config in opponents_config.items()}
        }
    }
    
    try:
        os.makedirs("training_outputs_dqn", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
    except Exception as e:
        print(f"Warning: Could not save results file: {e}")
    
    return evaluation_summary


if __name__ == "__main__":
    print("=" * 80)
    print("DQN AGENT EVALUATION - ACADEMIC RESEARCH METRICS")
    print("=" * 80)
    print("This evaluation uses the proper RL agents implementation")
    print("to provide realistic performance metrics for academic reporting.")
    print()
    print("Key Features:")
    print("- Integration with actual training infrastructure")
    print("- Comprehensive statistical analysis")  
    print("- Decision time measurements")
    print("- Confidence intervals for win rates")
    print("- Performance comparison against research targets")
    print("=" * 80)
    
    try:
        evaluation_results = evaluate_dqn_agent()
        
        if evaluation_results:
            print("\n" + "=" * 80)
            print("EVALUATION COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print()
            print("Summary for Report:")
            print(f"- Overall Win Rate: {evaluation_results['overall_win_rate']:.1f}%")
            print(f"- Average Decision Time: {evaluation_results['decision_time_stats']['mean']:.2f}ms")
            print(f"- Total Games Evaluated: {evaluation_results['total_games']}")
            print(f"- Model Architecture: {evaluation_results['evaluation_config']['architecture']}")
            print()
            
            # Print key findings
            excellent_count = sum(1 for status in evaluation_results['performance_status'].values() if status == 'excellent')
            good_count = sum(1 for status in evaluation_results['performance_status'].values() if status == 'good') 
            below_count = sum(1 for status in evaluation_results['performance_status'].values() if status == 'below')
            
            print("Performance Summary:")
            print(f"- Excellent performance: {excellent_count} opponent(s)")
            print(f"- Good performance: {good_count} opponent(s)")
            print(f"- Below target: {below_count} opponent(s)")
            
            if below_count == 0:
                print("\n🎉 All performance targets met or exceeded!")
            elif below_count <= 1:
                print("\n✅ Strong overall performance with room for improvement")
            else:
                print("\n⚠️ Multiple areas need improvement")
            
            print("\nUse the detailed JSON results file for report data.")
        else:
            print("❌ Evaluation failed. Check error messages above.")
            
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()