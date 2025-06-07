"""
Step 7: Evaluation and Deployment
This module provides comprehensive evaluation and deployment tools for trained RL agents.
"""
import numpy as np
import torch
import time
import os
import json
from typing import Dict, List, Any, Tuple, Optional
import logging
from collections import defaultdict

# Import Catanatron components
from catanatron.game import Game
from catanatron.models.player import Color
from catanatron.models.player import RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
try:
    from catanatron.players.value_function import ValueFunctionPlayer
    from catanatron.players.minimax import AlphaBetaPlayer
    HAS_ADVANCED_PLAYERS = True
except ImportError:
    ValueFunctionPlayer = None
    AlphaBetaPlayer = None
    HAS_ADVANCED_PLAYERS = False

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    plt = None
    sns = None
    HAS_VISUALIZATION = False

# Import our components
from ..core.player import CatanatronRLPlayer, create_rl_player
from ..training.pipeline import CatanatronTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CatanatronEvaluator:
    """Comprehensive evaluation system for Catanatron RL agents"""
    
    def __init__(self, model_path: str, agent_type: str = "dqn", device: str = "cpu"):
        self.model_path = model_path
        self.agent_type = agent_type
        self.device = device
        
        # Load the trained agent
        self.rl_player = None
        self._load_agent()
        
        # Evaluation results storage
        self.results = defaultdict(list)
        
    def _load_agent(self):
        """Load the trained RL agent"""
        logger.info(f"Loading {self.agent_type} agent from {self.model_path}")
        
        try:
            # Create RL player
            self.rl_player = create_rl_player(
                color=Color.RED,  # Will be changed during evaluation
                agent_type=self.agent_type,
                model_path=self.model_path,
                training_mode=False,
                device=self.device
            )
            
            # Initialize with dummy dimensions (will be properly set during evaluation)
            self.rl_player.initialize_agent(614, 4000)  # Approximate dimensions
            
            logger.info("RL agent loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load RL agent: {e}")
            raise
    
    def create_opponent_suite(self) -> Dict[str, Any]:
        """Create a suite of opponent players for evaluation"""
        opponents = {
            "Random": RandomPlayer,
            "WeightedRandom": WeightedRandomPlayer,
            "Greedy": GreedyPlayoutsPlayer,
        }
        
        # Add advanced players if available
        if ValueFunctionPlayer is not None:
            opponents["ValueFunction"] = ValueFunctionPlayer
        
        if AlphaBetaPlayer is not None:
            opponents["AlphaBeta"] = AlphaBetaPlayer
        
        return opponents
    
    def evaluate_against_opponent(self, opponent_class, opponent_name: str, 
                                num_games: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """Evaluate RL agent against a specific opponent"""
        if verbose:
            logger.info(f"Evaluating against {opponent_name} over {num_games} games...")
        
        wins = 0
        total_rewards = []
        game_lengths = []
        victory_points = []
        game_times = []
        
        for game_num in range(num_games):
            start_time = time.time()
            
            # Create players
            rl_player = create_rl_player(
                color=Color.RED,
                agent_type=self.agent_type,
                model_path=self.model_path,
                training_mode=False,
                device=self.device
            )
            rl_player.initialize_agent(614, 4000)
            
            # Create opponent players
            opponents = [
                opponent_class(Color.BLUE),
                opponent_class(Color.ORANGE),
                opponent_class(Color.WHITE)
            ]
            
            players = [rl_player] + opponents
            
            # Run game
            try:
                game = Game(players)
                game.play()
                
                # Collect results
                winner = game.winning_color()
                game_length = len(game.state.actions)
                game_time = time.time() - start_time
                
                # Get victory points for RL player
                rl_vp = game.state.player_state(f"p{Color.RED.value}")["public_victory_points"]
                
                if winner == Color.RED:
                    wins += 1
                    total_rewards.append(1.0)
                else:
                    total_rewards.append(-1.0)
                
                game_lengths.append(game_length)
                victory_points.append(rl_vp)
                game_times.append(game_time)
                
            except Exception as e:
                logger.warning(f"Game {game_num} failed: {e}")
                total_rewards.append(-1.0)
                game_lengths.append(0)
                victory_points.append(0)
                game_times.append(0)
        
        # Calculate statistics
        win_rate = wins / num_games
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(game_lengths)
        avg_vp = np.mean(victory_points)
        avg_time = np.mean(game_times)
        
        results = {
            "opponent": opponent_name,
            "win_rate": win_rate,
            "wins": wins,
            "total_games": num_games,
            "avg_reward": avg_reward,
            "avg_game_length": avg_length,
            "avg_victory_points": avg_vp,
            "avg_game_time": avg_time,
            "std_victory_points": np.std(victory_points),
            "min_victory_points": np.min(victory_points),
            "max_victory_points": np.max(victory_points)
        }
        
        if verbose:
            logger.info(f"Results vs {opponent_name}: "
                       f"Win rate={win_rate:.3f}, Avg VP={avg_vp:.1f}, "
                       f"Avg time={avg_time:.2f}s")
        
        return results
    
    def comprehensive_evaluation(self, num_games_per_opponent: int = 200) -> Dict[str, Any]:
        """Run comprehensive evaluation against all opponents"""
        logger.info("Starting comprehensive evaluation...")
        
        opponents = self.create_opponent_suite()
        all_results = {}
        
        for opponent_name, opponent_class in opponents.items():
            try:
                results = self.evaluate_against_opponent(
                    opponent_class, opponent_name, num_games_per_opponent
                )
                all_results[opponent_name] = results
                self.results[opponent_name] = results
                
            except Exception as e:
                logger.error(f"Evaluation against {opponent_name} failed: {e}")
                all_results[opponent_name] = {"error": str(e)}
        
        # Calculate overall performance
        overall_stats = self._calculate_overall_stats(all_results)
        all_results["Overall"] = overall_stats
        
        return all_results
    
    def _calculate_overall_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance statistics"""
        win_rates = []
        avg_vps = []
        
        for opponent, result in results.items():
            if "error" not in result:
                win_rates.append(result["win_rate"])
                avg_vps.append(result["avg_victory_points"])
        
        if not win_rates:
            return {"error": "No successful evaluations"}
        
        return {
            "avg_win_rate": np.mean(win_rates),
            "std_win_rate": np.std(win_rates),
            "min_win_rate": np.min(win_rates),
            "max_win_rate": np.max(win_rates),
            "avg_victory_points": np.mean(avg_vps),
            "opponents_beaten": sum(1 for wr in win_rates if wr > 0.5)
        }
    
    def benchmark_against_baselines(self) -> Dict[str, Any]:
        """Benchmark against known baseline performances"""
        # These are approximate baselines from the Catanatron documentation
        baselines = {
            "RandomPlayer": {"win_rate": 0.25, "avg_vp": 3.0},
            "WeightedRandomPlayer": {"win_rate": 0.30, "avg_vp": 4.0},
            "GreedyPlayer": {"win_rate": 0.35, "avg_vp": 5.0},
            "ValueFunctionPlayer": {"win_rate": 0.60, "avg_vp": 7.0},  # Strongest traditional AI
        }
        
        benchmark_results = {}
        
        for opponent_name, baseline in baselines.items():
            if opponent_name in self.results:
                our_result = self.results[opponent_name]
                
                benchmark_results[opponent_name] = {
                    "baseline_win_rate": baseline["win_rate"],
                    "our_win_rate": our_result["win_rate"],
                    "win_rate_improvement": our_result["win_rate"] - baseline["win_rate"],
                    "baseline_avg_vp": baseline["avg_vp"],
                    "our_avg_vp": our_result["avg_victory_points"],
                    "vp_improvement": our_result["avg_victory_points"] - baseline["avg_vp"]
                }
        
        return benchmark_results
    
    def save_results(self, save_path: str):
        """Save evaluation results to file"""
        results_dict = dict(self.results)
        
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
    
    def visualize_results(self, save_path: Optional[str] = None):
        """Create visualizations of evaluation results"""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Prepare data
        opponents = list(self.results.keys())
        win_rates = [self.results[opp]["win_rate"] for opp in opponents]
        avg_vps = [self.results[opp]["avg_victory_points"] for opp in opponents]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Win rates bar chart
        axes[0, 0].bar(opponents, win_rates, color='skyblue')
        axes[0, 0].set_title('Win Rates vs Different Opponents')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Win Rate')
        axes[0, 0].legend()
        
        # Victory points bar chart
        axes[0, 1].bar(opponents, avg_vps, color='lightgreen')
        axes[0, 1].set_title('Average Victory Points vs Different Opponents')
        axes[0, 1].set_ylabel('Average Victory Points')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Win rate vs VP scatter plot
        axes[1, 0].scatter(win_rates, avg_vps, s=100, alpha=0.7)
        for i, opp in enumerate(opponents):
            axes[1, 0].annotate(opp, (win_rates[i], avg_vps[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Win Rate')
        axes[1, 0].set_ylabel('Average Victory Points')
        axes[1, 0].set_title('Win Rate vs Victory Points')
        
        # Performance comparison (if baseline data available)
        try:
            baselines = self.benchmark_against_baselines()
            if baselines:
                baseline_names = list(baselines.keys())
                improvements = [baselines[name]["win_rate_improvement"] for name in baseline_names]
                
                colors = ['green' if imp > 0 else 'red' for imp in improvements]
                axes[1, 1].bar(baseline_names, improvements, color=colors)
                axes[1, 1].set_title('Win Rate Improvement vs Baselines')
                axes[1, 1].set_ylabel('Win Rate Improvement')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, 'Baseline comparison\nnot available', 
                           transform=axes[1, 1].transAxes, ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()

class CatanatronDeployment:
    """Deployment utilities for trained RL agents"""
    
    def __init__(self, model_path: str, agent_type: str = "dqn"):
        self.model_path = model_path
        self.agent_type = agent_type
        
    def create_tournament_player(self, color: Color) -> CatanatronRLPlayer:
        """Create a tournament-ready RL player"""
        player = create_rl_player(
            color=color,
            agent_type=self.agent_type,
            model_path=self.model_path,
            training_mode=False
        )
        
        # Initialize with proper dimensions
        player.initialize_agent(614, 4000)
        
        return player
    
    def run_tournament(self, opponents: List[Any], num_games: int = 100) -> Dict[str, Any]:
        """Run a tournament with the RL agent"""
        logger.info(f"Running tournament with {len(opponents)} opponents over {num_games} games")
        
        wins = 0
        games_played = 0
        detailed_results = []
        
        for game_num in range(num_games):
            try:
                # Create players
                rl_player = self.create_tournament_player(Color.RED)
                players = [rl_player] + opponents
                
                # Run game
                game = Game(players)
                game.play()
                
                # Record results
                winner = game.winning_color()
                if winner == Color.RED:
                    wins += 1
                
                detailed_results.append({
                    "game": game_num,
                    "winner": winner.value if winner else None,
                    "rl_won": winner == Color.RED,
                    "game_length": len(game.state.actions)
                })
                
                games_played += 1
                
            except Exception as e:
                logger.warning(f"Tournament game {game_num} failed: {e}")
        
        tournament_results = {
            "total_games": games_played,
            "wins": wins,
            "win_rate": wins / games_played if games_played > 0 else 0,
            "detailed_results": detailed_results
        }
        
        logger.info(f"Tournament completed: {wins}/{games_played} wins "
                   f"({tournament_results['win_rate']:.3f} win rate)")
        
        return tournament_results
    
    def export_for_cli(self, export_path: str):
        """Export model for use with Catanatron CLI"""
        # This would create a player class that can be used with the CLI
        cli_player_code = f'''
from catanatron.models.player import Player
from catanatron.models.enums import Color
from step5_rl_player import create_rl_player

class ExportedRLPlayer(Player):
    def __init__(self, color: Color):
        super().__init__(color)
        self.rl_player = create_rl_player(
            color=color,
            agent_type="{self.agent_type}",
            model_path="{self.model_path}",
            training_mode=False
        )
        self.rl_player.initialize_agent(614, 4000)
    
    def decide(self, game, playable_actions):
        return self.rl_player.decide(game, playable_actions)
'''
        
        with open(export_path, 'w') as f:
            f.write(cli_player_code)
        
        logger.info(f"CLI player exported to {export_path}")

def quick_evaluation(model_path: str, agent_type: str = "dqn", 
                    num_games: int = 100) -> Dict[str, Any]:
    """Quick evaluation function for trained models"""
    evaluator = CatanatronEvaluator(model_path, agent_type)
    
    # Run against Random and WeightedRandom opponents
    results = {}
    
    # Evaluate against Random player
    try:
        random_results = evaluator.evaluate_against_opponent(
            RandomPlayer, "Random", num_games
        )
        results["Random"] = random_results
    except Exception as e:
        logger.error(f"Failed to evaluate against Random: {e}")
    
    # Evaluate against WeightedRandom player
    try:
        weighted_results = evaluator.evaluate_against_opponent(
            WeightedRandomPlayer, "WeightedRandom", num_games
        )
        results["WeightedRandom"] = weighted_results
    except Exception as e:
        logger.error(f"Failed to evaluate against WeightedRandom: {e}")
    
    return results

if __name__ == "__main__":
    print("=== Catanatron RL Evaluation and Deployment ===")
    
    # Example usage (commented out since no trained model exists yet)
    # model_path = "models/best_dqn_model.pt"
    # 
    # if os.path.exists(model_path):
    #     # Quick evaluation
    #     results = quick_evaluation(model_path, "dqn", num_games=50)
    #     print(f"Quick evaluation results: {results}")
    #     
    #     # Comprehensive evaluation
    #     evaluator = CatanatronEvaluator(model_path, "dqn")
    #     comprehensive_results = evaluator.comprehensive_evaluation(100)
    #     
    #     # Visualize results
    #     evaluator.visualize_results("evaluation_results.png")
    #     
    #     # Save results
    #     evaluator.save_results("evaluation_results.json")
    #     
    #     # Create deployment
    #     deployment = CatanatronDeployment(model_path, "dqn")
    #     deployment.export_for_cli("exported_rl_player.py")
    # else:
    #     print(f"Model file not found: {model_path}")
    #     print("Train a model first using the training pipeline!")
    
    print("Evaluation and deployment tools ready!")
    print("Train a model first, then use these tools to evaluate and deploy it.")
