"""
Report-Compatible Evaluation System for RL Agents
This module provides evaluation functionality that aligns with the AASMA25 project report methodology.
"""
import numpy as np
import time
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import scipy.stats as stats

from catanatron.models.player import Color
from catanatron.game import Game
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.models.player import RandomPlayer

from ..core.player import CatanatronRLPlayer

logger = logging.getLogger(__name__)

class ReportEvaluator:
    """
    Evaluation system aligned with AASMA25 report methodology.
    Focuses on 2-player matchups with statistical validation.
    """
    
    def __init__(self, model_path: str, agent_type: str = "dqn", device: str = "cpu"):
        self.model_path = model_path
        self.agent_type = agent_type
        self.device = device
          # Benchmark targets from report
        self.benchmark_targets = {
            "AlphaBetaPlayer": {"win_rate": 0.89, "decision_time_ms": 14.12},
            "WeightedRandomPlayer": {"win_rate": 0.60, "decision_time_ms": 5.0},
            "MCTSPlayer": {"win_rate": 0.70, "decision_time_ms": 50.0},
            "GreedyPlayoutsPlayer": {"win_rate": 0.65, "decision_time_ms": 25.0}
        }
        
    def create_rl_player(self, color: Color) -> CatanatronRLPlayer:
        """Create and initialize an RL player"""
        player = CatanatronRLPlayer(
            color=color,
            agent_type=self.agent_type,
            model_path=self.model_path,
            training_mode=False,
            device=self.device
        )
        # Initialize with correct dimensions from trained model (290 actions, 614 obs)
        player.initialize_agent(614, 290)
        return player
    
    def create_opponent_suite(self) -> Dict[str, Any]:
        """Create the suite of heuristic opponents for evaluation"""
        return {
            "AlphaBetaPlayer": lambda color: AlphaBetaPlayer(color, prunning=False),
            "AlphaBetaImproved": lambda color: AlphaBetaPlayer(color, prunning_improved=True),
            "WeightedRandomPlayer": lambda color: WeightedRandomPlayer(color),
            "MCTSPlayer": lambda color: MCTSPlayer(color, num_simulations=100),
            "GreedyPlayoutsPlayer": lambda color: GreedyPlayoutsPlayer(color),
            "RandomPlayer": lambda color: RandomPlayer(color)
        }
    
    def evaluate_two_player_matchup(self, opponent_name: str, opponent_factory, 
                                  num_games: int = 500) -> Dict[str, Any]:
        """
        Evaluate RL agent in 2-player matchups against a specific opponent.
        
        Args:
            opponent_name: Name of the opponent
            opponent_factory: Function to create opponent instances
            num_games: Number of games to play (default 500 for statistical significance)
            
        Returns:
            Dictionary with evaluation metrics and statistical analysis
        """
        logger.info(f"Evaluating 2-player matchups vs {opponent_name} ({num_games} games)...")
        
        # Metrics tracking
        rl_wins = 0
        rl_decision_times = []
        opponent_decision_times = []
        game_durations = []
        rl_victory_points = []
        opponent_victory_points = []
        
        results_per_game = []
        
        for game_num in range(num_games):
            start_time = time.time()
            
            # Create players (alternate who goes first for fairness)
            if game_num % 2 == 0:
                rl_player = self.create_rl_player(Color.RED)
                opponent = opponent_factory(Color.BLUE)
                players = [rl_player, opponent]
            else:
                rl_player = self.create_rl_player(Color.BLUE)
                opponent = opponent_factory(Color.RED)
                players = [opponent, rl_player]
            
            # Reset decision time tracking
            rl_player.reset_decision_times()
            if hasattr(opponent, 'decision_times'):
                opponent.decision_times = []
            
            try:
                # Run game
                game = Game(players)
                game.play()
                
                # Collect results
                winner = game.winning_color()
                game_duration = time.time() - start_time
                
                # Determine RL player win
                rl_won = (winner == rl_player.color)
                if rl_won:
                    rl_wins += 1
                
                # Get victory points
                rl_vp = self._get_victory_points(game, rl_player.color)
                opp_vp = self._get_victory_points(game, opponent.color)
                
                # Collect decision times
                rl_avg_decision = rl_player.average_decision_time()
                opp_avg_decision = (opponent.average_decision_time() 
                                  if hasattr(opponent, 'average_decision_time') 
                                  else 0.0)
                
                # Store results
                rl_decision_times.append(rl_avg_decision)
                opponent_decision_times.append(opp_avg_decision)
                game_durations.append(game_duration)
                rl_victory_points.append(rl_vp)
                opponent_victory_points.append(opp_vp)
                
                results_per_game.append({
                    'rl_won': rl_won,
                    'rl_vp': rl_vp,
                    'opp_vp': opp_vp,
                    'rl_decision_time': rl_avg_decision,
                    'game_duration': game_duration
                })
                
            except Exception as e:
                logger.warning(f"Game {game_num} failed: {e}")
                results_per_game.append({
                    'rl_won': False,
                    'rl_vp': 0,
                    'opp_vp': 10,
                    'rl_decision_time': 0.0,
                    'game_duration': 0.0
                })
        
        # Calculate primary metrics
        win_rate = rl_wins / num_games
        avg_rl_decision_time = np.mean(rl_decision_times)
        avg_game_duration = np.mean(game_durations)
        avg_rl_vp = np.mean(rl_victory_points)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(
            results_per_game, opponent_name
        )
        
        # Compile results
        results = {
            "opponent": opponent_name,
            "num_games": num_games,
            "win_rate": win_rate,
            "wins": rl_wins,
            "losses": num_games - rl_wins,
            "avg_decision_time_ms": avg_rl_decision_time,
            "avg_game_duration_s": avg_game_duration,
            "avg_victory_points": avg_rl_vp,
            "std_victory_points": np.std(rl_victory_points),
            "statistical_analysis": statistical_analysis,
            "benchmark_comparison": self._compare_to_benchmark(
                opponent_name, win_rate, avg_rl_decision_time
            ),
            "raw_data": {
                "rl_decision_times": rl_decision_times,
                "game_durations": game_durations,
                "rl_victory_points": rl_victory_points
            }
        }
        
        logger.info(f"Results vs {opponent_name}: Win rate={win_rate:.3f}, "
                   f"Avg decision time={avg_rl_decision_time:.2f}ms")
        
        return results
    
    def comprehensive_tournament_evaluation(self, num_games_per_opponent: int = 500) -> Dict[str, Any]:
        """
        Run comprehensive tournament evaluation against all heuristic opponents
        """
        logger.info("Starting comprehensive tournament evaluation...")
        
        opponents = self.create_opponent_suite()
        all_results = {}
        
        for opponent_name, opponent_factory in opponents.items():
            try:
                results = self.evaluate_two_player_matchup(
                    opponent_name, opponent_factory, num_games_per_opponent
                )
                all_results[opponent_name] = results
                
            except Exception as e:
                logger.error(f"Evaluation against {opponent_name} failed: {e}")
                all_results[opponent_name] = {"error": str(e)}
        
        # Calculate overall tournament performance
        overall_stats = self._calculate_tournament_stats(all_results)
        all_results["Tournament_Summary"] = overall_stats
        
        return all_results
    
    def _get_victory_points(self, game: Game, color: Color) -> int:
        """Extract victory points for a specific player"""
        try:
            player_state = game.state.player_state(f"p{color.value}")
            return player_state["public_victory_points"]
        except:
            return 0
    
    def _perform_statistical_analysis(self, results: List[Dict], opponent_name: str) -> Dict[str, Any]:
        """
        Perform statistical analysis with paired t-tests and confidence intervals
        """
        win_results = [1 if r['rl_won'] else 0 for r in results]
        decision_times = [r['rl_decision_time'] for r in results]
        victory_points = [r['rl_vp'] for r in results]
          # Win rate confidence interval using binomial proportion
        win_rate = np.mean(win_results)
        n = len(win_results)
        successes = sum(win_results)
        # Use normal approximation for confidence interval
        p_hat = successes / n
        z_score = 1.96  # 95% confidence interval
        se = np.sqrt(p_hat * (1 - p_hat) / n)
        win_rate_ci = (
            max(0, p_hat - z_score * se), 
            min(1, p_hat + z_score * se)
        )
        
        # Decision time analysis
        decision_time_mean = np.mean(decision_times)
        decision_time_ci = stats.t.interval(
            0.95, len(decision_times)-1, 
            loc=decision_time_mean, 
            scale=stats.sem(decision_times)
        )
        
        # Victory points analysis
        vp_mean = np.mean(victory_points)
        vp_ci = stats.t.interval(
            0.95, len(victory_points)-1,
            loc=vp_mean,
            scale=stats.sem(victory_points)
        )
        
        # Benchmark comparison t-test (if benchmark exists)
        benchmark_test = None
        if opponent_name in self.benchmark_targets:
            benchmark_win_rate = self.benchmark_targets[opponent_name]["win_rate"]
            # One-sample t-test against benchmark
            t_stat, p_value = stats.ttest_1samp(win_results, benchmark_win_rate)
            benchmark_test = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
                "better_than_benchmark": win_rate > benchmark_win_rate
            }
        
        return {
            "win_rate_ci_95": win_rate_ci,
            "decision_time_ci_95": decision_time_ci,
            "victory_points_ci_95": vp_ci,
            "sample_size": len(results),
            "benchmark_test": benchmark_test
        }
    
    def _compare_to_benchmark(self, opponent_name: str, win_rate: float, 
                            decision_time: float) -> Dict[str, Any]:
        """Compare performance to benchmark targets from report"""
        if opponent_name not in self.benchmark_targets:
            return {"available": False}
        
        benchmark = self.benchmark_targets[opponent_name]
        
        return {
            "available": True,
            "benchmark_win_rate": benchmark["win_rate"],
            "our_win_rate": win_rate,
            "win_rate_improvement": win_rate - benchmark["win_rate"],
            "benchmark_decision_time": benchmark["decision_time_ms"],
            "our_decision_time": decision_time,
            "decision_time_ratio": decision_time / benchmark["decision_time_ms"],
            "beats_benchmark": win_rate > benchmark["win_rate"]
        }
    
    def _calculate_tournament_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall tournament statistics"""
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if not valid_results:
            return {"error": "No successful evaluations"}
        
        win_rates = [r["win_rate"] for r in valid_results.values()]
        decision_times = [r["avg_decision_time_ms"] for r in valid_results.values()]
        
        # Count victories against benchmark agents
        benchmark_victories = 0
        total_benchmark_opponents = 0
        
        for opponent_name, result in valid_results.items():
            if opponent_name in self.benchmark_targets:
                total_benchmark_opponents += 1
                benchmark = result.get("benchmark_comparison", {})
                if benchmark.get("beats_benchmark", False):
                    benchmark_victories += 1
        
        return {
            "avg_win_rate": np.mean(win_rates),
            "std_win_rate": np.std(win_rates),
            "min_win_rate": np.min(win_rates),
            "max_win_rate": np.max(win_rates),
            "avg_decision_time_ms": np.mean(decision_times),
            "opponents_evaluated": len(valid_results),
            "benchmark_victories": benchmark_victories,
            "total_benchmark_opponents": total_benchmark_opponents,
            "benchmark_victory_rate": (benchmark_victories / total_benchmark_opponents 
                                    if total_benchmark_opponents > 0 else 0)
        }
    
    def generate_latex_table(self, results: Dict[str, Any]) -> str:
        """Generate LaTeX table for report integration"""
        latex_lines = [
            "\\begin{table}[H]",
            "\\centering",
            "\\caption{RL Agent Performance Against Heuristic Opponents}",
            "\\label{tab:rl_evaluation}",
            "\\begin{tabularx}{\\textwidth}{lYYYY}",
            "\\hline",
            "Opponent & Win Rate & Decision Time (ms) & Avg Victory Points & 95\\% CI Win Rate \\\\",
            "\\hline"
        ]
        
        for opponent_name, result in results.items():
            if opponent_name == "Tournament_Summary" or "error" in result:
                continue
                
            win_rate = result["win_rate"]
            decision_time = result["avg_decision_time_ms"]
            avg_vp = result["avg_victory_points"]
            
            # Get confidence interval
            ci = result["statistical_analysis"]["win_rate_ci_95"]
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            
            latex_lines.append(
                f"{opponent_name} & {win_rate:.3f} & {decision_time:.2f} & "
                f"{avg_vp:.1f} & {ci_str} \\\\"
            )
        
        latex_lines.extend([
            "\\hline",
            "\\end{tabularx}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)

def quick_report_evaluation(model_path: str, agent_type: str = "dqn", 
                           num_games: int = 200) -> Dict[str, Any]:
    """Quick evaluation function for report validation"""
    evaluator = ReportEvaluator(model_path, agent_type)
    
    # Evaluate against key opponents
    key_opponents = {
        "AlphaBetaPlayer": lambda color: AlphaBetaPlayer(color, prunning_improved=True),
        "WeightedRandomPlayer": lambda color: WeightedRandomPlayer(color)
    }
    
    results = {}
    for name, factory in key_opponents.items():
        results[name] = evaluator.evaluate_two_player_matchup(name, factory, num_games)
    
    return results
