#!/usr/bin/env python3
"""
Main entry point for Catanatron RL training.

This script provides a convenient way to train RL agents using the 
reorganized module structure while maintaining backward compatibility.

Usage:
    python train_rl_agent.py --agent dqn --episodes 1000
    python train_rl_agent.py --agent ppo --config custom_config.json
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from rl_agents.training.train import main as train_main


def main():
    """Main entry point for RL training."""
    parser = argparse.ArgumentParser(description="Train Catanatron RL Agent")
    parser.add_argument("--agent", choices=["dqn", "ppo"], default="dqn",
                       help="Type of RL agent to train")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--config", type=str, 
                       default="rl_agents/config/training_config.json",
                       help="Path to training configuration file")
    parser.add_argument("--output-dir", type=str, default="training_outputs",
                       help="Directory to save training outputs")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                       help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Call the training function with parsed arguments
    train_main(
        agent_type=args.agent,
        num_episodes=args.episodes,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
