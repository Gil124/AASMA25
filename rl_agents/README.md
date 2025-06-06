# Catanatron RL Agents - Restructured

This directory contains the reorganized reinforcement learning components for the Catanatron Settlers of Catan game engine.

## Directory Structure

```
rl_agents/
├── __init__.py                    # Main package exports
├── core/                          # Core RL components
│   ├── __init__.py
│   ├── agents.py                  # DQN/PPO agent implementations
│   ├── player.py                  # Catanatron player integration
│   └── environment.py             # Environment testing utilities
├── training/                      # Training components
│   ├── __init__.py
│   ├── utilities.py               # Replay buffers, loggers, utilities
│   ├── pipeline.py                # Training pipeline orchestration
│   └── train.py                   # Main training script
├── evaluation/                    # Evaluation and deployment
│   ├── __init__.py
│   └── deployment.py              # Model evaluation and deployment
├── config/                        # Configuration
│   ├── __init__.py
│   └── training_config.json       # Default training parameters
└── docs/                          # Documentation
    ├── RL_AGENT_ARCHITECTURE.md   # Comprehensive architecture guide
    └── rl_agent_development.md     # Development guide
```

## Quick Start

### Training a New Agent

```python
# Method 1: Using the new entry point script
python train_rl_agent.py --agent dqn --episodes 1000

# Method 2: Direct import
from rl_agents.training.train import main
main(agent_type="dqn", num_episodes=1000)

# Method 3: Using the package
from rl_agents import CatanatronTrainer
trainer = CatanatronTrainer()
trainer.train(agent_type="dqn", episodes=1000)
```

### Loading a Trained Agent

```python
from rl_agents import create_rl_player
from catanatron.models.player import Color

# Create RL player
rl_player = create_rl_player(
    color=Color.BLUE,
    agent_type="dqn",
    model_path="training_outputs/models/best_DQN_model.pt",
    training_mode=False
)

# Use in games...
```

### Evaluation and Deployment

```python
from rl_agents import CatanatronEvaluator, CatanatronDeployment

# Evaluate trained model
evaluator = CatanatronEvaluator("training_outputs/models/best_DQN_model.pt")
results = evaluator.evaluate_vs_baselines(num_games=100)

# Deploy for production use
deployment = CatanatronDeployment("training_outputs/models/best_DQN_model.pt")
deployment.export_for_cli("exported_rl_player.py")
```

## Backward Compatibility

The old `step*.py` files remain in the root directory for backward compatibility, but they now import from the new structure. New development should use the reorganized modules.

### Migration Guide

| Old Import | New Import |
|------------|------------|
| `from step3_rl_agent_architecture import MaskedDQNAgent` | `from rl_agents.core.agents import MaskedDQNAgent` |
| `from step5_rl_player import create_rl_player` | `from rl_agents.core.player import create_rl_player` |
| `from step6_training_pipeline import CatanatronTrainer` | `from rl_agents.training.pipeline import CatanatronTrainer` |
| `from step7_evaluation_deployment import CatanatronEvaluator` | `from rl_agents.evaluation.deployment import CatanatronEvaluator` |

## Configuration

Training parameters are now centralized in `config/training_config.json`. You can:

1. Modify the default config file
2. Provide a custom config file: `python train_rl_agent.py --config my_config.json`
3. Override parameters programmatically when calling training functions

## Documentation

- **[RL_AGENT_ARCHITECTURE.md](docs/RL_AGENT_ARCHITECTURE.md)**: Comprehensive guide to the RL agent architecture, usage, and implementation details
- **[rl_agent_development.md](docs/rl_agent_development.md)**: Development guide for extending and customizing the RL agents

## Benefits of Restructuring

1. **Better Organization**: Logical separation of concerns
2. **Easier Navigation**: Clear module boundaries
3. **Improved Imports**: Clean, intuitive import paths
4. **Better Documentation**: Centralized docs in one location
5. **Configuration Management**: Centralized configuration
6. **Backward Compatibility**: Existing code continues to work
7. **Future Extensions**: Easy to add new components

## Examples

See the `examples/` directory in the main project for usage examples that work with both the old and new structure.
