# Catanatron

[![Coverage Status](https://coveralls.io/repos/github/bcollazo/catanatron/badge.svg?branch=master)](https://coveralls.io/github/bcollazo/catanatron?branch=master)
[![Documentation Status](https://readthedocs.org/projects/catanatron/badge/?version=latest)](https://catanatron.readthedocs.io/en/latest/?badge=latest)
[![Join the chat at https://gitter.im/bcollazo-catanatron/community](https://badges.gitter.im/bcollazo-catanatron/community.svg)](https://gitter.im/bcollazo-catanatron/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bcollazo/catanatron/blob/master/examples/Overview.ipynb)

Catanatron is a high-performance simulator and strong AI player for Settlers of Catan. You can run thousands of games in the order of seconds. The goal is to find the strongest Settlers of Catan bot possible. 

Get Started with the Full Documentation: https://docs.catanatron.com

## 🤖 Reinforcement Learning Agents

### Quick Start with RL Agents

```bash
# Train a new DQN agent
python train_rl_agent.py --agent dqn --episodes 1000

```

This repository includes a comprehensive Deep Q-Network (DQN) agent implementation trained to play Settlers of Catan:

- **Pre-trained Models**: Ready-to-use DQN models available in `training_outputs_dqn/`
- **Advanced Architecture**: 614-dimensional state space, 290-action space with action masking
- **Curriculum Learning**: 4-phase training progression over 2000 episodes
- **Production Ready**: Complete training pipeline and evaluation tools

### Pre-trained DQN Models

The `training_outputs_dqn/` folder contains:
- **`final_dqn_model.pt`**: Best performing model after 2000 episodes
- **Checkpoint models**: `dqn_episode_500.pt`, `dqn_episode_1000.pt`, `dqn_episode_1500.pt`, `dqn_episode_2000.pt`
- **Training metrics**: Episode-wise performance data and hyperparameter configuration
- **Architecture**: 614 → 256 → 128 → 290 neural network with experience replay

### Model Training Details

The DQN agent was trained using curriculum learning over 2000 episodes:
1. **Episodes 1-500**: RandomPlayer opponents (exploration phase)
2. **Episodes 501-1000**: WeightedRandomPlayer opponents  
3. **Episodes 1001-1500**: VictoryPointPlayer opponents
4. **Episodes 1501-2000**: AlphaBetaPlayer opponents (strongest phase)

Key hyperparameters: 20,000 replay buffer, ε-greedy decay over 1600 episodes, target network updates every 1000 steps.

## Command Line Interface
Catanatron provides a `catanatron-play` CLI tool to run large scale simulations.

<p align="left">
 <img src="https://raw.githubusercontent.com/bcollazo/catanatron/master/docs/source/_static/cli.gif">
</p>

### Installation

1. Clone the repository:

    ```bash
    git clone git@github.com:bcollazo/catanatron.git
    cd catanatron/
    ```
2. Create a virtual environment (requires Python 3.11 or higher) 

    ```bash
    python -m venv venv
    source ./venv/bin/activate
    ```
3. Install dependencies

    ```bash
    pip install -e .
    ```
4. (Optional) Install developer and advanced dependencies 

    ```bash
    pip install -e .[web,gym,dev]
    ```

### Usage

Run simulations and generate datasets via the CLI:

```bash
catanatron-play --players=R,R,R,W --num=100
```

Generate datasets from the games to analyze:
```bash
catanatron-play --num 100 --output my-data-path/ --output-format json
```

See more examples at https://docs.catanatron.com.


## Graphical User Interface

We provide Docker images so that you can watch, inspect, and play games against Catanatron via a web UI!

<p align="left">
 <img src="https://raw.githubusercontent.com/bcollazo/catanatron/master/docs/source/_static/CatanatronUI.png">
</p>


### Installation

1. Ensure you have Docker installed (https://docs.docker.com/engine/install/)
2. Run the `docker-compose.yaml` in the root folder of the repo:

    ```bash
    docker compose up
    ```
3. Visit http://localhost:3000 in your browser!

## Python Library

You can also use `catanatron` package directly which provides a core
implementation of the Settlers of Catan game logic.

```python
from catanatron import Game, RandomPlayer, Color

# Play a simple 4v4 game
players = [
    RandomPlayer(Color.RED),
    RandomPlayer(Color.BLUE),
    RandomPlayer(Color.WHITE),
    RandomPlayer(Color.ORANGE),
]
game = Game(players)
print(game.play())  # returns winning color
```

See more at http://docs.catanatron.com

## Gymnasium Interface
For Reinforcement Learning, catanatron provides an Open AI / Gymnasium Environment.

Install it with:
```bash
pip install -e .[gym]
```

and use it like:
```python
import random
import gymnasium
import catanatron.gym

env = gymnasium.make("catanatron/Catanatron-v0")
observation, info = env.reset()
for _ in range(1000):
    # your agent here (this takes random actions)
    action = random.choice(info["valid_actions"])

    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        observation, info = env.reset()
env.close()
```

See more at: https://docs.catanatron.com


## Documentation
Full documentation here: https://docs.catanatron.com

## Contributing

To develop for Catanatron core logic, install the dev dependencies and use the following test suite:

```bash
pip install .[web,gym,dev]
coverage run --source=catanatron -m pytest tests/ && coverage report
```

See more at: https://docs.catanatron.com

## Appendix
See the motivation of the project here: [5 Ways NOT to Build a Catan AI](https://medium.com/@bcollazo2010/5-ways-not-to-build-a-catan-ai-e01bc491af17).


