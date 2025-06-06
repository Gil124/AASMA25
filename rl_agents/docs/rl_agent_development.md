# RL Agent Development Guide for Catanatron

## Phase 1: Environment Setup

### Prerequisites
- Python 3.11 or higher
- Virtual environment
- Git repository cloned

### Installation Commands
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install Catanatron with RL dependencies
pip install -e .[gym,dev]

# Install additional RL libraries
pip install stable-baselines3[extra]
pip install tensorboard
pip install wandb  # Optional: for experiment tracking
```

## Phase 2: Understanding the Environment

### Key Components to Study
1. **Gymnasium Environment**: `catanatron/catanatron/gym/envs/catanatron_env.py`
2. **Feature Extraction**: `catanatron/catanatron/features.py`
3. **Action Space**: Defined in `catanatron/gym/envs/catanatron_env.py`
4. **Observation Space**: Game state representation for RL

### Environment Characteristics
- **Action Space**: Discrete actions (around 4000+ possible actions)
- **Observation Space**: Mixed (features + board tensor)
- **Reward Structure**: Sparse rewards (win/lose/victory points)
- **Episode Length**: Variable (until someone wins)
