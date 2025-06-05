#!/usr/bin/env python3
"""
Basic Test Script - Minimal Testing
"""

print("Starting basic test...")

try:
    import torch
    print("✓ PyTorch imported")
    
    import numpy as np
    print("✓ NumPy imported")
    
    import gymnasium
    print("✓ Gymnasium imported")
    
    # Test our modules
    from step3_rl_agent_architecture import MaskedDQNAgent
    print("✓ DQN Agent imported")
    
    # Create simple agent
    agent = MaskedDQNAgent(
        obs_dim=10,
        action_dim=5,
        hidden_dim=32,
        learning_rate=1e-3,
        buffer_size=100,
        batch_size=8
    )
    print("✓ DQN Agent created successfully")
    
    # Test forward pass
    obs = torch.randn(1, 10)
    action_mask = torch.ones(1, 5, dtype=torch.bool)
    action = agent.select_action(obs, action_mask)
    print(f"✓ Action selected: {action}")
    
    print("\n🎉 Basic functionality test PASSED!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
