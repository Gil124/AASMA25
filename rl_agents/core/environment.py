"""
Step 2: Test and understand the Catanatron Gymnasium Environment
This script helps you understand the environment structure and test basic functionality.
"""
import gymnasium as gym
import numpy as np
import catanatron.gym

def test_environment():
    """Test basic environment functionality"""
    print("=== Testing Catanatron Gymnasium Environment ===")
    
    # Create environment
    env = gym.make("catanatron/Catanatron-v0")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    observation, info = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    print(f"Valid actions available: {len(info['valid_actions'])}")
    print(f"Sample valid actions: {info['valid_actions'][:5]}")
    
    # Run a few random steps
    for step in range(10):
        # Choose random valid action
        action = np.random.choice(info['valid_actions'])
        
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step}: Action={action}, Reward={reward}, Done={terminated or truncated}")
        
        if terminated or truncated:
            print("Episode ended!")
            observation, info = env.reset()
    
    env.close()
    print("Environment test completed!")

def analyze_observation_space():
    """Analyze the observation space structure"""
    print("\n=== Analyzing Observation Space ===")
    
    env = gym.make("catanatron/Catanatron-v0")
    observation, info = env.reset()
    
    print(f"Observation type: {type(observation)}")
    print(f"Observation shape: {observation.shape}")
    print(f"Observation min: {observation.min()}")
    print(f"Observation max: {observation.max()}")
    
    # Sample a few observations to understand the range
    observations = []
    for _ in range(5):
        action = np.random.choice(info['valid_actions'])
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(observation)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    observations = np.array(observations)
    print(f"Sample observations mean: {observations.mean(axis=0)[:10]}...")  # First 10 features
    print(f"Sample observations std: {observations.std(axis=0)[:10]}...")   # First 10 features
    
    env.close()

if __name__ == "__main__":
    test_environment()
    analyze_observation_space()
