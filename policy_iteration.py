import gymnasium as gym
import numpy as np
import pygame
import time

def unwrap_env(env):
    current_env = env
    while hasattr(current_env, 'env'):
        current_env = current_env.env
    return current_env

def policy_evaluation(env, policy, gamma=0.9, theta=1e-5, max_iterations=1000):
    num_states = env.observation_space.n
    value_function = np.zeros(num_states)

    for _ in range(max_iterations):
        delta = 0  # To track convergence
        for state in range(num_states):
            old_value = value_function[state]
            action = policy[state]
            state_value = 0
            unwrapped_env = unwrap_env(env)
            for prob, next_state, reward, _ in unwrapped_env.P[state][action]:
              state_value += prob * (reward + gamma * value_function[next_state])

            value_function[state] = state_value
            delta = max(delta, abs(old_value - value_function[state]))
            
        if delta < theta:
            break
    return value_function

def policy_improvement(env, policy, value_function, gamma=0.9):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    policy_stable = True
    new_policy = {}

    for state in range(num_states):
        old_action = policy[state]
        action_values = np.zeros(num_actions)
        unwrapped_env = unwrap_env(env)
        for action in range(num_actions):
            for prob, next_state, reward, _ in unwrapped_env.P[state][action]:
                action_values[action] += prob * (reward + gamma * value_function[next_state])
        
        new_action = np.argmax(action_values)
        new_policy[state] = new_action
        
        if old_action != new_action:
            policy_stable = False

    return new_policy, policy_stable

def policy_iteration(env, gamma=0.9, theta=1e-5):
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    policy = {state: np.random.choice(num_actions) for state in range(num_states)}

    policy_stable = False

    while not policy_stable:
        value_function = policy_evaluation(env, policy, gamma, theta)
        policy, policy_stable = policy_improvement(env, policy, value_function, gamma)

    return policy, value_function

if __name__ == '__main__':
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode = "human")
    
    optimal_policy, optimal_value_function = policy_iteration(env)
    
    print("Optimal Policy:")
    print(optimal_policy) 

    print("\nOptimal Value Function:")
    print(optimal_value_function) 
    
    state = env.reset()[0]
    terminated = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not terminated and not truncated:
        action = optimal_policy[state]
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step_count += 1
        env.render()
        time.sleep(0.5) 
        
    print(f"Episode finished after {step_count} steps with total reward: {total_reward}")


    env.close()
