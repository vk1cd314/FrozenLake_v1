import gymnasium as gym
import numpy as np

seed = 0
np.random.seed(seed)
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False,render_mode='human')

import matplotlib.pyplot as plt

def value_iteration(env, gamma=0.99, eps=1e-5):
    V = np.zeros(env.observation_space.n)
    policy = np.zeros(env.observation_space.n)
    delta = np.inf
    i = 0
    V_history = [] 

    while delta >= eps * (1. - gamma) / gamma:
        i += 1
        delta = 0
        for state in range(env.observation_space.n):
            v = V[state]
            Q_table = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                for next_sr in env.unwrapped.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    Q_table[action] += (trans_prob * (reward_prob + gamma * V[next_state]))
            V[state] = max(Q_table)
            policy[state] = np.argmax(Q_table)
            delta = max(delta, abs(v - V[state]))
        V_history.append(V.copy())  
        
    print(f'Policy converged in {i} steps')
    return policy.astype(int), V, V_history

policy, V, V_history = value_iteration(env)
print(policy)

observation,_ = env.reset(seed=69)
while True:
    action = policy[observation]
    env.render()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation,_ = env.reset()
        break
env.close()

plt.figure(figsize=(10, 6))
for state in range(env.observation_space.n):
    plt.plot([v[state] for v in V_history], label=f'State {state}')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Convergence of State Values')
plt.legend()
plt.show()