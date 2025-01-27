import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.8, help='learning rate')
parser.add_argument('--gemma', type=float, default=0.95, help='discount factor')
parser.add_argument('--num_episodes', type=int, default=10000, help='number of episodes')
args = parser.parse_args()

env = gym.make('FrozenLake-v1', is_slippery=False)
current_state, info = env.reset(seed=69)

class Q_Learning:
    def __init__(self, learning_rate=0.1, discount=0.95):
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = 0.05
        self.rng = np.random.default_rng(5623)
        self.q = np.zeros((env.observation_space.n, env.action_space.n))

    def update_q(self, current_state, action, reward, new_state):
        self.q[current_state, action] += self.learning_rate * (reward + self.discount * np.max(self.q[new_state, :]) - self.q[current_state, action])

    def get_next_action(self, current_state):
        if (self.rng.uniform(0, 1) < self.epsilon) or (np.all(self.q[current_state, :]) == self.q[current_state, 0]):
            return env.action_space.sample()
        return np.argmax(self.q[current_state])

    def run_episode(self, episode):
        current_state, info = env.reset()
        # env.render()
        sample = episode in sample_at
        current_policy = []
        total_reward = 0
        for i in range(max_steps):
            action = self.get_next_action(current_state)
            new_state, reward, done, what, huh = env.step(action)
            env.render() # add render here.

            if sample:
                current_policy.append(action)

            if reward == 1:
                reward = 1 # Keep the reward consistent
            elif done:
                reward = 0 # Keep reward consistent

            if not done:
              reward = -0.1 # small penalty to encourage finishing the maze faster. 

            self.update_q(current_state, action, reward, new_state)
            total_reward += reward

            current_state = new_state
            if done:
               break 
        return current_policy, total_reward

EPISODES = args.num_episodes
max_steps = env.observation_space.n * 2
rewards_per_episode = []
sample_at = [3, 10, 33, 50, 98, 150, 170, 180, 200, 300, 500, 1000, 2000, 3000, 4000, 8000, 9000]
sample_policies = []

Agent = Q_Learning(learning_rate = args.lr, discount= args.gemma)
for episode in range(EPISODES):
    current_policy, total_reward = Agent.run_episode(episode)
    rewards_per_episode.append(total_reward)
    if episode in sample_at:
         sample_policies.append(current_policy)

plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode)
plt.title('Learning Curve - Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.grid(True)
plt.show()


env.close()


env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
current_state, info = env.reset(seed=69)
for i in range(len(sample_policies)):
    env.reset()
    print(f"Sample Policy at Episode {sample_at[i]}")
    time.sleep(1)
    for j in range(len(sample_policies[i])):
        action = sample_policies[i][j]
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            observation, _ = env.reset()
            break
env.close()
