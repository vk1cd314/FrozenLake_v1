# FrozenLake_v1

This repository contains implementations of three reinforcement learning algorithms (Policy Iteration, Value Iteration, and Q-Learning) to solve the `FrozenLake-v1` environment from Gymnasium. The implementations demonstrate both model-based and model-free approaches to find optimal policies for navigating the frozen lake.

## Algorithms

### 1. Policy Iteration (`policy_iteration.py`)
- **Model-based** dynamic programming method
- Alternates between:
  - *Policy Evaluation*: Iteratively estimates the value function under the current policy
  - *Policy Improvement*: Greedily updates the policy based on the computed values
- Outputs: Optimal policy, value function, and renders the agent's path

### 2. Value Iteration (`value_iteration.py`)
- **Model-based** method that directly computes optimal values
- Iteratively updates state values using Bellman optimality equations
- Automatically extracts optimal policy after convergence
- Includes visualization of value function convergence

### 3. Q-Learning (`qlearning.py`)
- **Model-free** temporal difference learning
- Learns action-value (Q) function through environment interactions
- Features:
  - ε-greedy exploration strategy
  - Learning rate and discount factor customization
  - Learning curve visualization
  - Policy sampling at intermediate training stages

## Requirements
- Python 3.7+
- Required packages:
  ```bash
  pip install gymnasium numpy matplotlib pygame argparse
  ```

## Usage

### Policy Iteration
```bash
python policy_iteration.py
```

### Value Iteration
```bash
python value_iteration.py
```

### Q-Learning
```bash
python qlearning.py 
```

## Key Features
- Environment customization:
  ```python
  gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
  ```
- Visualization components:
  - Real-time rendering of agent behavior
  - Value function convergence plots (Value Iteration)
  - Learning curve tracking (Q-Learning)
- Deterministic mode (`is_slippery=False`) for easier learning

