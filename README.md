# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement

In this experiment, the **FrozenLake-v1** environment is solved using the **Value Iteration** algorithm.

The agent starts from the initial state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process (MDP). Value Iteration repeatedly updates the value of each state using the Bellman Optimality Equation until the values converge.

The objectives are to compute:

- The optimal state-value function, $V^*(s)$
- The optimal policy, $\pi^*(s)$

---
## Software Requirements
Install the required Python packages:

```bash
pip install gymnasium numpy matplotlib
```
---
## Environment Description
The experiment uses the **Gymnasium FrozenLake-v1** environment.

FrozenLake is a grid-world reinforcement learning environment where the agent moves over frozen tiles and attempts to reach the goal without falling into holes.

For the default **4 × 4 FrozenLake** map:

| Component | Description |
|-----------|-------------|
| Environment | FrozenLake-v1 |
| Map Size | 4 × 4 |
| Observation Space | 16 discrete states |
| Action Space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal States | Goal state and Hole states |

---

## MDP Representation
A Markov Decision Process is represented as:

$$
MDP = (S, A, P, R, \gamma)
$$

Where:

| Symbol | Meaning |
|--------|---------|
| $S$ | Set of states |
| $A$ | Set of actions |
| $P$ | Transition probability function |
| $R$ | Reward function |
| $\gamma$ | Discount factor |

The FrozenLake environment contains **16 states**, **4 actions**, probabilistic state transitions, rewards, and a discount factor used to compute future rewards.

---

## Theory
Value Iteration is a **Dynamic Programming** algorithm used to compute the optimal state-value function and the optimal policy of a Markov Decision Process.

Instead of evaluating a fixed policy, Value Iteration directly computes the optimal value of each state by repeatedly applying the **Bellman Optimality Equation** until the value function converges.

The Bellman Optimality Equation is:


```math
V(s) = \max_{a} \sum_{s'} P(s' \mid s, a)
\left[ R(s, a, s') + \gamma V(s') \right]
```

Where:

| Symbol | Meaning |
|--------|---------|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^*(s)$ | Optimal state-value function |

After the value function converges, the optimal policy is obtained by selecting the action that produces the highest expected value for every state.

---
## Algorithm

1. Create the **FrozenLake-v1** environment using Gymnasium.
2. Initialize the value function $V(s)=0$ for all states.
3. Repeat until convergence:
   - Compute the value of each action.
   - Update every state's value using the Bellman Optimality Equation.
   - Store the maximum action value.
4. Stop when the change in state values becomes smaller than the convergence threshold.
5. Derive the optimal policy by selecting the action with the highest value for every state.
6. Display the optimal state-value function and the optimal policy.

---

## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env_desc = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
# write your code here
# Access the underlying environment
env = env.unwrapped

n_states = env.observation_space.n
n_actions = env.action_space.n

gamma = 0.99
theta = 1e-8
# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):
  
    """
    Performs value iteration and returns the optimal value function.
    """
    # write your code here
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Initialize the value function and policy
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)
    iteration = 0
    
    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]
            action_values = np.zeros(n_actions)
            
            for a in range(n_actions):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state])
            
            V[s] = np.max(action_values)
            delta = max(delta, abs(v - V[s]))
            
        iteration += 1
        
        if delta < theta:
            break
            
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        
        policy[s] = np.argmax(action_values)


    return V, policy, iteration
# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------

# write your code here
V, policy, iterations = value_iteration(env)

# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name:  PYNAM VINODH ")
print("Register Number: 212223240131")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))


action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()




```

---

## Output

<img width="440" height="285" alt="image" src="https://github.com/user-attachments/assets/a567a043-4845-45a7-9897-90453f49cb19" />

---

## Result
Value Iteration was successfully implemented using the Gymnasium FrozenLake-v1 environment. The algorithm repeatedly applied the Bellman optimality equation to update the state-value function until convergence. After convergence, the optimal policy was obtained by selecting the action with the maximum expected value for each state.

---

## Inference

Value Iteration provides an efficient Dynamic Programming approach for computing the optimal value function and policy of a finite MDP. It avoids the separate policy-evaluation phase used in Policy Iteration and directly performs maximization over possible actions. In the stochastic FrozenLake environment, transition probabilities from env.P[s][a] are incorporated while calculating the expected return. States that provide better chances of eventually reaching the goal obtain higher values, while terminal hole states have zero value.

---

