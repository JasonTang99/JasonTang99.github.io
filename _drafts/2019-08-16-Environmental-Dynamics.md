---
layout: post
title: 'Basics of Banditry: Environment Dynamics and Value Iteration'
date: 2019-08-13 13:00:00 +0500
categories: RL Python
---

In the previous part we outlined the basics of the value function estimation and how to choose actions from it. Today we will more formally define a general representation of an environment and generate a policy from it.

For the previous parts:

Part 1: [Reinforcement Learning]({{ site.baseurl }}{% post_url 2019-08-13-Reinforcement-Learning %})

## Markov Decision Processes

A Markov Decision Process (MDP) is a sequence of decision making events where the outcome is determined partly randomly and partly controlled based on action. They also have the Markov property, i.e. that all conditional probabilities of future states depend only upon the current state and not those that came before it. MDP leads to a sequence of States ($S$), Actions ($A$), and Rewards ($R$):

$$
S_0,A_0,R_1,S_1,A_1,R_2,S_2,A_2,R_3,\ldots
$$

Given State $s$ and Action $a$, we define probabilities of Next State $s'$ and Reward $r$ with $p(s',r\|s,a)$, where:

$$
\forall s\in S, \forall a \in A(s), \sum_{s'\in S} \sum_{r\in R} p(s',r|s,a) = 1
$$

The probabilites given by $p$ characterizes the entire environment's dynamics.

There are several other forms of representing $p$:
<!-- State Transition Probabilities
Expected rewards for State-Action Pairs
Expected Rewards for State - Action - Next-State triples:} -->

$$
\begin{align}
    p(s'|s,a) &= \sum_{r\in R}p(s',r|s,a) \\
    r(s,a) &= \sum_{r\in R}r \sum_{s'\in S}p(s',r|s,a) \\
    r(s,a) &= \sum_{r\in R}r \frac{p(s',r|s,a)}{p(s'|s,a)} \\
\end{align}
$$

## Rewards

We denote expected return for future rewards with respect to time $t$ as $G_t$. The task is not necessarily episodic so we use discounting, with rate $\gamma$ between 0 and 1:

$$
\begin{align}
    G_t &= \sum_{k=0}^\infty \gamma^k R_{t+k+1}\\
    &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \ldots \\
    &= R_{t+1} + \gamma \big(R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \ldots\big) \\
    &= R_{t+1} + \gamma G_{t+1}
\end{align}
$$

## Value Functions

Defines how good it is for an agent to be in a state, state-value function $v_\pi(s)$ means value at state $s$ and following policy $\pi$ from then on:

$$v_\pi(s) = \mathbb{E}_\pi [G_t | S_t = s] = \mathbb{E}_\pi \bigg[\sum_{k=0}^\infty \gamma^k R_{t+k+1}| S_t = s\bigg]$$

Action-value function $q_\pi(s, a)$ defines the value of taking action $a$ in state $s$ and following policy $\pi$ from then on:

$$q_\pi(s,a)=\mathbb{E}[G_t | S_t = s, A_t=a] = \mathbb{E}_\pi \bigg[\sum_{k=0}^\infty \gamma^k R_{t+k+1}| S_t = s, A_t=a\bigg]$$

Relational Equations:

$$v_\pi(s) = \sum_{a\in A} \pi(a|s) * q_\pi(s, a)$$

$$q_\pi(s, a) = \sum_{s', r\in S, R} p(s', r | s, a) * (r + \gamma v_\pi(s'))$$

Value functions satisfy a recursive relationship:

$$
\begin{align}
    v_\pi(s) &= \mathbb{E}_\pi [G_t | S_t = s] \\
    &= \mathbb{E}_\pi [R_{t+1} + \gamma G_{t+1} | S_t = s] \\
    &= \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) \big[r + \gamma \mathbb{E}_\pi [G_{t+1} | S_{t+1} = s']\big] \\
    &= \sum_a \pi(a|s) \sum_{s', r} p(s', r | s, a) \big[r + \gamma v_\pi(s')\big] \\
\end{align}
$$

This is known as the *Bellman Equation*, essentially a weighted average of all possible actions, next states, and discounted future rewards.\\

## Optimization Policies

Policy $\pi'$ is better than $\pi$ if its expected return is greater than or equal to $\pi$'s in all states. So there is always one policy greater than or equal to all others, the optimal policy, $\pi_*$. 

$$\forall s\in S, v_*(s)=\max_\pi v_\pi(s)$$

$$\forall s, a\in S, A, q_*(s,a)=\max_\pi q_\pi(s,a)$$

Bellman optimality equations:

$$v_*(s)=\max_a q_{\pi_*}(s,a) = \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma v_*(s')]$$

$$q_*(s,a) = \sum_{s',r}p(s', r | s, a) [r + \gamma  v_*(s')]= \sum_{s',r}p(s', r | s, a) [r + \gamma \max_a' q_*(s',a')]$$

Given the optimal value state function $v_*$, any policy that assigns nonzero probabilities only to maximal actions are an optimal policies.

Solving these equations directly depends on knowing the dynamics of the environment, having enough computing power to calculate the solution and the markov property. 

$$\pi_*(a|s) = q_{*}(s,a) == \max_a q_{*}(s,a)$$

$$\pi_*(a|s) = \sum_{s', r} p(s', r | s, a) [r + \gamma v_*$$


### Dynamic Programming

Iterative Policy Evaluation calculates state values using dynamic programming until the recursive formula becomes stable with less than $\Delta$ change between iterations.

Iterative Policy Improvement compares the values of taking each action at each state, and if they're better than their respective state values, then the policy is changed accordingly.

$$\pi'(s) = {\arg\max}_a q_\pi (s,a)$$

### Value Iteration

When we stop policy evaluation after only one sweep, and then iteratively improve it, we get a simple update operation called value iteration.

$$v_{k+1}(s) = \max_a \sum_{s',r} p(s',r|s,a)\big[r+\gamma v_k(s')\big]$$
```python
# -*- coding: utf-8 -*-
"""value_iteration

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iBKHjvNY60F7waLa-fkqI-1-9pe0PGns
"""

# This code is based on an exercise in the Reinforcement Learning textbook by Richard Sutton and Andrew Barto
# Link to the textbook: http://incompleteideas.net/book/RLbook2018.pdf
# The exercise is the car rental exercise found in section 4.4 on page 84
# The dependencies for this code are numpy and python 3.6+

import numpy as np

NUM_STATES = 101 # 0 - 100
V_s = np.zeros(NUM_STATES)
V_s[NUM_STATES - 1] = 1 # Only place where reward is 1

V_index = np.array(range(1, 100))

policy = np.ones(99) # 1 - 99 are not terminal states

def get_actions(s):
  return list(range(1, min(s, 100 - s) + 1))

DELTA_LIM = 0.000001
DISCOUNT = 1

p_h = 0.4

# At most there are 50 actions to take (at state 50, actions: [1,50])

prob_table = []

for s in V_index: # [1, 99]
  heads = []
  tails = []
  for a in get_actions(s): # [1, min(s, 100 - s)]
    heads.append([p_h, 0, s + a])
    tails.append([1 - p_h, 0, s - a])

  for _ in range(50 - len(heads)):
    heads.append([0,0,0])
    tails.append([0,0,0])

  prob_table.append(np.stack([np.array(heads), np.array(tails)], axis = 1))
  
prob_table = np.array(prob_table)
prob_table.shape

def value_iteration_optimized(V_s):
  delta = DELTA_LIM + 1
  
  new_policy = None
  
  while delta > DELTA_LIM:
    delta = 0
  
    v = V_s.copy()
    
    reward = prob_table[:, :, :, 1] + DISCOUNT * v[prob_table[:, :, :, 2].astype(np.intp)] # (99, 50, 2)
    reward *= prob_table[:, :, :, 0] # (99, 50, 2)
    reward = np.sum(reward, axis=2) # (99, 50)
    
    V_s = np.max(reward, axis = 1) # (99)
    V_s = np.array([0] + V_s.tolist() + [1])
    
    new_policy = np.argmax(reward, axis = 1) + 1
    
    delta = np.amax(np.abs(v - V_s))
    
    print("DELTA", np.round(delta, 6))
    
  return new_policy, V_s



def P(s, a):
  return [[p_h, s + a], [1-p_h, s - a]]

def value_iteration(V_s):
  delta = DELTA_LIM + 1
  
  new_policy = np.ones(99)
  
  while delta > DELTA_LIM:
    delta = 0
    for s in range(1, NUM_STATES - 1):
      v = V_s[s]
      rewards = []
      for a in get_actions(s):
        reward = 0
        for prob, next_state in P(s, a):
          if next_state == 101:
            reward += prob * (1 + DISCOUNT * V_s[next_state])
          else:
            reward += prob * (DISCOUNT * V_s[next_state])
        rewards.append(reward)
      V_s[s] = max(rewards)
      new_policy[s - 1] = np.argmax(rewards) + 1
      delta = max(delta, abs(v - V_s[s]))
    print("DELTA", np.round(delta, 6))
    
  return new_policy, V_s

policy, V_s = value_iteration_optimized(V_s)
print(policy)
print(V_s)

import matplotlib.pyplot as plt
plt.bar(range(1,100), policy)
plt.show()

plt.plot(V_s)
plt.show()
```
### Asynchronous DP

Iterative DP algorithm that updates over the state one by one, with no need to update all at once, it can update one section multiple times before updating another once. However, for convergence, it cannot stop updating values of a section after some point in the computation.

We try to pick update sections in order to improve the rate of progress. Some states need their values updated more often than others.

Given an MDP, we can run an iterative DP algorithm at the same time that the agent is actually experiencing the MDP. The experience gives the DP states to update, simultaneously, the latest value and policy guide the agent's decisions.


### Generalized Policy Iteration

Policy iteration is a mix of making the Value Function consistent with the policy (policy evaluation), and the other is making the policy greedy with respect to the Value Function (policy improvement).

These processes complete before the other runs. In Value iteration, we only do one sweep of policy evaluation before policy improvement. In Async DP, the evaluation and improvement process are even more closely inter-weaved.

Generalized Policy Iteration (GPI), means the general idea of letting policy evaluation and policy improvement interact, independent of their granularity. Most of RL methods fall under this. And once they processes become stable, we must have reached the optimal state and policy.