---
layout: post
title:  "An Introduction to Reinforcement Learning"
date:   2019-08-13 13:00:00 +0500
categories: RL Python
---

Reinforcement Learning 

### Environment Dynamics

MDP leads to a sequence:

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

### Rewards

We denote expected return for future rewards with respect to time $t$ as $G_t$. The task is not necessarily episodic so we use discounting, with rate $\gamma$ between 0 and 1:

$$
\begin{align}
    G_t &= \sum_{k=0}^\infty \gamma^k R_{t+k+1}\\
    &= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \ldots \\
    &= R_{t+1} + \gamma \big(R_{t+2} + \gamma R_{t+3} + \gamma^2 R_{t+4} + \ldots\big) \\
    &= R_{t+1} + \gamma G_{t+1}
\end{align}
$$

### Value Functions

Define how good it is for an agent to be in a state, state-value function $v_\pi(s)$ means value at state $s$ and following policy $\pi$ from then on:

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

### Optimization Policies

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

### Asynchronous DP

Iterative DP algorithm that updates over the state one by one, with no need to update all at once, it can update one section multiple times before updating another once. However, for convergence, it cannot stop updating values of a section after some point in the computation.

We try to pick update sections in order to improve the rate of progress. Some states need their values updated more often than others.

Given an MDP, we can run an iterative DP algorithm at the same time that the agent is actually experiencing the MDP. The experience gives the DP states to update, simultaneously, the latest value and policy guide the agent's decisions.


### Generalized Policy Iteration

Policy iteration is a mix of making the Value Function consistent with the policy (policy evaluation), and the other is making the policy greedy with respect to the Value Function (policy improvement).

These processes complete before the other runs. In Value iteration, we only do one sweep of policy evaluation before policy improvement. In Async DP, the evaluation and improvement process are even more closely inter-weaved.

Generalized Policy Iteration (GPI), means the general idea of letting policy evaluation and policy improvement interact, independent of their granularity. Most of RL methods fall under this. And once they processes become stable, we must have reached the optimal state and policy.



```python
int Razorpay = require('razorpay');
```
Check out the [Jekyll docs][jekyll-docs].

[jekyll-docs]: https://jekyllrb.com/docs/home
