---
layout: post
title: 'Basics of Banditry: Reinforcement Learning'
date: 2019-08-13 13:00:00 +0500
categories: RL Python
---

Reinforcement Learning is currently at the forefront of artificial intelligence. Many, including myself, believe that truly general artificial intelligence will come from either RL or something we have yet to discover. I hope that through this series of blog posts I can give you an introduction into the world of RL. 

Hold on, where did I learn everything? Well, most of it stems from the textbook [Reinforcement Learning by Richard Sutton and Andrew Barto][rlbook], it's an great read and this entire blog post series is based on it.

Now let's jump right into the basics.

## $k$-armed Bandits

No this is not about mutant highwaymen, this about slot machines and choices. Consider a slot machine (often refered to as a 1-armed Bandit), but in this case, it has $k$ levers to pull, each leading to a play on the machine with different, independent odds of winning. In order to maximize one's winnings over repeated plays, one would seek out the highest performing levers and keep pulling them. 

{% include image.html url="/assets/photos/bandits.PNG" description="The above figure details a 10-armed version of the $k$-armed bandit problem. Each lever has an associated reward sampling distribution, where some levers are more likely to have higher rewards." %}

Here's the setup for this environment in python:
```python
import numpy as np

q = np.random.normal(0.0, 2.0, size=10)
std = 0.5
```

Out of our $k$ choices of levers to pull, each one has an expected value of what reward we will get. Let's denote the action at time $t$ as $A_t$, and the reward from it as $R_{t+1}$ (the time is different since technically the reward occurs in the next time step but you'll see it expressed both ways). We define the quality of action $a$ as the expected reward from taking action $a$:

$$q_*(a) = \mathbb{E}[R_{t+1}|A_t=a]$$

Where the $*$ represents the optimal (true) of action value function, generally we don't have this so we denote it as $q(a)$. 

At any time step, there must be an estimated value that is the largest. Always choosing that one is known as *exploiting*, and to choose any non maximally estimated choices is *exploring*. Generally, the tradeoff between the 2 choices is short term gain in exploitation, or long term gain in exploration. 

## Action Value Estimation

So how do we update the action value function $q(a)$ so that it better approximates $q_*(a)$? One common idea is to keep a running average of the returns that occured from taking that action. The naive way to code this would be to keep an array of all past rewards and calculate the average every time step, which takes $\mathcal(O)(n)$ time and space per time step. A way to do this in constant time and space is to keep track of a count variable and update using:

$$
\begin{align}
Q_{n+1} &= \frac{1}{n} \sum^{n+1}_{i=1}R_i \\
&= \frac{1}{n} \big(R_{n+1} + (n-1)\frac{1}{n-1}\sum^{n}_{i=1}R_i \big)\\
&= \frac{1}{n} (R_{n+1} + (n-1)Q_n)\\
&= \frac{1}{n} (R_{n+1} + nQ_n - Q_n)\\
&= Q_n + \frac{1}{n} (R_{n+1} - Q_n)\\
\end{align}
$$

Here's a simple program that performs this update:

```python
q_a = np.array([0.0] * len(q))
n_a = np.array([0] * len(q))

for _ in range(n):
    action = np.random.randint(10)
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += (reward - q_a[action]) / n_a[action]
```

## $\epsilon$-greedy Methods

Given this, how do we decide on which action to take (above we just selected randomly)? If we use the greedy one, it might spend all its time exploiting a good action but not the *optimal* action. We can combat this by using $\epsilon-greedy$ methods, where we have a $1-\epsilon$ chance of taking the greedy action, and an $\epsilon$ chance to take any action at random (includes the greedy action). However, if we have a constant $\epsilon$, then even after an infinite number of iterations and convergence to the optimal value function, we will not take the optimal (greedy) action with more than $(1-\epsilon) + (1/n)\epsilon$ chance, where $n$ is the total number of actions possible.

```python
q_a = np.array([0.0] * len(q))
n_a = np.array([0] * len(q))

def greedy_epsilon(epsilon):
  for _ in range(1000):
    action = None
    if np.random.random() < 1 - epsilon:
      action = np.argmax(q_a)
    else:
      action = np.random.randint(10)
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += (reward - q_a[action]) / n_a[action]

greedy_epsilon(epsilon = 0.1)
```

## Optimistic Initialization

One way to improve this algorithm with a bit of knowledge about the rewards is to initialize the values of $q$ optimistically, which means to over estimate all of the expected rewards so that each reward is used once early on in the algorithm (since every initial value is eventually the greedy choice since they are higher than those of any true ones). In our example, we would just change the first line to:

```python
q_a = np.array([5.0] * len(q))
```

## Moving Rewards

Consider the situation where the rewards of the levers slowly change over the course of time steps. In this case, it would make sense to weigh recent rewards more compared to past rewards. So we weigh our update using a step-size $\alpha \in (0, 1]$ instead:

$$Q_{n+1} = Q_n + \alpha [R_n - Q_n]$$

We can also set $\alpha$ to be a function that changes over the time steps (e.g. setting it to $\frac{1}{n}$ makes it the sample average from above). 

```python
def alpha(action):
  # return 1/n_a[action]
  return 0.1

```

## Upper-Confidence Bound Action Selection

In $\epsilon$-greedy methods, we do exploration with no discernment of which non-greedy action should be tried. In Upper-Confidence Bound (UCB) action selection we take into account how close their estimates are to being maximal and how uncertain the estimate is using:

$$\underset{a}{\mathrm{argmax}} \bigg[q(a) + c \sqrt{\frac{ln(t)}{n_a(a)}}\bigg]$$

Where the $q(a)$ term measures the how maximal the estimate is, the $\sqrt{\frac{ln(t)}{n_a(a)}}$ term measures how uncertain the estimate is, and the $c$ controls the degree of exploration. Here's a simple implementation of it:

```python
def ucb(c):
  t = 0
  for _ in range(n):
    t += 1
    action = np.argmax([q_a[i] + c * np.sqrt(np.log(t)/np.max([n_a[i], 1])) for i in range(len(q_a))])
    reward = np.random.normal(q[action], std)
    n_a[action] += 1
    q_a[action] += alpha(action) * (reward - q_a[action])

ucb(c = 2)
```

## Gradient Bandits

Until now, we have been estimating the action value function $q(a)$ and then apply some method to choose an action to perform ($\epsilon$-greedy, UCB). We can also apply a function of numerical preference $H_t(a)$ to each action. Which we then put through as softmax function in order to obtain $\pi(a)$, a function for the probability of taking action $a$. This algorithm is based on the idea of Stochastic Gradient Descent (SGD). The preference function is updated with: 

$$
\begin{align}
H_{t+1}(A_t) &= H_t(A_t) + \alpha (R_t - \bar{R}_t)(1-\pi_t(A_t)) \\
H_{t+1}(a) &= H_t(a) + \alpha (R_t - \bar{R}_t)\pi_t(a) \\
\end{align}
$$

Where $\bar{R}_t$ is the average of all past rewards, $A_t$ is the action selected at time $t$, and all other actions update according to the second equation. Here's an implementation:

```python
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sgd(a):
  reward_avg = 0
  t = 0
  for _ in range(n):
    random_num = np.random.random()
    sm = softmax(q_a)
    action = None
    for i in range(len(sm)):
      random_num -= sm[i]
      if random_num <= 0:
        action = i
        break
    
    t += 1
    n_a[action] += 1
    reward = np.random.normal(q[action], std)
    reward_avg += 1/t * (reward - reward_avg)

    q_a[action] += a * (reward - reward_avg) * (1 - sm[action])
    for i in range(len(sm)):
      if i != action:
        q_a[i] -= a * (reward - reward_avg) * (sm[i])

sgd(0.1)
```

### Additional Notes

For a deeper dive on Artificial Intelligence, check out this [WaitButWhy][wbw] blog post.

[rlbook]: http://incompleteideas.net/book/RLbook2018.pdf
[wbw]: https://waitbutwhy.com/2015/01/artificial-intelligence-revolution-1.html
