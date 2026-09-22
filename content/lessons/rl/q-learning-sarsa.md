---
summary: The on-policy/off-policy distinction made concrete by one symbol, and the cliff-walking example that shows what it means.
prereqs: [monte-carlo-td, dynamic-programming]
---

SARSA and Q-learning differ in exactly one term of their update. That difference is the
on-policy/off-policy distinction, and it produces visibly different behaviour.

## SARSA

Named for the tuple it uses: $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$.

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t,a_t)\right]
$$

The target uses $a_{t+1}$ — the action the policy **actually took**. So SARSA learns the
value of the policy it is following, $\epsilon$-greedy exploration included.

## Q-learning

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_t + \gamma\max_{a'}Q(s_{t+1},a') - Q(s_t,a_t)\right]
$$

The target uses $\max_{a'}$ — the **best** action, regardless of what was taken. So
Q-learning learns $Q^*$, the optimal action-value function, while behaving however it likes.

::: key
$Q(s_{t+1}, a_{t+1})$ versus $\max_{a'}Q(s_{t+1},a')$. That is the whole difference.

**SARSA is on-policy**: it evaluates and improves the policy it is executing, so the
learned values account for the cost of exploration.

**Q-learning is off-policy**: it learns about the greedy policy while executing another.
This is what lets it learn from a replay buffer, from a human demonstration, or from any
other data source — which is why it scales to deep RL.
:::

```python
import random
from collections import defaultdict

def q_learning(env, n_actions, episodes=5000, alpha=0.5, gamma=0.99, eps=0.1):
    Q = defaultdict(float)
    for _ in range(episodes):
        state, done = env.reset(), False
        while not done:
            action = epsilon_greedy(Q, state, n_actions, eps)
            next_state, reward, done = env.step(action)
            best_next = 0.0 if done else max(Q[(next_state, a)] for a in range(n_actions))
            Q[(state, action)] += alpha * (reward + gamma * best_next - Q[(state, action)])
            state = next_state
    return Q


def sarsa(env, n_actions, episodes=5000, alpha=0.5, gamma=0.99, eps=0.1):
    Q = defaultdict(float)
    for _ in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, n_actions, eps)
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy(Q, next_state, n_actions, eps)
            target = reward + gamma * (0.0 if done else Q[(next_state, next_action)])
            Q[(state, action)] += alpha * (target - Q[(state, action)])
            state, action = next_state, next_action        # commit to the sampled action
    return Q
```

Note the structural difference in the loops: SARSA must choose the next action *before*
updating, because it needs it for the target.

## Cliff walking

The canonical demonstration. A grid with a cliff along the bottom edge: stepping into it
costs $-100$ and resets to the start. Every other step costs $-1$. The goal is the far
corner.

- **Q-learning learns the optimal path** — hugging the cliff edge, which is shortest.
- **SARSA learns a safer path** — one row further from the cliff.
- **During training, SARSA earns more reward.**

Q-learning learns the value of the greedy policy, which never falls off the cliff. But it is
*executing* an $\epsilon$-greedy policy, which does fall off — 10% of the time near the
edge, at $-100$ each. Its learned values do not account for that.

SARSA's targets include the exploratory action, so it learns that the cliff-edge path is
dangerous *for a policy that sometimes acts randomly*, and it routes around.

::: insight
Neither is wrong. They answer different questions.

**Q-learning** answers "what is the best policy?" — correct if you will act greedily at
deployment.

**SARSA** answers "what is the best policy given that I will keep exploring?" — correct if
exploration continues in deployment, or if mistakes during learning are expensive.

For a robot that breaks when it falls, or an agent acting in a live system, SARSA's question
is the right one.
:::

## Expected SARSA

Replace the sampled next action with its expectation under the policy:

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha\left[r_t + \gamma\sum_{a'}\pi(a'\mid s_{t+1})Q(s_{t+1},a') - Q(s_t,a_t)\right]
$$

This removes the variance from sampling $a_{t+1}$ while keeping SARSA's on-policy
semantics. It costs an expectation over actions per step — negligible for small action
spaces — and generally outperforms both SARSA and Q-learning.

It also unifies them: with a greedy $\pi$, the expectation becomes the max and Expected
SARSA *is* Q-learning.

::: check
SARSA and Q-learning differ in one term. What is the consequence for what each learns?

- [x] SARSA's target uses the action actually taken, so it learns the value of the $\epsilon$-greedy policy including the cost of exploring; Q-learning's uses $\max_{a'}$, so it learns $Q^*$ while behaving however it likes
  > Being off-policy is what lets Q-learning train from a replay buffer, a human demonstration, or any other data source — which is why it scales to deep RL.
- [ ] SARSA converges faster because it bootstraps from a single action
  > Both bootstrap from one step. Convergence speed is not the distinction.
- [ ] Q-learning is on-policy and SARSA off-policy
  > The other way around: the $\max$ is what makes Q-learning off-policy.
- [ ] They learn the same values and differ only in variance
  > On a cliff-walking task they learn visibly different policies — SARSA takes the safe route, Q-learning the optimal one it cannot safely execute.
:::

## Maximisation bias

::: warning
Q-learning's $\max$ is biased upward. $\mathbb{E}[\max_a X_a] \ge \max_a \mathbb{E}[X_a]$ by
Jensen's inequality, so taking the max over noisy estimates systematically overestimates.

With $n$ actions whose true values are all equal and estimates that are unbiased with noise
$\sigma$, the max overestimates by roughly $\sigma\sqrt{2\ln n}$. Early in training, when
estimates are noisy, this is large.

The consequence is a self-reinforcing loop: overestimated actions are selected more, which
gives them more updates, which... The effect is worse in deep RL where function
approximation adds correlated noise.
:::

**Double Q-learning** fixes it by decoupling selection from evaluation. Keep two estimates
and use one to pick the action and the other to value it:

```python
def double_q_update(Q1, Q2, s, a, r, s_next, done, n_actions, alpha, gamma):
    if random.random() < 0.5:
        best = max(range(n_actions), key=lambda x: Q1[(s_next, x)])   # select with Q1
        target = r + gamma * (0 if done else Q2[(s_next, best)])      # evaluate with Q2
        Q1[(s, a)] += alpha * (target - Q1[(s, a)])
    else:
        best = max(range(n_actions), key=lambda x: Q2[(s_next, x)])
        target = r + gamma * (0 if done else Q1[(s_next, best)])
        Q2[(s, a)] += alpha * (target - Q2[(s, a)])
```

Because the two estimates have independent noise, the one used for evaluation does not share
the error that made the selected action look best. Double DQN (lesson 6.06) is this idea
applied to neural networks.

## Convergence

Tabular Q-learning converges to $Q^*$ with probability 1, given that every state-action pair
is visited infinitely often and the learning rate satisfies the Robbins–Monro conditions:

$$
\sum_t \alpha_t = \infty, \qquad \sum_t \alpha_t^2 < \infty
$$

$\alpha_t = 1/t$ satisfies both. A constant $\alpha$ satisfies the first and not the second,
so it converges to a noise ball rather than a point — the same phenomenon as constant-step
SGD in lesson 1.16, and the same reason you decay.

These guarantees vanish with function approximation. Lesson 6.04's deadly triad is why.

::: exercise
You train an agent to control a chemical process where an unsafe action causes an expensive
shutdown. You will deploy it with $\epsilon = 0$. Which algorithm?
:::

::: solution
**The two clauses pull in opposite directions, and the training-time cost decides it.**

**Deployment at $\epsilon = 0$ argues for Q-learning.** You will act greedily, so you want
$Q^*$ — the value of the greedy policy. SARSA would learn a policy that is conservative
about exploration you are not going to do, leaving performance on the table.

**Expensive training-time mistakes argue for SARSA.** During learning you *are* exploring,
and Q-learning's values do not price in the cost of an exploratory action near a dangerous
state. It will happily learn a policy that operates close to the unsafe boundary, and while
learning it will step over it.

**The resolution: neither, alone.** For a system with expensive failures, on-policy versus
off-policy is the wrong axis to optimise. Do this instead:

1. **Train in simulation with Q-learning.** Exploration is free there, and you get $Q^*$ —
   the right target for greedy deployment. This handles both clauses if the simulator is
   adequate, and it is the standard answer.

2. **Add a safety layer, not a softer objective.** A hard constraint that vetoes actions
   predicted to enter the unsafe region is far more reliable than hoping a learned value
   function has internalised the danger. Constrained MDPs and shielded RL formalise this.
   Never rely on a learned value to enforce a safety property.

3. **If you must learn on the real system**, use SARSA or Expected SARSA *and* restrict
   exploration to a verified-safe action set. SARSA's conservatism is a useful secondary
   defence, not a primary one.

4. **Consider offline RL** (lesson 6.13). If you have logs of the process under existing
   controllers, learn from those without any exploration at all. This is often the right
   answer for industrial control and is frequently overlooked.

**The framing that matters:** "which algorithm is safer" is the wrong question. Safety comes
from constraints and simulators, not from the choice between $\max_{a'}$ and $a_{t+1}$. The
algorithm choice then follows from the deployment condition — which here is Q-learning,
because you will act greedily.
:::

## What to carry forward

- SARSA's target uses the action taken; Q-learning's uses the best action.
- On-policy learns the value of what you do; off-policy learns the value of what is optimal.
- SARSA's conservatism near cliffs is correct behaviour, not a defect.
- The max is biased upward; Double Q-learning decouples selection from evaluation.
- Tabular convergence needs decaying $\alpha$; constant $\alpha$ gives a noise ball.
