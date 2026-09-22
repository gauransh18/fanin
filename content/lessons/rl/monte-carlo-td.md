---
summary: Learning values from experience instead of a model — the bias-variance tradeoff between waiting for the outcome and bootstrapping from a guess.
prereqs: [dynamic-programming, expectation-variance-concentration]
---

Dynamic programming needs $P$. Monte Carlo and temporal-difference methods need only
experience — sequences of states, actions and rewards. The choice between them is a
bias–variance tradeoff, and it recurs everywhere in RL.

## Monte Carlo

Wait until the episode ends, then use the actual return:

$$
V(s_t) \leftarrow V(s_t) + \alpha\left[G_t - V(s_t)\right], \qquad G_t = \sum_{k=0}^{T-t}\gamma^k r_{t+k}
$$

```python
from collections import defaultdict

def mc_prediction(env, policy, episodes=10_000, gamma=0.99, alpha=0.05):
    V = defaultdict(float)
    for _ in range(episodes):
        trajectory, state, done = [], env.reset(), False
        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action)
            trajectory.append((state, reward))
            state = next_state

        G = 0.0
        for state, reward in reversed(trajectory):        # backwards: G accumulates
            G = reward + gamma * G
            V[state] += alpha * (G - V[state])
    return V
```

**Unbiased** — $G_t$ is a real sample of the return, so $\mathbb{E}[G_t] = V^\pi(s_t)$
exactly. **High variance** — it depends on every random choice for the rest of the episode.
And it requires episodes to **terminate**, which rules out continuing tasks.

## Temporal difference

Do not wait. After one step, bootstrap from the current estimate of the next state:

$$
V(s_t) \leftarrow V(s_t) + \alpha\left[\underbrace{r_t + \gamma V(s_{t+1})}_{\text{TD target}} - V(s_t)\right]
$$

The bracketed quantity is the **TD error** $\delta_t$, and it appears in essentially every
algorithm from here on.

```python
from collections import defaultdict

def td_prediction(env, policy, episodes=10_000, gamma=0.99, alpha=0.1):
    V = defaultdict(float)
    for _ in range(episodes):
        state, done = env.reset(), False
        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action)
            target = reward + gamma * V[next_state] * (not done)
            V[state] += alpha * (target - V[state])       # update immediately
            state = next_state
    return V
```

**Biased** — the target uses $V(s_{t+1})$, which is currently wrong. **Low variance** — it
depends on one transition, not a whole trajectory. And it works **online** and on continuing
tasks.

::: key
| | Monte Carlo | TD(0) |
|---|---|---|
| Bias | None | Yes, from bootstrapping |
| Variance | High | Low |
| Needs termination | Yes | No |
| Updates | End of episode | Every step |
| Uses the Markov property | No | Yes |

That last row is underappreciated. MC estimates the return directly and does not care
whether the state is Markov. TD's target assumes $V(s_{t+1})$ summarises the future, which
is only true under the Markov property. **In a non-Markov environment, MC is more robust and
TD is biased in a way that does not vanish with more data** — a useful diagnostic when TD
plateaus and MC does not.
:::

TD usually wins in practice: lower variance dominates, and the bias shrinks as $V$ improves.

::: check
Monte Carlo value estimation is unbiased. What does it cost?

- [x] High variance, and it needs episodes to terminate at all
  > The return depends on every random choice for the rest of the episode. TD(0) trades the other way: it bootstraps from the current estimate, so it is biased and low variance, and works on continuing tasks. Everything between is the $n$-step spectrum.
- [ ] Bias that grows with the discount factor
  > It has no bias. That is the one thing it guarantees.
- [ ] It cannot be used with function approximation
  > It can, and in fact it is the safest of the three legs of the deadly triad to combine with approximation.
- [ ] It requires knowing the transition model
  > It needs only experience. Needing the model is dynamic programming's problem.
:::

## n-step methods

The two are endpoints of a spectrum. Use $n$ real rewards, then bootstrap:

$$
G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1}r_{t+n-1} + \gamma^n V(s_{t+n})
$$

$n=1$ is TD(0); $n=\infty$ is Monte Carlo. Intermediate $n$ — typically 3 to 10 — usually
beats both, and this is exactly what GAE generalises in lesson 6.08.

::: check
What are the three ingredients of the deadly triad?

- [x] Function approximation
  > A table cannot diverge this way; a network generalising between states can.
- [x] Bootstrapping
  > Updating an estimate towards another estimate is what lets an error feed back into itself.
- [x] Off-policy training
  > Learning about one policy from another's data removes the corrective feedback that would fix an overestimate.
- [ ] A discount factor below 1
  > Discounting is what makes returns well behaved; it is not a source of instability.
- [ ] Stochastic transitions
  > Stochasticity adds variance, not divergence. Deterministic environments hit the triad too.
:::

## TD(λ) and eligibility traces

Rather than picking one $n$, average all of them with geometrically decaying weights:

$$
G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}
$$

Computing this forward requires the future. The **backward view** computes it exactly, in
$O(1)$ per step, using an eligibility trace:

```python
from collections import defaultdict

def td_lambda(env, policy, episodes=1000, gamma=0.99, lam=0.9, alpha=0.1):
    V, e = defaultdict(float), defaultdict(float)
    for _ in range(episodes):
        e.clear()
        state, done = env.reset(), False
        while not done:
            action = policy(state)
            next_state, reward, done = env.step(action)
            delta = reward + gamma * V[next_state] * (not done) - V[state]

            e[state] += 1                                 # this state is eligible
            for s in list(e):                             # credit ALL recent states
                V[s] += alpha * delta * e[s]
                e[s] *= gamma * lam
                if e[s] < 1e-6:
                    del e[s]
            state = next_state
    return V
```

The trace records how recently each state was visited. A TD error updates every recently
visited state in proportion to its eligibility, which spreads credit backwards through time
without storing trajectories.

## Control: from prediction to action

For control, learn $Q$ and act $\epsilon$-greedily on it:

```python
import random
from collections import defaultdict

def epsilon_greedy(Q, state, n_actions, eps):
    if random.random() < eps:
        return random.randrange(n_actions)
    q = [Q[(state, a)] for a in range(n_actions)]
    return max(range(n_actions), key=lambda a: q[a])
```

$\epsilon$-greedy is the minimal exploration mechanism: act greedily most of the time, act
randomly otherwise. Lesson 6.11 covers why this is weak and what replaces it.

## The deadly triad

::: warning
Three ingredients, each individually reasonable, can diverge when combined:

1. **Function approximation** — a network instead of a table.
2. **Bootstrapping** — a target built from your own estimate.
3. **Off-policy learning** — learning about a policy other than the one acting.

With all three, the value function can diverge without bound, and this is not rare. It is
the root cause of most instability in deep RL, and it is why DQN (lesson 6.06) needs target
networks and replay buffers — both are mitigations for the triad, not incidental tricks.

Drop any one of the three and convergence is much better behaved. Monte Carlo drops
bootstrapping; on-policy methods drop the third; tabular methods drop the first.
:::

::: exercise
You estimate $V$ for a policy in an environment where episodes last 1,000 steps and rewards
arrive only at the end. Compare MC and TD(0), and say which you would use.
:::

::: solution
**Monte Carlo.** Each update uses the full return — one number per episode, unbiased. But
the variance is enormous: the return depends on 1,000 stochastic transitions, so the
estimator's standard deviation grows with the horizon. With a terminal reward, every state
in the episode receives the same return, so there is no discrimination between early states
within an episode. You need many episodes to separate them, and you can only update once per
episode.

**TD(0).** With rewards only at the end, $r_t = 0$ for every intermediate step, so the
update is $V(s_t) \leftarrow V(s_t) + \alpha[\gamma V(s_{t+1}) - V(s_t)]$. Information about
the terminal reward propagates **one step per episode**: after one episode only $s_{999}$
has a nonzero value, after two episodes $s_{998}$ does, and so on.

Reaching $s_0$ takes roughly 1,000 episodes. TD(0) is correct and unusably slow here.

**Neither, on its own. Use $n$-step or TD(λ).**

With $n = 50$, credit propagates 50 states per episode and reaches $s_0$ in about 20
episodes — a 50× speedup over TD(0) — while variance is still far below Monte Carlo's,
because you bootstrap after 50 steps instead of 1,000.

TD(λ) with $\lambda \approx 0.95$ is better still: the eligibility trace credits every
recently visited state on every TD error, so information propagates through the whole
trajectory within a single episode, at $O(1)$ cost per step.

**How to choose $\lambda$:** the effective horizon is $1/(1-\lambda)$ steps. For a
1,000-step episode with terminal reward, $\lambda = 0.99$ gives a 100-step horizon, which
balances propagation speed against variance. Sweep it — the optimum is task-dependent and
the curve is usually flat near the top.

**The general rule:** sparse, delayed reward pushes you toward larger $n$ or higher
$\lambda$. Dense reward lets you use TD(0). This is the same tradeoff GAE's $\lambda$
controls in lesson 6.08, and for the same reason.
:::

## What to carry forward

- MC is unbiased with high variance; TD is biased with low variance and works online.
- TD assumes the Markov property; MC does not — a useful diagnostic when TD stalls.
- $n$-step and TD(λ) interpolate, and usually beat both endpoints.
- Eligibility traces implement the λ-return exactly at $O(1)$ per step.
- Function approximation + bootstrapping + off-policy is the deadly triad, and it can diverge.
