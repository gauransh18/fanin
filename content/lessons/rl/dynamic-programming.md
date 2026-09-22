---
summary: The two exact algorithms, why policy iteration converges in remarkably few steps, and the assumption that makes both unusable at scale.
prereqs: [value-functions-bellman]
---

Dynamic programming solves an MDP exactly when you know $P$ and $R$ and can enumerate
states. Both conditions fail in practice — but every model-free algorithm in this track is a
sampled approximation of one of these two, so the structure is worth having.

## Policy iteration

Alternate two steps until the policy stops changing:

**Evaluation** — compute $V^\pi$ for the current policy, by solving the linear system or by
iterating the Bellman expectation backup.

**Improvement** — act greedily with respect to $V^\pi$:

$$
\pi'(s) = \arg\max_a \sum_{s'} P(s'\mid s,a)\left[R + \gamma V^\pi(s')\right]
$$

```python
import numpy as np

def policy_iteration(P, R, gamma=0.9, tol=1e-10):
    S, A, _ = P.shape
    policy = np.zeros(S, dtype=int)
    for iteration in range(1000):
        # Evaluation: solve V = R_pi + gamma P_pi V exactly.
        P_pi = P[np.arange(S), policy]                    # (S, S)
        R_pi = R[np.arange(S), policy]                    # (S,)
        V = np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)

        # Improvement.
        Q = R + gamma * np.einsum('sat,t->sa', P, V)
        new_policy = Q.argmax(axis=1)
        if np.array_equal(new_policy, policy):
            return policy, V, iteration
        policy = new_policy
    return policy, V, iteration
```

::: key
**The policy improvement theorem**: acting greedily with respect to $V^\pi$ gives a policy
$\pi'$ with $V^{\pi'}(s) \ge V^\pi(s)$ for every state, with strict improvement somewhere
unless $\pi$ is already optimal.

Since there are finitely many deterministic policies and each iteration strictly improves,
policy iteration **terminates at the optimum in finite time** — no epsilon, no asymptotics.

In practice it converges in a handful of iterations, often fewer than ten even for large
state spaces. The cost is that each iteration solves an $S\times S$ system.
:::

## Value iteration

Skip the full evaluation. Apply one Bellman optimality backup per sweep:

$$
V_{k+1}(s) = \max_a \sum_{s'}P(s'\mid s,a)\left[R + \gamma V_k(s')\right]
$$

```python
import numpy as np

def value_iteration(P, R, gamma=0.9, tol=1e-8):
    S, A, _ = P.shape
    V = np.zeros(S)
    for sweep in range(100_000):
        Q = R + gamma * np.einsum('sat,t->sa', P, V)
        V_new = Q.max(axis=1)
        if np.abs(V_new - V).max() < tol * (1 - gamma) / (2 * gamma):
            return V_new, Q.argmax(1), sweep
        V = V_new
    return V, Q.argmax(1), sweep
```

The stopping criterion is not arbitrary. If successive sweeps differ by less than
$\epsilon(1-\gamma)/2\gamma$, then $\lVert V - V^*\rVert_\infty < \epsilon$ — a bound that
follows directly from the contraction property of lesson 6.02.

## Comparing them

| | Policy iteration | Value iteration |
|---|---|---|
| Per iteration | $O(S^3)$ — solve a linear system | $O(S^2A)$ — one backup |
| Iterations | Few (often under 10) | Many, $O(\log(1/\epsilon)/(1-\gamma))$ |
| Termination | Exact, finite | Asymptotic, needs a tolerance |
| Better when | $S$ small, $\gamma$ near 1 | $S$ large, $\gamma$ moderate |

The $\gamma$ dependence matters. Value iteration's convergence rate is $\gamma^k$, so at
$\gamma = 0.999$ it needs thousands of sweeps. Policy iteration is nearly insensitive to
$\gamma$, because exact evaluation handles the long horizon in one solve.

**Modified policy iteration** interpolates: run $k$ evaluation sweeps instead of solving
exactly. $k=1$ is value iteration, $k=\infty$ is policy iteration, and small $k$ is usually
the best of both.

## Generalised policy iteration

::: key
Nearly every algorithm in this track is an instance of one pattern: **make the value
function consistent with the policy, and make the policy greedy with respect to the value
function.** The two processes pull toward each other and their joint fixed point is the
optimum.

| Algorithm | Evaluation | Improvement |
|---|---|---|
| Policy iteration | Exact solve | Full greedy |
| Value iteration | One sweep | Implicit in the max |
| Monte Carlo (6.04) | Sampled returns | $\epsilon$-greedy |
| SARSA (6.05) | TD on the taken action | $\epsilon$-greedy |
| Q-learning (6.05) | TD with a max | Implicit in the max |
| Actor–critic (6.08) | Learned critic | Gradient step on the actor |

Recognising an algorithm's place in this table tells you most of what it does before you
read the details.
:::

## Asynchronous updates

Sweeping every state in order is wasteful. The contraction guarantee survives if you update
states in any order, provided every state is updated infinitely often.

That freedom is what makes the approach scalable in the directions that matter:

- **Prioritised sweeping** — update states whose Bellman error is largest first. Focuses
  effort where the value is most wrong.
- **Real-time DP** — update only the states the agent actually visits. The states a good
  policy visits are a tiny fraction of $\mathcal{S}$, and the rest do not need accurate
  values.

::: check
Dynamic programming solves an MDP exactly. Which two assumptions stop it being used in practice?

- [x] You must know $P$ and $R$
  > Every model-free method in this track exists because you usually do not.
- [x] You must be able to enumerate the states
  > A sweep touches every state, which is impossible for continuous or combinatorially large spaces.
- [ ] The policy must be deterministic
  > Policy iteration handles stochastic policies without difficulty.
- [ ] The reward must be bounded
  > Bounded rewards help the convergence argument and are not one of the two blocking assumptions.
:::

## The two blocking assumptions

**You must know $P$ and $R$.** In most problems you do not — you can only sample. Lesson
6.04's Monte Carlo and temporal-difference methods replace the expectation over $s'$ with
sampled transitions, which is the entire move from DP to RL.

**You must enumerate states.** $O(S^2A)$ per sweep is fine for $10^4$ states and impossible
for $10^{20}$. Function approximation replaces the table with a network, and lesson 6.06
covers what that costs — including the loss of the convergence guarantee that makes DP so
well-behaved.

::: exercise
You have an MDP with 10,000 states, 4 actions, and $\gamma = 0.999$. Which algorithm, and
why?
:::

::: solution
**The $\gamma$ is the deciding factor.**

**Value iteration's cost.** Convergence is $\gamma^k$, so reaching $\epsilon = 10^{-6}$
needs

$$
k \approx \frac{\ln(\epsilon(1-\gamma))}{\ln\gamma} = \frac{\ln(10^{-9})}{\ln 0.999} \approx 20{,}700 \text{ sweeps}
$$

Each sweep is $O(S^2A) = 4\times10^8$ operations, so roughly $8\times10^{12}$ total. Hours,
if the transition matrix is dense.

**Policy iteration's cost.** Each iteration solves a $10{,}000\times10{,}000$ system:
$O(S^3) = 10^{12}$ operations, a few minutes with a good LAPACK. It typically converges in
under 10 iterations regardless of $\gamma$, so about $10^{13}$ operations total — the same
order, but with a *finite termination guarantee* rather than a tolerance.

**So it is close on raw operations, and policy iteration wins on two counts:** it terminates
exactly, and its iteration count does not degrade as $\gamma \to 1$.

**But the real answer is modified policy iteration.** Run $k \approx 20$ evaluation sweeps
instead of an exact solve:

- Per iteration: $20 \times O(S^2A) = 8\times10^9$ — **125× cheaper** than the $O(S^3)$
  solve.
- Iterations: still small, typically 10–20, because partial evaluation is enough to drive
  good improvement.
- Total: around $10^{11}$ operations. An order of magnitude better than either extreme.

**One thing to check first:** is $P$ sparse? With 4 actions, most states likely reach only a
handful of successors. If so, store $P$ as a sparse matrix — $O(S^2A)$ becomes
$O(S \cdot A \cdot \bar{d})$ with $\bar{d}$ the branching factor, which at $\bar{d} = 4$ is
$1.6\times10^5$ per sweep rather than $4\times10^8$. That is a 2,500× speedup and it makes
plain value iteration perfectly comfortable.

**Ordering the decision:** check sparsity first — it dominates the algorithm choice. Then
use modified policy iteration with $k \approx 20$. Reserve exact policy iteration for small
$S$ where the $O(S^3)$ solve is cheap.
:::

## What to carry forward

- Policy iteration alternates exact evaluation and greedy improvement, terminating in finitely many steps.
- Value iteration does one backup per sweep and converges at rate $\gamma^k$.
- Modified policy iteration interpolates and is usually best.
- Generalised policy iteration is the pattern behind nearly every algorithm in this track.
- DP needs a known model and enumerable states; removing each of those is the rest of the track.
