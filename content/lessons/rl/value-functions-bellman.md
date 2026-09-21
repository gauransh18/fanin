---
summary: The two value functions, the recursive equations that define them, and the contraction argument that guarantees the algorithms converge.
prereqs: [markov-decision-processes, eigenvalues-eigenvectors]
---

Value functions answer "how good is this situation". They are the central object in RL
because almost every algorithm either estimates one or uses one to reduce variance.

## Two functions

**State value** — expected return from $s$ under policy $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k} \;\middle|\; s_t = s\right]
$$

**Action value** — expected return from taking $a$ in $s$, then following $\pi$:

$$
Q^\pi(s,a) = \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k} \;\middle|\; s_t=s, a_t=a\right]
$$

They are related by $V^\pi(s) = \mathbb{E}_{a\sim\pi}[Q^\pi(s,a)]$.

$Q$ is the more useful of the two for control, because it lets you act greedily without a
model: $\pi(s) = \arg\max_a Q(s,a)$. Acting greedily on $V$ requires knowing $P$ to see
where each action leads.

## The Bellman expectation equations

Value has a recursive structure — the value of now is the reward now plus the discounted
value of next:

$$
V^\pi(s) = \sum_a \pi(a\mid s)\sum_{s'} P(s'\mid s,a)\left[R(s,a,s') + \gamma V^\pi(s')\right]
$$

$$
Q^\pi(s,a) = \sum_{s'} P(s'\mid s,a)\left[R(s,a,s') + \gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\right]
$$

These are **linear** systems in the unknowns. For small state spaces you can solve them
exactly.

## The Bellman optimality equations

For the optimal policy, replace the expectation over actions with a maximum:

$$
V^*(s) = \max_a \sum_{s'} P(s'\mid s,a)\left[R(s,a,s') + \gamma V^*(s')\right]
$$

$$
Q^*(s,a) = \sum_{s'} P(s'\mid s,a)\left[R(s,a,s') + \gamma\max_{a'} Q^*(s',a')\right]
$$

The $\max$ makes these **nonlinear**, so they cannot be solved by linear algebra. They are
solved by iteration — which is where the convergence guarantee comes in.

## Why iteration converges

Define the Bellman optimality operator $\mathcal{T}$:

$$
(\mathcal{T}V)(s) = \max_a \sum_{s'}P(s'\mid s,a)\left[R + \gamma V(s')\right]
$$

::: key
$\mathcal{T}$ is a **$\gamma$-contraction** in the sup norm:

$$
\lVert \mathcal{T}V_1 - \mathcal{T}V_2\rVert_\infty \le \gamma\lVert V_1 - V_2\rVert_\infty
$$

By the Banach fixed-point theorem, a contraction on a complete space has a **unique** fixed
point, and iterating from any starting point converges to it geometrically.

Two consequences you can rely on. The optimal value function exists and is unique. And
value iteration converges from any initialisation at rate $\gamma^k$ — with error after $k$
sweeps bounded by $\gamma^k \lVert V_0 - V^*\rVert_\infty / (1-\gamma)$.

This is the same geometric-decay argument as lesson 1.05's spectral radius, and $\gamma$ is
playing the role of $\rho$.
:::

The proof is short. For any $s$, $|(\mathcal{T}V_1)(s) - (\mathcal{T}V_2)(s)|$ is at most
the largest over $a$ of $\gamma\sum_{s'}P(s'|s,a)|V_1(s')-V_2(s')|$, which is at most
$\gamma\lVert V_1-V_2\rVert_\infty$ since the probabilities sum to one.

## Solving a small MDP exactly

```python
import numpy as np

def policy_evaluation_exact(P, R, policy, gamma=0.9):
    """P: (S, A, S). R: (S, A). policy: (S, A). Solves V = R_pi + gamma P_pi V."""
    S = P.shape[0]
    P_pi = np.einsum('sa,sat->st', policy, P)        # (S, S)
    R_pi = np.einsum('sa,sa->s', policy, R)          # (S,)
    return np.linalg.solve(np.eye(S) - gamma * P_pi, R_pi)


def value_iteration(P, R, gamma=0.9, tol=1e-8):
    S, A, _ = P.shape
    V = np.zeros(S)
    for sweep in range(10_000):
        Q = R + gamma * np.einsum('sat,t->sa', P, V)
        V_new = Q.max(axis=1)
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < tol:
            break
    return V, Q.argmax(axis=1), sweep
```

`np.linalg.solve` costs $O(S^3)$, so exact evaluation is fine up to a few thousand states
and hopeless beyond. Value iteration costs $O(S^2A)$ per sweep — better, and still requiring
enumeration of every state.

That enumeration is the wall. Backgammon has $10^{20}$ states; Go has $10^{170}$; a language
model's state space is every possible token sequence. Function approximation — replacing the
table with a neural network — is what the rest of this track is about, and it is where the
convergence guarantee is lost.

## Advantage

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
$$

How much better than average is this action, in this state. Note $\mathbb{E}_{a\sim\pi}[A^\pi(s,a)] = 0$
by construction.

Advantage is what policy gradient methods actually use (lesson 6.08), for a reason that is
easy to state: absolute returns vary enormously across states — being in a good state gives
high return regardless of what you do — while the advantage isolates the part attributable
to the *action*. Subtracting $V$ removes a large, action-independent term and dramatically
reduces gradient variance.

This is also why lesson 5.08's RLHF uses advantages rather than raw rewards: the reward
model's absolute scale is not identified (lesson 5.07), so only differences carry
information anyway.

::: exercise
A gridworld gives $-1$ per step and $0$ at the goal. With $\gamma = 1$, is $V^*$
well-defined? What changes at $\gamma = 0.99$?
:::

::: solution
**At $\gamma = 1$, it depends on whether the goal is reachable.**

From any state with a path to the goal, $V^*(s) = -d(s)$ where $d$ is the shortest-path
distance — a finite number, because the optimal policy terminates. So $V^*$ is well-defined
on the reachable set.

From a state with no path to the goal, the return is $-1$ at every step forever:
$V^*(s) = -\infty$. Undefined in any useful sense.

More subtly, the contraction argument **does not apply** at $\gamma = 1$: $\mathcal{T}$ is
non-expansive rather than contractive, so uniqueness and geometric convergence are not
guaranteed. Value iteration may still converge here because the problem is an episodic
shortest-path task, but the guarantee comes from a different argument (proper policies),
not from Banach.

**At $\gamma = 0.99$, everything is well-defined.** Rewards are bounded by 1, so

$$
|V^*(s)| \le \frac{1}{1-\gamma} = 100
$$

for every state, including unreachable ones — an unreachable state has value exactly
$-100$. The contraction holds, the fixed point is unique, and value iteration converges
geometrically at rate $0.99^k$.

**The practical tradeoff.** $\gamma = 0.99$ gives an effective horizon of 100 steps. If the
goal is 150 steps away, the discounted value of reaching it is $0.99^{150} \approx 0.22$ of
its undiscounted value — the agent may prefer a shorter path to a worse outcome, or fail to
see the goal at all. Set $\gamma$ so that $1/(1-\gamma)$ comfortably exceeds your task's
horizon.

**What people actually do:** use $\gamma$ slightly below 1 for the guarantees, and cap
episode length so unreachable states terminate rather than accruing infinite cost. RLHF's
$\gamma = 1$ is safe precisely because episodes are finite by construction.
:::

## What to carry forward

- $V$ evaluates states, $Q$ evaluates state-action pairs and lets you act without a model.
- Bellman expectation equations are linear; optimality equations are nonlinear and need iteration.
- The Bellman operator is a $\gamma$-contraction, so the fixed point is unique and iteration converges at $\gamma^k$.
- Exact methods need to enumerate states, which is why function approximation is unavoidable.
- Advantage $Q - V$ removes a large action-independent term and is what policy gradients use.
