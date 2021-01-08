## Introduction

A game is a collection of `n` *costs* (or *rewards*)
that agents try to *minimize* (or *maximize*).
These costs map from the space of all of the *actions* of the agents to a scalar value. 

We design and analyze learning algorithms that compute the equilibrium of the costs. 
These algorithms are based on the local first- and second-order gradients of the costs.

There are several equilibrium concepts that are significant in games. One equilibrium is where all agents are individually at a *local minimum* (or *maximum*). Another is when an agents is at a local optimum given that the other agent will perform optimally.
These equilibria can be verified with the second order derivatives of the costs and the Schur complement of the game Jacobian.


## Installation
Make sure to run the package in its own environment for testing.
Install using poetry:
```
cd learning-in-games
poetry install
```

Run python notebooks:
```
poetry run jupyter notebooks
```

Run tests:
```
poetry run python tests.py
```

## Formulation
Concretely, an `n`-player game denoted by `G = (f1, f2, ..., fn)` where 
1. costs are `fi: X -> R, i = 1, 2, ..., n` 
2. actions are `xi` in `Xi` which is a euclidean action space
3. action profile is `x = (x1, x2, ..., fn)` in `X = (X1, X2, ..., Xn)`

The learning algorithms are primarily gradient-based, so they can take advantage of auto diff software.

The following derivatives are useful:
1. game form `g = (D1f1,  D2f2,  ...,  Dnfn)`
2. game jacobian `J = [[D11f1,  D12f1],  [D21f2,  D22f2]]`
3. interaction terms `(D2f1,  D1f2)`

These operators are applied at action profile `x(t)` at time `t`, typically following
```x(t+1) = x(t) - lr * g(x(t))``` or some other history-dependent first order methods like acceleartion.

The joint gradients `g = (g1, g2, ..., gn)` is defined by some combination of the gradients of the costs `(f1, f2, ..., fn)`.

For example, gradient descent-ascent is `g = (D1f, -D2f)` whereas potential games are `g = (D1V, D2p)` for potential function `p:X->R`.


## Implementations

### Two-player games
We start with `n = 2`.
* Quadratic costs
* Linear dynamics, quadratic costs
* ames 
* Zero-sum games
* Potential games
* General-sum games

### *n*-player games
* time-varying bimatrix games
* grid world games

### Single player games
As a benchmark, we try to always include a single player scenario (optimization problems, LQR, classification, regression, etc).

