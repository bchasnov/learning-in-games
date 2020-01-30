A game *G* is defined by a tuple of objectives with a shared domain.
To solve a game, in some precise sense, is to find (a set of) points 
that satisfies optimality conditions of each of the objectives **individually**.
Concretely, an *n*-player game denoted by *G=(f1, f2, ... fn)* where 
1. player *i*'s action space is *Xi*,
2. the shared domain is *X=X1 x ... x Xn*, and
3. player *i*'s objective *fi* maps from the set *X* to a real number.

(*x* is the cartesian product and *R* is the real numbers)

In non-cooperative (selfish) games, the optimality conditions are defined with respect to 
the individual's objective over their own action space. That is, if the objective for players are 
to find (local) minima, then their choice *xi* in *Xi* must be a (local) minimum of *fi* holding all other agents' choice constant.

*Learning* in games describes the process with which players iterate sequentially
in these settings, in hopes of finding an `optimal' solution. 
