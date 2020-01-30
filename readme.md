A game *G* is defined by a tuple of objectives with a shared domain.
To solve a game, in some precise sense, is to find (a set of) points 
that satisfies optimality conditions of each of the objectives **individually**.
Concretely, an *n*-player game denoted by *G=(f1, f2, ... fn)* where 
1. player *i*'s action space is *Xi*,
2. the shared domain is *X=X1 x ... x Xn* where *x* is the cartesian product, and
3. player *i*'s objective *fi* maps from the set *X* to a real number.

In non-cooperative (selfish) games, the optimality conditions are defined with respect to 
the individual's objective and their action space. If the objective is
to find (local) minima, then their action *xi* in *Xi* must be a (local) minimum of *fi* holding all other agents' choice constant.
*Learning* in games describes the process with which players iterate sequentially
in these settings, in hopes of finding such an 'optimal' solution. 
Various solution concepts arise from different structures imposed on the game. 
A common baseline concept in these settings are Nash equilibria. [1]


[1] Fudenberg, Drew, and Levine, David. "The theory of learning in games." MIT press (1998).
[2] Nash, John. "Non-cooperative games." Annals of mathematics (1951)
