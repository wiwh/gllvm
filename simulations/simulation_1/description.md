# Poisson simulations in gllvm's turf

The goal is to compare the gaussian procrustes , bias, variance of estimation across a sweep of conditions.

We use the /simulations/poisson.ipynb notebook as guide for setup and simulations.

Fully dense.

We go like this:

q = 2
p = 10, 20, 50, 100
n = 20, 100, 500

=12 settings
each repeated H = 20 times

we want a wrapper for this simulation (not in the source code, specific for this simulation)

wrapper takes: q, p, n, seed (for the simulation)

returns: 

time it took for each fit

ZQE: T=log1p, for both decoder and encoder, gaussian map encoder, specification as in simulations/poisson.ipynb
gllvm: the R wrapper we currently use

for each method and each true model, all parameters stored flattened, after procrustes rotation: the loadings as wella s the intercepts.


i dunno how best to save this, you will know better. it needs to be easily reproducible. maybe each setting creates a csv or parquet file or whatever is good with the results, as well as a file that explains what each column is. I DO NOT KNOW. you figure it out the best.


Make it so that i can increase to H = 100 if needed, without requiring to redo the first 20, see? ok good.

## analysis
once done, we want a notebook for the analysis of the result

somehow i want plot of procrustes error across the sweep, but also bias and variance. maybe some boxplots or stuff would be nice. 