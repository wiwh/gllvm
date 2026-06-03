# GLM FIT

Let X be of size p x q (p > q) fixed design matrix
Let Y be of size p x n Poisson rv, written as Y1, ..., Yn where where

Yi   = X beta_i

beta _i of dimension qx1

Goal: estimate all the beta_i in a matrix beta_1, ..., beta_n. of size n xq.

Method: start with the gaussian solve on log(1+Yi). Then proceed with newton method, being very careful about not doing computations twice.
You'll need to solve n linear systems per iteration.... this can be done batched with pytorch.


Do everything with pytorch.

Dimensions: n, p, q
X ~ N(0, alpha^2) all independent: alpha is a parameter to choose
We assume there is no intercept, but there can be an offset, which is for now 0.
Regularize with 1/2 ||beta_i^2||_2. 

Good luck! that's very simple but do it well.

Create a new glm_fit_newton.ipynb for that.