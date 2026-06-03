# L1 loss experimentation

now we have plenty of loadings 0 in true model

we don't know which one are which.

but we KNOW the loading matrix is sparse. this we know

we want a method, maybe l1 penalization., or another method for sparse estimation, that will make most loadings 0


in essence we are interested in learning the loadings, but also which are 0.

the current notebook zqe_gaussian_map works quite well for poisson data. use it as reference, maybe add l1 norm, i dunno. experiment. 

You need to:
1. FOR THE TRUE MODEL ONLY (ONLY !!!), create a "sparse mask"
2. THE MODEL THAT WE FIT should not know about this sparse mask. we will use the method we develop here to recover sparsity and hopefuly find the true nonzero loadings. propose a solution

implement time! modify the zqe_gaussian_map_sparse.ipynb notebook to implement that in an easy way... just testing. and discuss if this is worth pursuing. 

THANKS!