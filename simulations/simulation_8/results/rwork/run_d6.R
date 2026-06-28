Y <- as.matrix(read.csv("/home/willwhite/GitHub/gllvm/simulations/simulation_8/results/rwork/Y_d6.csv", header=FALSE))
suppressPackageStartupMessages(library(gllvm))
fit <- gllvm(Y, num.lv=2, family="poisson", method="VA", sd.errors=TRUE,
             control=list(TMB=TRUE, maxit=6000, trace=FALSE))
sig <- fit$params$sigma.lv
W   <- sweep(fit$params$theta, 2, sig, "*")
seW <- sweep(fit$sd$theta,     2, sig, "*")
write.csv(W,   "/home/willwhite/GitHub/gllvm/simulations/simulation_8/results/rwork/W_d6.csv", row.names=FALSE)
write.csv(seW, "/home/willwhite/GitHub/gllvm/simulations/simulation_8/results/rwork/SE_d6.csv", row.names=FALSE)
cat("done\n")