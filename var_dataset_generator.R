library(tsDyn)

results = list()
B1<-matrix(c(1, 0, 0, 1), 2)

for(i in 1:100) {
  results[[toString(i)]] <- VAR.sim(B=B1,n=100,include="none")
}

results_together <- do.call(rbind,lapply(names(results),function(x){
  transform(as.data.frame(results[[x]]), Name = x)
}))

write.csv(results_together, file="Desktop/variables_2_covar_identity.csv")