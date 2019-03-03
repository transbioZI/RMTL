## ----setup, include = FALSE----------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ---- echo=FALSE, results='markup'---------------------------------------
my.df <- data.frame(Omega=c("$||W||_1$", "$||W||_{2,1}$", "$||W||_*$", "$||WG||_F^2$", "$tr(W^TW)-tr(F^TW^TWF)$"),
                    Regularization=c("**Lasso**", "**L21**", "**Trace**", "**Graph**", "**CMTL**"),  
                    Problem=c("R/C", "R/C", "R/C", "R/C", "R/C"), 
                    lam1=c("$\\lambda_1 > 0$", "$\\lambda_1 > 0$", "$\\lambda_1 > 0$", "$\\lambda_1 \\geq 0$", 
                           "$\\lambda_1 > 0$"),
                    lam2=c("$\\lambda_2 \\geq 0$", "$\\lambda_2 \\geq 0$", "$\\lambda_2 \\geq 0$", "$\\lambda_2 \\geq 0$", 
                           "$\\lambda_2 > 0$"),
                    Extra=c("None", "None", "None", "G", "k"),
                    warm=c("Yes", "Yes", "Yes", "No", "No"), 
                    reference=c("[@Tibshirani1996]", "[@Jun2009]", "[@Pong2010]", "[@Widmer2014]", "[@Jiayu2011]"))
rownames(my.df) = c("sparse structure", "joint feature learning", "low-rank structure", "network incorperation", "task clustering")
colnames(my.df) = c("$\\Omega(W)$", "Regularization Type", "Problem Type", "$\\lambda_1$", "$\\lambda_2$", "Extra Input", "Warm Start", 
                    "Reference")

knitr::kable(t(my.df), caption="Table 1: Summary of Algorithms in RMTL", format="pandoc")

## ---- fig.width=4,fig.height=3, fig.cap="Figure 1: CV errors across the sequence of $\\lambda_1$"----
#create simulated data
library(RMTL)
datar <- Create_simulated_data(Regularization="L21", type="Regression")

#perform the cross validation
cvfitr <- cvMTL(datar$X, datar$Y, type="Regression", Regularization="L21", Lam1_seq=10^seq(1,-4, -1),  Lam2=0, opts=list(init=0,  tol=10^-6, maxIter=1500), nfolds=5, stratify=FALSE, parallel=FALSE)

# meta-information and results of CV 
#sequence of lam1
cvfitr$Lam1_seq

#value of lam2
cvfitr$Lam2

#the output lam1 value with minimum CV error
print (paste0("estimated lam1: ", cvfitr$Lam1.min))

#plot CV errors across lam1 sequence in the log space
plot(cvfitr)

## ------------------------------------------------------------------------
datac <- Create_simulated_data(Regularization="L21", type="Classification", n=100)
# CV without parallel computing
start_time <- Sys.time()
cvfitc<-cvMTL(datac$X, datac$Y, type="Classification", Regularization="L21", stratify=TRUE, parallel=FALSE)
Sys.time()-start_time

# CV with parallel computing
start_time <- Sys.time()
cvfitc<-cvMTL(datac$X, datac$Y, type="Classification", Regularization="L21", stratify=TRUE, parallel=TRUE, ncores=2)
Sys.time()-start_time

## ----fig.width=7,fig.height=6, fig.cap="Figure 2: The convergence of objective values across iterations"----

#train a MTL model
model<-MTL(datar$X, datar$Y, type="Regression", Regularization="L21",
  Lam1=cvfitr$Lam1.min, Lam2=0, opts=list(init=0,  tol=10^-6,
  maxIter=1500), Lam1_seq=cvfitr$Lam1_seq)

#demo model
model

# learnt models {W, C}
head(model$W)
head(model$C)

# Historical objective values
str(model$Obj)

# other meta infomration
model$Regularization
model$type
model$dim
str(model$opts)

#plot the historical objective values in the optimization
plotObj(model)

## ------------------------------------------------------------------------
# create simulated data for regression and classification problem
datar <- Create_simulated_data(Regularization="L21", type="Regression")
datac <- Create_simulated_data(Regularization="L21", type="Classification")

# perform CV
cvfitr<-cvMTL(datar$X, datar$Y, type="Regression", Regularization="L21")
cvfitc<-cvMTL(datac$X, datac$Y, type="Classification", Regularization="L21")

# train
modelr<-MTL(datar$X, datar$Y, type="Regression", Regularization="L21",
  Lam1=cvfitr$Lam1.min, Lam1_seq=cvfitr$Lam1_seq)
modelc<-MTL(datac$X, datac$Y, type="Classification", Regularization="L21",
  Lam1=cvfitc$Lam1.min, Lam1_seq=cvfitc$Lam1_seq)

# test  
# for regression problem
calcError(modelr, newX=datar$X, newY=datar$Y) # training error
calcError(modelr, newX=datar$tX, newY=datar$tY) # test error
# for calssification problem
calcError(modelc, newX=datac$X, newY=datac$Y) # training error
calcError(modelc, newX=datac$tX, newY=datac$tY) # test error

# predict
str(predict(modelr, datar$tX)) # for regression
str(predict(modelc, datac$tX)) # for classification

## ---- fig.show = "hold", fig.width=7,fig.height=4, fig.cap="Figure 3: Calculate rough (Left) and precise (Right) by controling the optimization"----
par(mfrow=c(1,2))
model<-MTL(datar$X, datar$Y, type="Regression", Regularization="L21",
  Lam1=cvfitr$Lam1.min, Lam2=0, opts=list(init=0,  tol=10^-2,
  maxIter=10))
plotObj(model)
model<-MTL(datar$X, datar$Y, type="Regression", Regularization="L21",
  Lam1=cvfitr$Lam1.min, Lam2=0, opts=list(init=0,  tol=10^-8,
  maxIter=100))
plotObj(model)

## ---- fig.width=7,fig.height=6, fig.show = "hold", fig.cap="Figure 4: The solution path using warm-start"----
Lam1_seq=10^seq(1,-4, -0.1)
opts=list(init=0,  tol=10^-6,maxIter=100)

mat=vector()
for (i in Lam1_seq){
  m=MTL(datar$X, datar$Y, type="Regression", Regularization="L21", Lam1=i,opts=opts)
  opts$W0=m$W
  opts$C0=m$C
  opts$init=1
  mat=rbind(mat, sqrt(rowSums(m$W^2)))
}
matplot(mat, type="l", xlab="Lambda1", ylab="Significance of Each Predictor")

## ------------------------------------------------------------------------
#warm-start
model1<-MTL(datar$X, datar$Y, type="Regression", Regularization="L21", Lam1=0.01, Lam1_seq=10^seq(0,-4, -0.1))
str(model1$W)

#cold-start
model2<-MTL(datar$X, datar$Y, type="Regression", Regularization="L21", Lam1=0.01)
str(model2$W)

## ------------------------------------------------------------------------
data <- Create_simulated_data(Regularization="Trace", type="Regression", p=20, n=10, t=100)
names(data)
# number of tasks
length(data$X)
# number of subjects and predictors
dim(data$X[[1]])

## ---- fig.show = "hold", fig.width=7, fig.height=4, fig.cap="Figure 5: Comparision of $\\hat{W}$ and $W$"----
#create data
data <- Create_simulated_data(Regularization="Lasso", type="Regression")
#CV
cvfit<-cvMTL(data$X, data$Y, type="Regression", Regularization="Lasso")
cvfit$Lam1.min
#Train
m=MTL(data$X, data$Y, type="Regression", Regularization="Lasso", Lam1=cvfit$Lam1.min, Lam1_seq=cvfit$Lam1_seq)
#Test
paste0("test error: ", calcError(m, data$tX, data$tY))

#Show models 
par(mfrow=c(1,2))
image(t(m$W), xlab="Task Space", ylab="Predictor Space")
title("The Learnt Model")
image(t(data$W), xlab="Task Space", ylab="Predictor Space")
title("The Ground Truth")

## ---- fig.show = "hold", fig.width=7, fig.height=4, fig.cap="Figure 6: Comparision of $\\hat{W}$ and $W$"----
#create datasets
data <- Create_simulated_data(Regularization="L21", type="Regression")
#CV
cvfit<-cvMTL(data$X, data$Y, type="Regression", Regularization="L21")
#Train
m=MTL(data$X, data$Y, type="Regression", Regularization="L21", Lam1=cvfit$Lam1.min, Lam1_seq=cvfit$Lam1_seq)
#Test
paste0("test error: ", calcError(m, data$tX, data$tY))
#Show models
par(mfrow=c(1,2))
image(t(m$W), xlab="Task Space", ylab="Predictor Space")
title("The Learnt Model")
image(t(data$W), xlab="Task Space", ylab="Predictor Space")
title("The Ground Truth")

## ---- fig.show = "hold", fig.width=7, fig.height=4, fig.cap="Figure 7: Comparision of the learnt task relatedness and ground truth"----
#create data
data <- Create_simulated_data(Regularization="Trace", type="Classification")
#CV
cvfit<-cvMTL(data$X, data$Y, type="Classification", Regularization="Trace")
#Train
m=MTL(data$X, data$Y, type="Classification", Regularization="Trace", Lam1=cvfit$Lam1.min, Lam1_seq=cvfit$Lam1_seq)
#Test
paste0("test error: ", calcError(m, data$tX, data$tY))
#Show task relatedness
par(mfrow=c(1,2))
image(cor(m$W), xlab="Task Space", ylab="Task Space")
title("The Learnt Model")
image(cor(data$W), xlab="Task Space", ylab="Task Space")
title("The Ground Truth")

## ---- fig.show = "hold", fig.width=7, fig.height=4, fig.cap="Figure 8: Compare the Learnt Task Relatedness with the Ground Truth"----
#create datasets
data <- Create_simulated_data(Regularization="Graph", type="Classification")
#CV
cvfit<-cvMTL(data$X, data$Y, type="Classification", Regularization="Graph", G=data$G)
#Train
m=MTL(data$X, data$Y, type="Classification", Regularization="Graph", Lam1=cvfit$Lam1.min, Lam1_seq=cvfit$Lam1_seq, G=data$G)
#Test 
print(paste0("the test error is: ", calcError(m, newX=data$tX, newY=data$tY)))
#Show task relatedness
par(mfrow=c(1,2))
image(cor(m$W), xlab="Task Space", ylab="Task Space")
title("The Learnt Model")
image(cor(data$W), xlab="Task Space", ylab="Task Space")
title("The Ground Truth")

## ---- fig.show = "hold", fig.width=7, fig.height=4, fig.cap="Figure 9: Compare the Learnt Task Relatedness with the Ground Truth"----
#Create datasets
data <- Create_simulated_data(Regularization="CMTL", type="Regression")
cvfit<-cvMTL(data$X, data$Y, type="Regression", Regularization="CMTL", k=data$k)
m=MTL(data$X, data$Y, type="Regression", Regularization="CMTL", Lam1=cvfit$Lam1.min, Lam1_seq=cvfit$Lam1_seq, k=data$k)
#Test
paste0("the test error is: ", calcError(m, newX=data$tX, newY=data$tY))
#Show task relatedness
par(mfrow=c(1,2))
image(cor(m$W), xlab="Task Space", ylab="Task Space")
title("The Learnt Model")
image(cor(data$W), xlab="Task Space", ylab="Task Space")
title("Ground Truth")

