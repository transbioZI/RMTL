#' Train a multi-task learning model. 
#'
#' Train a multi-task learning model. 
#' 
#' @param X A set of feature matrices
#' @param Y A set of responses, could be binary (classification
#'     problem) or continues (regression problem). The valid
#'     value of binary outcome \eqn{\in\{1, -1\}}
#' @param type The type of problem, must be \code{Regression} or
#'     \code{Classification}
#' @param Regularization The type of MTL algorithm (cross-task regularizer). The value must be
#'     one of \{\code{L21}, \code{Lasso}, \code{Trace}, \code{Graph}, \code{CMTL} \} 
#' @param Lam1 A positive constant \eqn{\lambda_{1}} to control the
#'     cross-task regularization
#' @param Lam2 A non-negative constant \eqn{\lambda_{2}} to improve the
#'     generalization performance with the default value of 0 (except for
#'     \code{Regularization=CMTL})
#' @param Lam1_seq A positive sequence of \code{Lam1}. If the parameter
#'     is given, the model is trained using warm-start technique. Otherwise, the
#'     model is trained based on the \code{Lam1} and the initial search point (\code{opts$init}). 
#' @param opts Options of the optimization procedure. One can set the
#'     initial search point, the tolerance and the maximized number of
#'     iterations using this parameter. The default value is
#'     \code{list(init=0,  tol=10^-3, maxIter=1000)} 
#' @param G A matrix to encode the network information. This parameter
#'     is only used in the MTL with graph structure (\code{Regularization=Graph} )
#' @param k A positive number to modulate the structure of clusters
#'     with the default of 2. This parameter is only used in MTL with
#'     clustering structure (\code{Regularization=CMTL} ) Note, the larger number is adapted to more
#'     complex clustering structure.
#' @return The trained model including the coefficient matrix \code{W}
#'     and intercepts \code{C} and related meta information
#' 
#' @examples
#' #create the example data
#' data<-Create_simulated_data(Regularization="L21", type="Regression")
#' #train a MTL model
#' #cold-start
#' model<-MTL(data$X, data$Y, type="Regression", Regularization="L21",
#'     Lam1=0.1, Lam2=0, opts=list(init=0,  tol=10^-6, maxIter=1500))
#' #warm-start
#' model<-MTL(data$X, data$Y, type="Regression", Regularization="L21",
#'     Lam1=0.1, Lam1_seq=10^seq(1,-4, -1), Lam2=0, opts=list(init=0,  tol=10^-6, maxIter=1500))
#' #meta-information
#' str(model)
#' #plot the historical objective values
#' plotObj(model)
#' @export
MTL <- function(X, Y, type="Classification", Regularization="L21",
                Lam1=0.1, Lam1_seq=NULL, Lam2=0,
                opts=list(init=0,  tol=10^-3,
                maxIter=1000), G=NULL, k=2)
{

    #test vilidity of input data
    if (!missing(X) & !missing(Y)){
        if (all(sapply(X, class)!="matrix")){
            X <- lapply(X, function(x){as.matrix(x)})
        }
        if (all(sapply(Y, class)!="matrix")){
            Y <- lapply(Y, function(x){as.matrix(x)})
        }
    }else{
        stop("data X or Y doesnot exists")
    }
    
    #test the validity of problem type
    if(type=="Classification"){
        method <- "LR"
    }else if(type=="Regression"){
        method <- "LS"
    }else{
        stop("neither Regression or Classification")
    }

    #test the validity of regularization 
    allRegularizations <- c("L21", "Lasso", "Graph", "CMTL", "Trace")
    if (is.element(Regularization, allRegularizations)){
        method <- paste0(method, "_", Regularization)
    }else{
    stop("Regularization is not recognizable")}

    #test validity of Lam1 and Lam2
    if (Lam1<0) {stop("Lam1 must be positive")}
    if (Lam2<0) {stop("Lam2 must be positive")}
    
    #collect arguments 
    args <- list(X=X, Y=Y, lam1=Lam1, lam2=Lam2, opts=opts)
    
    if (Regularization=="CMTL"){
        if (k>0){
            args$k <- k
        }else(stop("for CMTL, k must be positive interger"))
    }
    if (Regularization=="Graph"){
        if(!is.null(G)){
            args$G <- G
        }else{stop("graph matrix G is not provided")}
    }

    #call solver
    if (any(Regularization==c("L21", "Lasso", "Trace"))){
        #sparse routine
        if( !is.null(Lam1_seq) & length(Lam1_seq)>0){
            #with warm start
            opt <- opts
            for (x in Lam1_seq){
                args$lam1 <- x
                m <- do.call(method, args)
                opt$init <- 1
                opt$W0 <- m$W
                opt$C0 <- m$C
                args$opts <- opt
                if (x<=Lam1) break
            }
        } else {
            #without warm start
            m <- do.call(method, args)
        }
    } else if(any(Regularization==c("Graph", "CMTL"))){
        m <- do.call(method, args)
    }
    
    m$call <- match.call()
    m$Lam1 <- args$lam1
    m$Lam2 <- args$Lam2
    m$opts <- args$opts
    m$dim <- sapply(X, function(x)dim(x))
    m$type=type
    m$Regularization=Regularization
    m$method=method
    class(m) <- "MTL"
    return(m)
}


#' Predict the outcomes of new individuals
#'
#' Predict the outcomes of new individuals. For classification, the
#'     probability of the individual being assigned to positive label P(y==1) is estimated, and for regression, the
#'     prediction score is estimated
#'
#' @param object A trained MTL model
#' @param newX The feature matrices of new individuals
#' @param ... Other parameters
#' @return The predictive outcome
#' @examples
#' #Create data
#' data<-Create_simulated_data(Regularization="L21", type="Regression")
#' #Train
#' model<-MTL(data$X, data$Y, type="Regression", Regularization="L21",
#'     Lam1=0.1, Lam2=0, opts=list(init=0,  tol=10^-6, maxIter=1500)) 
#  #Predict
#' predict(model, newX=data$tX)
#'
#' @export
predict.MTL <- function(object, newX=NULL, ...){
    if(!is.null(newX)){
        task_num <- length(newX)
        score <- lapply(c(1:task_num), function(x)
            newX[[x]] %*% object$W[,x] + object$C[x])
        if (object$type=="Classification"){
            y <- lapply(c(1:task_num),
                        function(x) exp(score[[x]]))
            y <- lapply(y, function(x) x/(1+x))
        }else if (object$type=="Regression"){
            y <- score
        }
        return(y)
    }else{stop("no new data (X) is provided")}
}



#' Calculate the prediction error
#'
#' Calculate the averaged prediction error across tasks. For
#' classification problem, the miss-classification rate is returned, and for
#' regression problem, the mean square error(MSE) is returned. 
#'
#' @param m A MTL model
#' @param newX The feature matrices of new individuals
#' @param newY The responses of new individuals
#' @return The averaged prediction error
#' @examples
#' #create example data
#' data<-Create_simulated_data(Regularization="L21", type="Regression")
#' #train a model 
#' model<-MTL(data$X, data$Y, type="Regression", Regularization="L21",
#'     Lam1=0.1, Lam2=0, opts=list(init=0,  tol=10^-6, maxIter=1500)) 
#' #calculate the training error
#' calcError(model, newX=data$X, newY=data$Y)
#' #calculate the test error
#' calcError(model, newX=data$tX, newY=data$tY)
#'
#' @export
calcError <- function(m, newX=NULL, newY=NULL){
    if(class(m)!="MTL"){
        stop("The first arguement is not a MTL model")}
    if(!is.null(newX) & !is.null(newY)){
        task_num <- length(newY)
        yhat <- predict.MTL(m,newX)
        if(m$type=="Classification"){
            residue <- lapply(1:task_num, function(x)
                newY[[x]]-(round(yhat[[x]])-0.5)*2)
            error <- sapply(residue, function(x){sum(x!=0)/length(x)})
        }else if(m$type=="Regression"){
            error <- sapply(1:task_num, function(x)
                mean((newY[[x]]-yhat[[x]])^2))
        }
        return(mean(error))
    }else{stop(" no new data (X or Y) are provided ")}
}



#' Plot the historical values of objective function 
#'
#' Plot the values of objective function across iterations in the
#' optimization procedure. This function indicates the "inner status" of the
#' solver during the optimization, and could be used for diagnosis of the
#' solver and training procedure. 
#'
#' @param m A trained MTL model
#' @examples
#' #create the example date
#' data<-Create_simulated_data(Regularization="L21", type="Regression")
#' #Train a MTL model
#' model<-MTL(data$X, data$Y, type="Regression", Regularization="L21",
#'     Lam1=0.1, Lam2=0, opts=list(init=0,  tol=10^-6, maxIter=1500))
#' #plot the objective values
#' plotObj(model)
#' @importFrom graphics plot
#'
#' @export
plotObj <- function(m){
    if(class(m)!="MTL"){
        stop("The first arguement is not a MTL model")}
    graphics::plot(m$Obj, xlab="iterations", ylab="objective value")
}


#' Print the  meta information of the model
#'
#' @export
#' @param x A trained MTL model
#' @param ... Other parameters
#' @importFrom utils head
#' @examples
#' #create data
#' data<-Create_simulated_data(Regularization="L21", type="Regression")
#' #train a MTL model
#' model<-MTL(data$X, data$Y, type="Regression", Regularization="L21",
#'     Lam1=0.1, Lam2=0, opts=list(init=0,  tol=10^-6, maxIter=1500)) 
#' #print the information of the model
#' print(model)
print.MTL <- function(x, ...)
{
    cat("\nHead Coefficients:\n")
    print(utils::head(x$W))
    cat("Call:\n")
    print(x$call)
    cat("type:\n")
    print(x$type)

    formulas <- list(Lasso="||W||_1",
                     L21="||W||_{2,1}",
                     Trace="||W||_*",
                     Graph="||WG||{_F}{^2}",
                     CMTL="tr(W^TW)-tr(F^TW^TWF)")
    cat("Formulation:\n")
    print(paste0('SUM_i Loss_i(W) + Lam1*',
                        formulas[[x$Regularization]],
                        ' + Lam2*||W||{_2}{^2}'))
}



#' K-fold cross-validation
#'
#' Perform the k-fold cross-validation to estimate the \eqn{\lambda_1}.
#'
#' @param X A set of feature matrices
#' @param Y A set of responses, could be binary (classification
#'     problem) or continues (regression problem). The valid
#'     value of binary outcome \eqn{\in\{1, -1\}}
#' @param type The type of problem, must be \code{Regression} or
#'     \code{Classification}
#' @param Regularization The type of MTL algorithm (cross-task regularizer). The value must be
#'     one of \{\code{L21}, \code{Lasso}, \code{Trace}, \code{Graph}, \code{CMTL} \} 
#' @param Lam2 A positive constant \eqn{\lambda_{2}} to improve the
#'     generalization performance
#' @param Lam1_seq A positive sequence of \code{Lam1} which controls the
#'     cross-task regularization
#' @param opts Options of the optimization procedure. One can set the
#'     initial search point, the tolerance and the maximized number of
#'     iterations through the parameter. The default value is
#'     \code{list(init=0,  tol=10^-3, maxIter=1000)} 
#' @param G A matrix to encode the network information. This parameter
#'     is only used in the MTL with graph structure (\code{Regularization=Graph} )
#' @param k A positive number to modulate the structure of clusters
#'     with the default of 2. This parameter is only used in MTL with
#'     clustering structure (\code{Regularization=CMTL} ) Note, the larger number is adapted to more
#'     complex clustering structure.
#' @param stratify \code{stratify=TRUE} is used for stratified
#'     cross-validation
#' @param nfolds The number of folds
#' @param parallel \code{parallel=TRUE} is used for parallel computing
#' @param ncores The number of cores used for parallel computing with the default value of 2
#' 
#' @return The estimated \eqn{\lambda_1} and related information
#' 
#' @examples
#' #create the example data
#' data<-Create_simulated_data(Regularization="L21", type="Classification")
#' #perform the cross validation
#' cvfit<-cvMTL(data$X, data$Y, type="Classification", Regularization="L21", 
#'     Lam2=0, opts=list(init=0,  tol=10^-6, maxIter=1500), nfolds=5,
#'     stratify=TRUE, Lam1_seq=10^seq(1,-4, -1))
#' #show meta-infomration
#' str(cvfit)
#' #plot the CV accuracies across lam1 sequence
#' plot(cvfit)
#' @importFrom foreach %dopar%
#' @importFrom foreach foreach
#' @importFrom doParallel registerDoParallel
#' @export
cvMTL <- function(X, Y, type="Classification", Regularization="L21",
                   Lam1_seq=10^seq(1,-4, -1), Lam2=0, G=NULL, k=2,
                   opts=list(init=0, tol=10^-3, maxIter=1000),
                   stratify=FALSE, nfolds=5, ncores=2, parallel=FALSE){
    #test vilidity of input data
    if (!missing(X) & !missing(Y)){
        if (all(sapply(X, class)!="matrix")){
            X <- lapply(X, function(x){as.matrix(x)})
        }
        if (all(sapply(Y, class)!="matrix")){
            Y <- lapply(Y, function(x){as.matrix(x)})
        }
    }else{
        stop("data X or Y doesnot exists")
    }
    task_num <- length(X)
    if(stratify & type=="Regression"){
        stop("stratified CV is not applicable to regression")}
    cvPar <- getCVPartition(Y, nfolds, stratify)

#cv
if (!parallel){
cvm <- rep(0, length(Lam1_seq))
for (i in 1:nfolds){
    cv_Xtr <- lapply(c(1:task_num),
                     function(x) X[[x]][cvPar[[i]][[1]][[x]], ])
    cv_Ytr <- lapply(c(1:task_num),
                     function(x) Y[[x]][cvPar[[i]][[1]][[x]]])
    cv_Xte <- lapply(c(1:task_num),
                     function(x) X[[x]][cvPar[[i]][[2]][[x]], ])
    cv_Yte <- lapply(c(1:task_num),
                     function(x) Y[[x]][cvPar[[i]][[2]][[x]]])
    opt <- opts
    for (p_idx in 1: length(Lam1_seq)){
        m <- MTL(X=cv_Xtr, Y=cv_Ytr, type=type,
                 Regularization=Regularization, Lam1=Lam1_seq[p_idx],
                 Lam2=Lam2, opts=opt, k=k, G=G)
        #non sparse model training
        if (!is.element(Regularization, c("Graph", "CMTL"))){
            opt$init <- 1
            opt$W0 <- m$W
            opt$C0 <- m$C
        }
        cv_err <- calcError(m, newX=cv_Xte, newY=cv_Yte)
        cvm[p_idx] = cvm[p_idx]+cv_err
    }
}
cvm = cvm/nfolds
} else {
requireNamespace('doParallel')
requireNamespace('foreach')
doParallel::registerDoParallel(ncores)
cvm <- foreach::foreach(i = 1:nfolds, .combine="cbind") %dopar%{
    cv_Xtr <- lapply(c(1:task_num),
                     function(x) X[[x]][cvPar[[i]][[1]][[x]], ])
    cv_Ytr <- lapply(c(1:task_num),
                     function(x) Y[[x]][cvPar[[i]][[1]][[x]]])
    cv_Xte <- lapply(c(1:task_num),
                     function(x) X[[x]][cvPar[[i]][[2]][[x]], ])
    cv_Yte <- lapply(c(1:task_num),
                     function(x) Y[[x]][cvPar[[i]][[2]][[x]]])
    opt <- opts
    cvVec=rep(0, length(Lam1_seq))
    for (p_idx in 1: length(Lam1_seq)){
        m <- MTL(X=cv_Xtr, Y=cv_Ytr, type=type,
                 Regularization=Regularization, Lam1=Lam1_seq[p_idx],
                 Lam2=Lam2, opts=opt, k=k, G=G)
        #non sparse model training
        if (!is.element(Regularization, c("Graph", "CMTL")) ){
            opt$init <- 1
            opt$W0 <- m$W
            opt$C0 <- m$C
        }
        cv_err <- calcError(m, newX=cv_Xte, newY=cv_Yte)
        cvVec[p_idx] <- cv_err
    }
    return(cvVec)
}
cvm <- rowMeans(cvm)
}

best_idx <- which(cvm==min(cvm))[1]
cv <- list(Lam1_seq=Lam1_seq, Lam1.min=Lam1_seq[best_idx],
           Lam2=Lam2, cvm=cvm)
class(cv) <- "cvMTL"
return(cv)
}

#' Plot the cross-validation curve
#'
#' @export
#' @param x The returned object of function \code{cvMTL}
#' @param ... Other parameters
#' @examples
#' #create the example data
#' data<-Create_simulated_data(Regularization="L21", type="Classification")
#' #perform the cv
#' cvfit<-cvMTL(data$X, data$Y, type="Classification", Regularization="L21", 
#'     Lam2=0, opts=list(init=0,  tol=10^-6, maxIter=1500), nfolds=5,
#'     stratify=TRUE, Lam1_seq=10^seq(1,-4, -1))
#' #plot the curve
#' plot(cvfit)
#' @importFrom graphics plot
#' @export
plot.cvMTL <- function(x, ...){
    graphics::plot(log10(x$Lam1_seq), x$cvm, xlab="log10(Lambda1)",
         ylab="error")
}


getCVPartition <- function(Y, cv_fold, stratify){
task_num = length(Y);

randIdx <- lapply(Y, function(x) sample(1:length(x),
           length(x), replace = FALSE))        
cvPar = {};
for (cv_idx in 1: cv_fold){
    # buid cross validation data splittings for each task.
    cvTrain = {};
    cvTest = {};


        
    #stratified cross validation
    for (t in 1: task_num){
        task_sample_size <- length(Y[[t]]);

        if (stratify){
            ct <- which(Y[[t]][randIdx[[t]]]==-1);
            cs <- which(Y[[t]][randIdx[[t]]]==1);
            ct_idx <- seq(cv_idx, length(ct), cv_fold);
            cs_idx <- seq(cv_idx, length(cs), cv_fold);
            te_idx <- c(ct[ct_idx], cs[cs_idx]);
            tr_idx <- seq(1,task_sample_size)[
                !is.element(1:task_sample_size, te_idx)];

        } else {
            te_idx <- seq(cv_idx, task_sample_size, by=cv_fold)
            tr_idx <- seq(1,task_sample_size)[
                !is.element(1:task_sample_size, te_idx)];
        }

        cvTrain[[t]] = randIdx[[t]][tr_idx]
        cvTest[[t]] = randIdx[[t]][te_idx]
   }
    
    cvPar[[cv_idx]]=list(cvTrain, cvTest);
}
return(cvPar)
}

