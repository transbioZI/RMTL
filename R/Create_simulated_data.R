#' Create an example dataset for testing the MTL algorithm
#'
#' Create an example dataset which contains 1), training datasets (X: feature matrices, Y: response vectors); 2), test datasets 
#' (tX: feature matrices, tY: response vectors); 3), the ground truth model (W: coefficient matrix) and 4), extra
#' information for some algorithms (i.e. a matrix for encoding the network information is necessary for calling the MTL method with network 
#' structure(\code{Regularization=Graph} )
#'
#' @param t Number of tasks
#' @param p Number of features 
#' @param n Number of samples of each task. For simplicity, all tasks
#'     contain the same number of samples.
#' @param type The type of problem, must be "Regression" or
#'     "Classification"
#' @param Regularization The type of MTL algorithm (cross-task regularizer). The value must be
#'     one of \{\code{L21}, \code{Lasso}, \code{Trace}, \code{Graph}, \code{CMTL} \} 
#' @return The example dataset. 
#' 
#' @examples
#' data<-Create_simulated_data(t=5,p=50, n=20, type="Regression", Regularization="L21")
#' str(data)
#' @export
Create_simulated_data <- function(t=5,p=50, n=20, type="Regression",
                                  Regularization="L21"){
W <- matrix(data=stats::rnorm(t*p),ncol=t, nrow = p)

if(Regularization=="Lasso"){
    mask <- matrix(data=stats::rnorm(t*p),ncol=t, nrow = p)
    mask[mask<0] <- 0
    mask[mask>0] <- 1
    W <- W*mask
} else if (Regularization=="L21"){
    W[1:p*0.9,] <- 0
} else if (Regularization=="Trace"){
    requireNamespace('corpcor')
    eigen <- corpcor::fast.svd(W)
    d <- eigen$d
    d[3:length(d)] <- 0
    W <- eigen$u %*% diag(d) %*% t(eigen$v)
} else if (Regularization=="Graph"){
    t1 <- as.integer(t/2)
    w <- matrix(data=stats::rnorm(2*p),ncol=2, nrow = p)
    W <- matrix(data=stats::rnorm(t*p)*0.5,ncol=t, nrow = p)
    W[,1:t1] <- W[,1:t1]+matrix(rep(w[,1],t1), ncol=t1) 
    W[,(t1+1):t] <- W[,(t1+1):t]+matrix(rep(w[,2],t-t1), ncol=t-t1)

    G <- matrix(0, ncol=t, nrow=t)
    G[1:t1, 1:t1] <- -1/t1
    G[(t1+1):t, (t1+1):t] <- -1/t1
    diag(G) <- (t1-1)/t1
} else if (Regularization=="CMTL"){
    t1 <- as.integer(t/2)
    w <- matrix(data=stats::rnorm(2*p),ncol=2, nrow = p)
    W <- matrix(data=stats::rnorm(t*p)*0.5,ncol=t, nrow = p)
    W[,1:t1] <- W[,1:t1]+matrix(rep(w[,1],t1), ncol=t1) 
    W[,(t1+1):t] <- W[,(t1+1):t]+matrix(rep(w[,2],t-t1), ncol=t-t1)
    k <- 2
} 

X <- list(); Y <- list(); tX <- list(); tY <- list()
for(i in 1:t){
    X[[i]] <- matrix(data=stats::rnorm(n*p),ncol=p, nrow = n)
    tX[[i]] <- matrix(stats::rnorm(p*n),nrow=n)
    if (type=="Classification"){
        Y[[i]] <- sign(X[[i]] %*% W[,i] + 0.5 * stats::rnorm(n))
        tY[[i]] <- sign(tX[[i]] %*% W[, i] + stats::rnorm(n) * 0.5)
    } else if(type=="Regression"){
        Y[[i]] <- X[[i]] %*% W[,i] + 0.5 * stats::rnorm(n)
        tY[[i]] <- tX[[i]] %*% W[, i] + stats::rnorm(n) * 0.5
    }
}

if(Regularization=="Lasso" | Regularization=="L21" |
   Regularization=="Trace"){
    data <- list(X, Y, tX, tY, W)
    names(data) <- c("X", "Y", "tX", "tY", "W")
    return(data)
} else if (Regularization=="Graph"){
    data <- list(X, Y, tX, tY, W, G)
    names(data) <- c("X", "Y", "tX", "tY", "W", "G")
    return(data)
} else if (Regularization=="CMTL"){
    data <- list(X, Y, tX, tY, W, k)
    names(data) <- c("X", "Y", "tX", "tY", "W", "k")
    return(data)
} 
    
}
