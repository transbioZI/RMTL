#################################
#least-square solver for regression
#################################
LS_CMTL <- function (X, Y, lam1, lam2, k, opts){

#------------------------------------------------------------
# private functions
gradVal_eval <- function (W, C, M){
requireNamespace('MASS')
requireNamespace('psych')
IM = (eta * diag(task_num) + M)
invIM <- MASS::ginv(IM)
invEtaMWt = invIM %*% t(W)

r <- lapply(c(1:task_num), function(x)
            LS_grad_eval(W[, x], C[x], X[[x]], Y[[x]]))
grad_W <- sapply(r, function(x)x[[1]]) + 2 * c * t(invEtaMWt)
grad_C <- sapply(r, function(x)x[[2]])
funcVal = sum(sapply(r, function(x)x[[3]])) + c * psych::tr(W %*% invEtaMWt)
grad_M = - c * (t(W) %*% W %*% invIM %*% invIM )      #M component

return(list(grad_W, grad_C, grad_M, funcVal))
}

funVal_eval <- function (W, C, M_Pz, M_DiagSigz){
requireNamespace('psych')
invIM = M_Pz %*% (diag( 1/(eta + M_DiagSigz))) %*% t(M_Pz)
invEtaMWt = invIM %*% t(W);
return(sum(sapply(c(1:task_num), function(x)
        LS_funcVal_eval(W[, x], C[x], X[[x]], Y[[x]]))) +
        c * psych::tr( W %*% invEtaMWt))
}
    
#------------------------------------------------------------

task_num <- length (X);
dimension = dim(X[[1]])[2];
Obj <- vector(); 

#initialize a starting point
if(opts$init==0){
   W0 <- matrix(0, nrow=dimension, ncol=task_num);
   C0 <- rep(0, task_num);
   M0 <- diag (task_num) * k / task_num;
}else if(opts$init==1){
   W0 <- opts$W0
   C0 <- opts$C0
   M0 <- opts$M0
}    

#precomputation
eta <- lam2 / lam1;
c <- lam1 * eta * (1 + eta);


bFlag <- 0; 
Wz <- W0;
Mz <- M0;
Cz <- C0;
Wz_old <- W0;
Cz_old <- C0;
Mz_old <- M0;

t <- 1;
t_old <- 0;
iter <- 0;
gamma <- 1;
gamma_inc <- 2;

while (iter < opts$maxIter){
    alpha <- (t_old - 1) /t;
    
    Ws <- (1 + alpha) * Wz - alpha * Wz_old;
    Cs <- (1 + alpha) * Cz - alpha * Cz_old;
    Ms <- (1 + alpha) * Mz - alpha * Mz_old;

    # compute function value and gradients of the search point
    r <- gradVal_eval(Ws, Cs, Ms);
    gWs <- r[[1]]
    gCs <- r[[2]]
    gMs <- r[[3]]
    Fs <- r[[4]]
    
    # the Armijo Goldstein line search scheme
    while (TRUE){
        Wzp = Ws - gWs/gamma;
        Czp <- Cs - gCs/gamma;
        r <- singular_projection (Ms - gMs/gamma, k);
        Mzp <- r[[1]]
        Mzp_Pz <- r[[2]]
        Mzp_DiagSigz <- r[[3]]
        Fzp = funVal_eval(Wzp, Czp, Mzp_Pz, Mzp_DiagSigz);
        
        delta_Wzp <- Wzp - Ws;
        delta_Czp <- Czp - Cs;
        delta_Mzp <- Mzp - Ms;
        nrm_delta_Wzp <- norm(delta_Wzp, 'f')^2;
        nrm_delta_Czp <- sum(delta_Czp * delta_Czp);
        nrm_delta_Mzp <- norm(delta_Mzp, 'f')^2;
        r_sum <- (nrm_delta_Wzp+nrm_delta_Czp+nrm_delta_Mzp)/2;
        
        Fzp_gamma = Fs + sum(delta_Wzp* gWs) + 
            sum(delta_Mzp * gMs)+ sum(delta_Czp * gCs)+
            gamma * r_sum;
        
        if (r_sum <=1e-20){
            bFlag=1; 
            break;
        }
        
        if (Fzp <= Fzp_gamma) break else {gamma = gamma * gamma_inc}
  
    }
    
    Mz_old <- Mz;
    Cz_old <- Cz;
    Wz_old = Wz;
    Wz = Wzp;
    Cz <- Czp;
    Mz <- Mzp;
    
    Obj = c(Obj, Fzp);
    
    #test stop condition.
    if (bFlag) break;
    if (iter>=2){
        if (abs( Obj[length(Obj)] - Obj[length(Obj)-1] ) <= opts$tol)
            break;
    }
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);

}

W = Wzp;
C = Czp;
M = Mzp;

return(list(W=W, C=C, Obj=Obj, M=M))
}


#################################
#logistic regression solver for classification
#################################
LR_CMTL <- function (X, Y, lam1, lam2, k,  opts){
#------------------------------------------------------------
#private function


gradVal_eval <- function (W, C, M){
requireNamespace('MASS')
requireNamespace('psych')
IM = (eta * diag(task_num) + M)
invIM <- MASS::ginv(IM)
invEtaMWt = invIM %*% t(W)
r <- lapply(c(1:task_num),
    function(x)LR_grad_eval( W[, x], C[x], X[[x]], Y[[x]]))
grad_W <- sapply(r, function(x)x[[1]]) + 2 * c * t(invEtaMWt)
grad_C <- sapply(r, function(x)x[[2]])
grad_M = - c * (t(W) %*% W %*% invIM %*% invIM )      #M component

funcVal = sum(sapply(r, function(x)x[[3]])) +
    c * psych::tr( W %*% invEtaMWt)
return(list(grad_W, grad_C, grad_M, funcVal))
}    

funVal_eval <- function (W, C, M_Pz, M_DiagSigz){
requireNamespace('psych')
invIM = M_Pz %*% (diag( 1/(eta + M_DiagSigz))) %*% t(M_Pz)
invEtaMWt = invIM %*% t(W);
return(sum(sapply(c(1:task_num),
    function(x)LR_funcVal_eval(W[, x], C[x], X[[x]], Y[[x]]))) +
    c * psych::tr( W %*% invEtaMWt))
}
#------------------------------------------------------------

#main algorithm
task_num <- length (X);
dimension = dim(X[[1]])[2];
subjects <- dim(X[[1]])[1];
Obj <- vector(); 

#initialize a starting point
if(opts$init==0){
   W0 <- matrix(0, nrow=dimension, ncol=task_num);
   C0 <- rep(0, task_num);
   M0 <- diag (task_num) * k / task_num;
}else if(opts$init==1){
   W0 <- opts$W0
   C0 <- opts$C0
   M0 <- opts$M0
}    

#precomputation
eta <- lam2 / lam1;
c <- lam1 * eta * (1 + eta);
    
bFlag <- 0; 
Wz <- W0;
Cz <- C0;
Mz <- M0;
Wz_old <- W0;
Cz_old <- C0;
Mz_old <- M0;
    
t <- 1;
t_old <- 0;
iter <- 0;
gamma <- 1;
gamma_inc <- 2;

while (iter < opts$maxIter){
    alpha <- (t_old - 1) /t;
    
    Ws <- (1 + alpha) * Wz - alpha * Wz_old;
    Cs <- (1 + alpha) * Cz - alpha * Cz_old;
    Ms <- (1 + alpha) * Mz - alpha * Mz_old;
    
    # compute function value and gradients of the search point
    r <- gradVal_eval(Ws, Cs, Ms);
    gWs <- r[[1]]
    gCs <- r[[2]]
    gMs <- r[[3]]
    Fs <- r[[4]]


    # the Armijo Goldstein line search scheme
    while (TRUE){

        Wzp = Ws - gWs/gamma;
        Czp = Cs - gCs/gamma;
        r <- singular_projection (Ms - gMs/gamma, k);
        Mzp <- r[[1]]
        Mzp_Pz <- r[[2]]
        Mzp_DiagSigz <- r[[3]]
        Fzp = funVal_eval(Wzp, Czp, Mzp_Pz, Mzp_DiagSigz);
        
        delta_Wzp <- Wzp - Ws;
        delta_Czp <- Czp - Cs;
        delta_Mzp <- Mzp - Ms;
        
        nrm_delta_Wzp <- norm(delta_Wzp, 'f')^2;
        nrm_delta_Czp <- sum(delta_Czp * delta_Czp);
        nrm_delta_Mzp <- norm(delta_Mzp, 'f')^2;
        
        r_sum <- (nrm_delta_Wzp+nrm_delta_Czp+nrm_delta_Mzp)/2;
        
        Fzp_gamma = Fs + sum(delta_Wzp* gWs) + 
            sum(delta_Czp * gCs) + sum(delta_Mzp * gMs)+
            gamma * r_sum
                    
        if (r_sum <=1e-20){
            bFlag=1; 
            break;
        }
        
        if (Fzp <= Fzp_gamma) break else {gamma = gamma * gamma_inc}
  
    }
    
    Wz_old <- Wz;
    Cz_old <- Cz;
    Mz_old <- Mz;
    Wz <- Wzp;
    Cz <- Czp;
    Mz <- Mzp;
    
    
    Obj = c(Obj, Fzp);
    
    
    #test stop condition.
    if (bFlag) break;
    if (iter>=2){
        if (abs( Obj[length(Obj)] - Obj[length(Obj)-1] ) <= opts$tol)
            break;
    }
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);

}

W = Wzp;
C = Czp;
M = Mzp;

return(list(W=W, C=C, Obj=Obj, M=M))
}
