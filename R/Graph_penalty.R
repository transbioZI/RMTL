#################################
#least-square solver for regression
#################################
LS_Graph <- function (X, Y, G, lam1, lam2, opts){
#------------------------------------------------------
# private functions
gradVal_eval <- function (W, C){
    r <- lapply(c(1:task_num), function(x)
            LS_grad_eval(W[, x], C[x], X[[x]], Y[[x]]))
    grad_W <- sapply(r, function(x)x[[1]]) + 2*lam1*W %*% GGt + 2* lam2*W
    grad_C <- sapply(r, function(x)x[[2]])
    funcVal <- sum(sapply(r, function(x)x[[3]])) +
        lam1*norm(W%*%G,'f')^2 + lam2*norm(W,'f')^2
    
    return(list(grad_W, grad_C, funcVal))
}

funVal_eval <- function (W, C){
return(sum(sapply(c(1:task_num),
    function(x) LS_funcVal_eval(W[, x], C[x], X[[x]], Y[[x]])))+
                lam1*norm(W%*%G,'f')^2 + lam2*norm(W,'f')^2)
}
#-------------------------------------------------------    

# Main algorithm
task_num <- length (X);
dimension = dim(X[[1]])[2];
Obj <- vector(); 

#precomputation
GGt <- G %*% t(G)
    
#initialize a starting point
if(opts$init==0){
   W0 <- matrix(0, nrow=dimension, ncol=task_num);
   C0 <- rep(0, task_num);
}else if(opts$init==1){
   W0 <- opts$W0
   C0 <- opts$C0
}    

bFlag <- 0; 
Wz <- W0;
Cz <- C0;
Wz_old <- W0;
Cz_old <- C0;

t <- 1;
t_old <- 0;
iter <- 0;
gamma <- 1;
gamma_inc <- 2;

while (iter < opts$maxIter){
    alpha <- (t_old - 1) /t;
    
    Ws <- (1 + alpha) * Wz - alpha * Wz_old;
    Cs <- (1 + alpha) * Cz - alpha * Cz_old;
    
    # compute function value and gradients of the search point
    r <- gradVal_eval(Ws, Cs);
    gWs <- r[[1]]
    gCs <- r[[2]]
    Fs <- r[[3]]


    # the Armijo Goldstein line search scheme
    while (TRUE){
        Wzp <- Ws - gWs/gamma;
        Czp <- Cs - gCs/gamma;
        Fzp <- funVal_eval  (Wzp, Czp);
        
        delta_Wzp <- Wzp - Ws;
        delta_Czp <- Czp - Cs;
        nrm_delta_Wzp <- norm(delta_Wzp, 'f')^2;
        nrm_delta_Czp <- sum(delta_Czp * delta_Czp);
        r_sum <- (nrm_delta_Wzp+nrm_delta_Czp)/2;
        
        
        Fzp_gamma = Fs + sum(delta_Wzp* gWs) + 
            sum(delta_Czp * gCs) + gamma * r_sum
        
        if (r_sum <=1e-20){
            bFlag=1; 
            break;
        }
        
        if (Fzp <= Fzp_gamma) break else {gamma = gamma * gamma_inc}
  
    }
    
    Wz_old = Wz; 
    Cz_old = Cz;
    Wz = Wzp;
    Cz = Czp;
    
    Obj = c(Obj, Fzp );
    
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
return(list(W=W, C=C, Obj=Obj))
}




#################################
#logistic regression solver for classification
#################################
LR_Graph <- function (X, Y, G, lam1, lam2, opts){
#------------------------------------------------------------
# private functions
    
gradVal_eval <- function (W, C){
r <- lapply(c(1:task_num),
    function(x) LR_grad_eval( W[, x], C[x], X[[x]], Y[[x]]))
grad_W <- sapply(r, function(x)x[[1]])+2*lam1*W %*% GGt + 2* lam2 * W
grad_C <- sapply(r, function(x)x[[2]])
funcVal = sum(sapply(r, function(x)x[[3]])) + lam1*norm(W%*%G,'f')^2 +
    lam2 * norm(W,'f')^2
return(list(grad_W, grad_C, funcVal))
}    

funVal_eval <- function (W, C){
return(sum(sapply(c(1:task_num),
    function(x) LR_funcVal_eval(W[, x], C[x], X[[x]], Y[[x]])))+
    lam1*norm(W%*%G,'f')^2) + lam2 * norm(W,'f')^2
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
}else if(opts$init==1){
   W0 <- opts$W0
   C0 <- opts$C0
}    

#precomputation
GGt <- G %*% t(G)
    
bFlag <- 0; 
Wz <- W0;
Cz <- C0;
Wz_old <- W0;
Cz_old <- C0;

t <- 1;
t_old <- 0;
iter <- 0;
gamma <- 1;
gamma_inc <- 2;

while (iter < opts$maxIter){
    alpha <- (t_old - 1) /t;
    
    Ws <- (1 + alpha) * Wz - alpha * Wz_old;
    Cs <- (1 + alpha) * Cz - alpha * Cz_old;
    
    # compute function value and gradients of the search point
    r <- gradVal_eval(Ws, Cs);
    gWs <- r[[1]]
    gCs <- r[[2]]
    Fs <- r[[3]]


    # the Armijo Goldstein line search scheme
    while (TRUE){
        Wzp <- Ws - gWs/gamma;
        Czp <- Cs - gCs/gamma;
        Fzp <- funVal_eval  (Wzp, Czp);
        
        delta_Wzp <- Wzp - Ws;
        delta_Czp <- Czp - Cs;
        nrm_delta_Wzp <- norm(delta_Wzp, 'f')^2;
        nrm_delta_Czp <- sum(delta_Czp * delta_Czp);
        r_sum <- (nrm_delta_Wzp+nrm_delta_Czp)/2;
        
        Fzp_gamma = Fs + sum(delta_Wzp* gWs) + 
            sum(delta_Czp * gCs) + gamma * r_sum
        
        if (r_sum <=1e-20){
            bFlag=1; 
            break;
        }
        
        if (Fzp <= Fzp_gamma) break else {gamma = gamma * gamma_inc}
  
    }
    
    Wz_old = Wz;
    Cz_old = Cz;
    Wz = Wzp;
    Cz = Czp;
    
    Obj = c(Obj, Fzp)
    
    
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
return(list(W=W, C=C, Obj=Obj))
}
