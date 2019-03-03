
#########################
#gradients and functions evaluation
#########################
#logistic regression
LR_grad_eval <- function( w, c, x, y){
    weight <- 1/length(y)
    l <- -y*(x %*% w + c)
    lp <- l
    lp[lp<0] <- 0
    funcVal <- sum(weight * ( log( exp(-lp) +  exp(l-lp) ) + lp ))
    b <- (-weight*y)*(1 - 1/ (1+exp(l)))
    grad_c <- sum(b)
    grad_w <- t(x) %*% b
    return(list(grad_w, grad_c, funcVal))
}
LR_funcVal_eval <- function ( w, c, x, y){
    weight <- 1/length(y)
    l <- -y*(x %*% w + c)
    lp <- l
    lp[lp<0] <- 0
    return(sum(weight * ( log( exp(-lp) +  exp(l-lp) ) + lp )))
}
#Least square
LS_grad_eval <- function( w, c, x, y, xy, xx){
    grad_w <-  t(x) %*% (x %*% w + c - y) / nrow(x)
    grad_c <-  mean(x %*% w + c -y)
    funcVal <- 0.5 * mean((y - x %*% w -c)^2)
    return(list(grad_w, grad_c, funcVal))
}

LS_funcVal_eval <- function ( w, c, x, y){
    return(0.5 * mean((y - x %*% w -c)^2))
}







############################
#Different Projections
############################
l1_projection <- function (W, lambda ){
p <- abs(W) - lambda
p[p<0] <- 0
Wp <- sign(W) * p
return(Wp)
}

L21_projection <- function (W, lambda ){
thresfold <- sqrt(rowSums(W^2))
zeros <- which(thresfold==0)              
temp <- 1 - lambda/thresfold
temp <- ifelse(temp<0, 0, temp)
Wp = matrix(rep(temp, ncol(W)), nrow=length(temp))*W
Wp[zeros,] <- 0
return(Wp)
}

trace_projection <- function (W, lambda ){
requireNamespace('corpcor')
eigen <- corpcor::fast.svd(W)
d <- eigen$d
thresholded_value = d - lambda / 2;
dp <- thresholded_value * ( thresholded_value > 0 )
if (length(dp)>1){
    Wp <- eigen$u %*% diag(dp) %*% t(eigen$v)
}else{
    Wp <- (eigen$u * dp) %*% t(eigen$v)
}
return(list(Wp, sum(dp)))
}

singular_projection <- function(Msp, k){
requireNamespace('MASS')
eig <- eigen(Msp, symmetric=TRUE)
EVector <- eig$vector
EValue  <- eig$values
DiagSigz <- bsa_ihb(EValue, rep(1,length(EValue)),
                    k, rep(1,length(EValue)))
DiagSigz <- DiagSigz[[1]]
Mzp <- EVector %*% diag(DiagSigz) %*% t(EVector)
Mzp_Pz <- EVector
Mzp_DiagSigz <- DiagSigz
return(list(Mzp, Mzp_Pz, Mzp_DiagSigz))
}

bsa_ihb <- function(a,b,r,u){
# initilization
break_flag <- 0;
t_l <- a/b; t_u <- (a - u)/b;
T <- c(t_l, t_u)
t_L <- -Inf; t_U <- Inf;
g_tL <- 0; g_tU <- 0;

iter = 0
while (length(T)!=0){
    iter <- iter + 1
    g_t <- 0
    t_hat <- stats::median(T)
    
    U <- t_hat < t_u
    M <- (t_u <= t_hat) & (t_hat <= t_l)

    if (sum(U)){
       g_t <- g_t + t(b[U]) %*% u[U] 
    }

    if (sum(M)){
        g_t <- g_t + sum(b[M] %*% (a[M] - t_hat * b[M]))
    }
    
    if (g_t > r){
        t_L <- t_hat
        T <- T[T > t_hat]
        g_tL = g_t
    } else if (g_t < r){
        t_U <- t_hat
        T <- T[T < t_hat]
        g_tU <- g_t
    } else{
        t_star <- t_hat
        break_flag <- 1
        break
    }
}
if (!break_flag){
     t_star <- t_L - (g_tL -r) * (t_U - t_L)/(g_tU - g_tL)     
}
temp <- sapply(a - rep(t_star, length(b))*b, function(x)max(0,x))
  x_star <- sapply(temp, function(x) min(x,u))
return(list(x_star,t_star,iter))
}
