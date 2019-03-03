# RMTL
Regularized Multi-task Learning in R

# Description 
This package provides an efficient implementation of regularized multi-task learning comprising 10 algorithms applicable for regression, classification, joint feature selection, task clustering, low-rank learning, sparse learning and network incorporation. All algorithms are implemented basd on the accelerated gradient descent method and feature a complexity of O(1/k^2). Sparse model structure is induced by the solving the proximal operator.

# Required Packages
Four packages have to be instaled in advanced to enable functions i.e. eigen-decomposition, 2D plotting: ‘MASS’, ‘psych’, ‘corpcor’ and ‘fields’. You can install them from the CRAN.
```R
install.packages("MASS")
install.packages("psych")
install.packages("corpcor")
install.packages("fields")
```

# Installation

1) Install directly in R (Recommend)
```R
install.packages("devtools")
library("devtools")
install_github("transbioZI/RMTL")
```

2) Install in shell
```shell
git clone https://github.com/transbioZI/RMTL.git
R CMD build ./RMTL/
R CMD INSTALL RMTL*.tar.gz
```

# Details
Please check ["RMTL-manuel.pdf"](https://github.com/transbioZI/RMTL/blob/master/RMTL-manual.pdf) for more details.

# Reference
[Cao, Han, Jiayu Zhou and Emanuel Schwarz. "RMTL: An R Library for Multi-Task Learning." Bioinformatics (2018).](https://doi.org/10.1093/bioinformatics/bty831)


# Contact
If you have any question, please contact: hank9cao@gmail.com
