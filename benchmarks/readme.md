# Analyses in *Massi, Franco et al. (2022)*

This folder contains the source code files required to reproduce all the analyses described in the paper.

In particular, it consists of 4 files for the benchmark experiments run on *simulated data*:
1. _**hiPRS and Penalized PRSs.ipynb**_: a Jupyter Notebook that contains the code to run hiPRS and the benchmark Penalized PRSs (Lasso and Ridge penalizations) and collect results for the experiments described in Massi, Franco et al.
2. _**behravan.ipynb**_: a Jupyter Notebook that contains the python modules to run the algorithm described in [1]. The original code was collected from [2]. In the same notebook we include the code to run the benchmark experiment described in our paper.
This notebook generates and stores the simulated datasets, with our custom python function, that are automatically imported in the R file described below.
3. _**glinternet.R**_: an R script that runs the glinternet algorithm [3], freely distributed in the CRAN: [4]. This file uploads the simulated datasets generated in behravan.ipynb, therefore, to run this script they have to be located in the same working directory.
4. _**badre.ipynb**_: a Jupyter Notebook containing the source code to run the Deep Neural Network-based algorithm described in [5].



## References
[1] Behravan, H., Hartikainen, J.M., Tengström, M. et al. (2018) Machine learning identifies interacting genetic variants contributing to breast cancer risk: A case study in Finnish cases and controls. Sci Rep 8, 13149

[2] Behravan GitHub repository: https://github.com/hambeh/breast-cancer-risk-prediction

[3] Michael Lim & Trevor Hastie (2015) Learning Interactions via Hierarchical Group-Lasso Regularization, Journal of Computational and Graphical Statistics, 24:3, 627-654

[4] Glinternet R package: https://cran.r-project.org/web/packages/glinternet/index.html

[5] Badré, A., Zhang, L., Muchero, W. et al. (2021) Deep neural network improves the estimation of polygenic risk scores for breast cancer. J Hum Genet 66, 359–369.

