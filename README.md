# The hiprs package
*hiPRS* is a novel algorithm for building Polygenic Risk Scores (PRSs) that account for high-order interaction terms between genetic loci (encoded in terms of Single Nucleotide Polymorfisms, SNPs). The methodology was developed in *Massi et al., Learning High-Order Interactions for Polygenic Risk Prediction (2022)*, and this repository provides a Python implementation of the routines thereby detailed.

The repository consists of:
- The source files required for installing the *hiprs* library (see **Installation guidelines** below). The latter consists of 4 main modules: the main one, *scores.py*, which allows the user to construct the Polygenic Risk Scores; *interactions.py*, a module for efficiently handling interaction terms; *mrmr.py* an auxiliary module that implements Maximum Relevance — Minimum Redundancy algorithms; *snps.py* an auxiliary module used for data simulation. All the modules come with a complete internal documentation: e.g., after having imported the module *hiprs.scores*, run the command *help(hiPRS)* to see further details on hiPRS like objects.
- A Jupyter Notebook, *benchmark/hiPRS and Penalized PRSs.ipynb*, where hiPRS is tested on simulated data and compared with traditional scoring methods. In the same folder, we include additional Jupyter Notebooks and R files where other scoring methods are considered as benchmark. For a comprehensive description of these we refer to the aforementioned work by Massi et al.


# Installation guidelines
The hiprs package can be installed via *pip* or *conda* directly from the current GitHub repository.

*Installation in Anaconda environments*
- First, run the command **conda install git pip** to enable *pip* in in your conda environment,
- Then run **pip install git+https://github.com/NicolaRFranco/hiprs.git**.

*Installation via pip*
- Simply run **pip install git+https://github.com/NicolaRFranco/hiprs.git**.

Congratulations, you have now installed the hiprs package. Modules can be imported as *import hiprs.scores*, *from hiprs import snps* etc.

*Remark*. The hiprs library is based on the following Python packages: *mlxtend*, *numpy*, *pandas*, *scipy*, *scikit-learn*. However, these should be automatically installed by either conda or pip when running the commands above.
