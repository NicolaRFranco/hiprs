# The hiprs package
*hiPRS* is a novel algorithm for building Polygenic Risk Scores (PRSs) that account for high-order interaction terms between genetic loci (encoded in terms of Single Nucleotide Polymorfisms, SNPs). This repository provides a Python implementation of the method, as detailed in [Massi, Franco et al., Learning High-Order Interactions for Polygenic Risk Prediction (biorxiv preprint, 2022)](https://www.biorxiv.org/content/10.1101/2022.04.22.489134v1).

The repository consists of:
- The source files required for installing the *hiprs* library (see **Installation guidelines** below). The latter is comprised of 4 main modules: 
    - *scores.py*, which allows the user to construct the Polygenic Risk Scores; 
    - *interactions.py*, a module for efficiently handling interaction terms; 
    - *mrmr.py* an auxiliary module that implements Maximum Relevance — Minimum Redundancy algorithms; 
    - *snps.py* an auxiliary module used for data simulation. 
    
  All modules come with a complete internal documentation that can be accessed via the Python **help** command.
- A folder named *benchmark/*, containing the Jupyter Notebook *hiPRS and Penalized PRSs.ipynb*. There, hiPRS is tested on simulated data and compared with traditional scoring methods. In the same folder, we include additional Jupyter Notebooks and R files where other scoring methods are considered as benchmark. For a comprehensive description of these we refer to the additional readme file contained in the folder, and the aforementioned work by Massi, Franco et al.


# Installation guidelines
The hiprs package can be installed via *pip* or *conda* directly from the current GitHub repository.

*Installation in Anaconda environments*
- First, run the command **conda install git pip** to enable *pip* in in your conda environment,
- Then run **pip install git+https://github.com/NicolaRFranco/hiprs.git**.

*Installation via pip*
- Simply run **pip install git+https://github.com/NicolaRFranco/hiprs.git**.

Modules can then be imported either as **import hiprs.scores** or **from hiprs import scores**.

*Remark*. The hiprs library is based on the following Python packages: *mlxtend*, *numpy*, *pandas*, *scipy*, *scikit-learn*. These should be automatically installed by either conda or pip when running the commands above.


# Development
This repository was developed and is currently maintained by [Nicola Rares Franco](https://github.com/NicolaRFranco/)<sup>1</sup> and [Michela Carlotta Massi](https://github.com/MichCM)<sup>1,2</sup>.

<sup>1</sup> MOX, Department of Mathematics, Politecnico di Milano, Milan, Italy <br>
<sup>2</sup> CHDS - Center for Health Data Science, Human Technopole, Milan, Italy

The *hiprs* algorithm and methodology were developed at [MOX - Politecnico di Milano](https://mox.polimi.it/) under the support of the [RADPrecise project](https://www.era-learn.eu/network-information/networks/era-permed/1st-joint-transnational-call-for-proposals-2018/personalized-radiotherapy-incorporating-cellular-response-to-irradiation-in-personalized-treatment-planning-to-minimize-radiation-toxicity) (ERA-NET ERA PerMed / FRRB grant agreement n. ERAPERMED2018-244).
