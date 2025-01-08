# Changepoint Detection in Dynamic Graphs

## Overview
This repository provides a Python implementation of changepoint detection in dependent dynamic nonparametric random dot product graphs (NonPar-RDPG-CPD). It is based on the theoretical framework developed by Oscar Hernan Madrid Padilla et al. in their paper ["Change point localization in dependent dynamic nonparametric random dot product graphs"](https://arxiv.org/abs/1911.07494).
It includes the original simulation study in the paper as well as an implementation with the Enron email dataset https://www.cis.jhu.edu/~parky/Enron/.

## Repository Structure
- **RDPG.py** - Core implementation of the NonPar-RDPG-CPD algorithm.
- **SimulationStudy.py** - Script for generating synthetic graph data for simulations.
- **utils.py** - Utility functions for PCA scaling, matrix operations, and BIC calculations.
- **enron.py** - Script for loading and processing the Enron email dataset into adjacency matrices.
- **main.py** - Main script to run changepoint detection experiments.
- **config/** - Configuration files for tuning parameters.


### Running the Code

1. **Enron Dataset**  
   ```bash
   python main.py --dataset enron
   ```

2. **Simulated Data**  
   ```bash
   python main.py --dataset simulation
   ```

# 📖 **References**

1. 📄 O. H. M. Padilla, Y. Yu, and C. E. Priebe. *"Change point localization in dependent dynamic nonparametric random dot product graphs,"* Journal of Machine Learning Research, 2022.  
2. 🔗 Haotian Xu. *"[Changepoint detection on a graph of time series,"* GitHub repository link](https://github.com/HaotianXu/changepoints?utm_source=chatgpt.com).  

---

# 📬 **Contact**

For questions, feel free to contact:  
**Max Riffi-Aslett** 📧 **max.riffi3@gmail.com**
