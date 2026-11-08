<div align="center">

# 5COTE
Bruno Casella and Marco Aldinucci

[![Conference](https://img.shields.io/badge/DAMI-2027-blue)](TODO)

</div>

# Overview

This repository contains the code to run and reproduce the experiments of the 5COTE algorithm and baselines.

Please cite as:

```bibtex
@article{casella20275COTE,
  author = {Casella, Bruno and Aldinucci, Marco},
  title        = {5COTE: a Vertical Federated Learning Ensemble for Time Series Classification},
  journal      = {Data Mining and Knowledge Discovery},
  volume       = {XX},
  pages        = {xxx--yyy},
  year         = {xxxx},
  url          = {xxx.yyy},
  doi          = {xxx.yyy},
}
```

# Abstract
Time series classification (TSC) is increasingly performed on data that cannot be centralised, such as multivariate signals whose channels are held by different parties. We introduce 5COTE, an ensemble for vertical federated multivariate TSC that adapts the principles of the state-of-the-art HIVE-COTE2.0 (HC2.0) algorithm to a setting where clients hold different variables of the same samples. 5COTE trains independent federated models of different families (kernel-, shapelet-, interval-based, and a neural component trained with split learning), each with its own aggregation scheme, and combines their class probabilities with a federated accuracy-weighted ensemble (CAWPE). As part of it, we propose FedDrCIF, a federated version of DrCIF. On 22 UEA datasets, 5COTE obtains the best average rank among the federated families and results comparable to the centralised HC2.0, and it clearly outperforms a client working alone with its own variables. Its accuracy remains stable when the variables are distributed among up to 16 clients, at an increasing computational cost, and depends on how the variables are assigned to the clients.

# Method
<img width="1004" height="1056" alt="5cote_method" src="https://github.com/user-attachments/assets/fe84210d-a6a5-446d-8b21-bbbfa31bff84" />

## Usage
- Clone this repo.
- Install the requirements. 
- Run experiments with the `code/vertical5cote.py` script.

## Results
The results directory reports the accuracy and F1-score metrics for all the considered datasets of the UEA Archive. Results are reported for both 5COTE and baselines and expressed in terms of accuracy and f1-score.

## Contributors
* Bruno Casella <bruno.casella@unito.it>  

* Marco Aldinucci <marco.aldinucci@unito.it>
