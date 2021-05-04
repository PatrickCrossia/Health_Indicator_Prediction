# Health Indicator Prediction
[Data](https://github.com/anonymousAuthor404/Health_Indicator_Prediction/tree/master/data/data_cleaned) and experiment code for replication of project health indicator study.

The [paper](https://arxiv.org/pdf/2006.07240.pdf) has been submitted to EMSE 2021. 3 RQs asked in the paper can be answered here.

## Abstract

Software developed on  public platforms is a source of data that can be used to make predictions about those projects. While the individual developing activity may be random and hard to predict, when large groups of developers work together on software projects, the developing behavior on project level can be predicted with good accuracy. 

To demonstrate this, we use 64,181 months of data from 1,159 GitHub projects to make various predictions about the current status of those projects (as of April 2020). We find that traditional estimation algorithms make many mistakes. Algorithms like k-nearest neighbors (KNN), support vector regression (SVR), random forest (RFT), linear regression (LNR), and regression trees (CART) have high error rates. But that error rate can be  greatly reduced using the DECART hyperparameter optimization. DECART is a differential evolution (DE) algorithm that tunes the CART data mining system to the particular details of a specific project.

To the best of our knowledge, this is the largest study yet conducted, using recent data for predicting multiple health indicators of open-source projects. 

## Experiment Replication

To reproduce the experiment results, execute `main.py` in directory `experiments`, you will get all `mon{a}_g{b}_mt{c}_{d}.csv` file which stores the numeric experiment results in directory `result_experiment`. Here, `a` indicates which month to predict (1,3,6,12), `b` indicates which goal to predict (1-7), `c` indicates which metrics from prediction (0, 1) and `d` is the name of the project. The details about these parameters are commented in `main.py`. 

We store all experiment data in directory `data/data_cleaned` [Data](https://github.com/anonymousAuthor404/Health_Indicator_Prediction/tree/master/data/data_cleaned), the stats information of data can be obtained by excuting `stats/stats_data.py`.

To represent the results of RQ1, run `stats.py` in directory `RQs/experiments`, the results (Table 6-9) are stored in `RQs/result_stats/stats.csv`.

To represent the results of RQ2, run `internal_feature_select.py` in directory `RQs/validation`, to get result in (Table 11), note that our methods use a modified version of `sklearn.tree.export_text`, please see comments in code for details.

To represent the results of RQ3, use the stats results generated in `RQs/result_stats/stats.csv` since its information are already collected from `stats.py` (Table 12-13).

For results of predicting mid-lifecycle in Discussion section (Table 14), run `runner.py` in directory `RQs/experiments`, and set related data range into `N-24`, please see comments in code for details.
