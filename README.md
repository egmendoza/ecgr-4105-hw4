# ECGR 4105 Homework 4: Support Vector Machines

- `problem1_cancer_svm.py`: SVM classifier on the breast cancer dataset with linear, polynomial, RBF and sigmoid kernels, compared with the Homework 3 logistic regression results.
- `problem2_housing_svr.py`: SVR on the housing dataset with the same four kernels, compared with linear regression with an L2 penalty.
- `data/`: `Cancer.csv` and `Housing.csv`.
- `figures/`: plots written by the scripts.
- `results_problem1.txt`, `results_problem2.txt`: console output of each script.

## Running

```
pip install -r requirements.txt
python problem1_cancer_svm.py > results_problem1.txt
python problem2_housing_svr.py > results_problem2.txt
```

Both scripts use an 80/20 train/test split with `random_state = 42`.
