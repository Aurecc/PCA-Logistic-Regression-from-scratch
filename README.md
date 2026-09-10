# 🧠 PCA & Logistic Regression: Behind the Scenes (From Scratch)

This project provides a comprehensive, ground-up implementation of **Principal Component Analysis (PCA)** and **Logistic Regression** built completely **from scratch**. The main objective is to explore the underlying mathematical foundations (Covariance Matrices, Eigenvalues, Gradient Descent) without relying on high-level libraries like Scikit-Learn for the core algorithms, while solving a credit risk classification problem.

## 🎯 Project Objectives
1. **Mathematical Depth:** Implement Machine Learning algorithms at the matrix and algebraic level.
2. **Business Vision:** Optimize the model not just for statistical metrics, but to minimize expected financial cost (Cost Matrix).
3. **Production Quality:** Develop a robust pipeline with probability calibration (*Platt Scaling*) for real-world pricing applications.

---

## 📊 Dataset
We use the **Default of Credit Card Clients** dataset from the UC Irvine Machine Learning Repository.
* **Volume:** 30,000 observations and 24 variables.
* **Target:** Default payment probability (Yes = 1, No = 0).
* **Imbalance:** ~22% default rate.

---

## ⚙️ Architecture & Pipeline

### 1. Data Transformation & EDA
* **Feature Engineering:** Correction of categorical inconsistencies and ordinal remapping.
* **Logarithmic Transformation:** Applied to skewed monetary variables ($$log(x) = sign(x) \times \log_{10}(|x| + 1)$$) to approximate Gaussian distributions.
* **Zero-Flagging:** Creation of boolean variables to capture the structural information of zero balances.

### 2. PCA (Custom Implementation)
Feature extraction and dimensionality reduction using pure linear algebra:
* **Covariance Matrix:** Computation of $$Cov(X) = \frac{1}{n-1}(X - \bar{X})^T(X - \bar{X})$$.
* **Jacobi Method:** Iterative algorithm implemented to find Eigenvectors and Eigenvalues using orthogonal rotations.
* **Projection:** Transformation of original data into the new dimensional space.

### 3. Logistic Regression (Custom Implementation)
* **Cost Function:** Cross-Entropy with **L2 Regularization (Ridge)**.
* **Optimization:** Gradient Descent with partial derivatives calculated via the chain rule.
* **Class Weights:** Integrated directly into the loss function to tackle the 22% class imbalance ($$w_0 \approx 0.64$$, $$w_1 \approx 2.27$$).

### 4. Calibration & Evaluation (Dual Engine)
* **Platt Scaling:** Training a calibrator model on Out-of-Fold (OOF) probabilities to correct the probability inflation introduced by class weights.
* **Cross-Validation:** Stratified K-Fold coupled with Random Search for hyperparameter tuning.

---

## 📈 Results & Conclusions

### Benchmarking vs Scikit-Learn
The custom model was rigorously compared against `sklearn.decomposition.PCA` and `sklearn.linear_model.LogisticRegression`:
* **Relative Error (L2 Norm):** < 1% for coefficients and intercept.
* **Explained Variance:** Error close to 0%.
* **Confusion Matrix:** Identical results (margin of error < 0.01%).

### Model Performance
* **PR AUC:** **0.51** (A **127%** improvement over the 0.22 random baseline).
* **ROC AUC:** **0.75**.
* **Interpretability (PC2):** The second principal component (12% variance) proved to be the strongest predictive feature, acting as a proxy for **"Credit History"**.

### Business Optimization (Cost Matrix)
In credit risk, a False Negative (approving a bad payer) is significantly more expensive than a False Positive (rejecting a good client). 
* A **1:5** cost ratio was applied (FP: 500, FN: 2500).
* The decision threshold was dynamically optimized to minimize the total expected cost, demonstrating that **maximizing Recall is financially superior to maximizing overall Accuracy**.
