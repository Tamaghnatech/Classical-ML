# Classical Machine Learning

This repository captures the full suite of classical ML — including regression, classification, clustering, dimensionality reduction, and ensemble methods — implemented with rigorous preprocessing and evaluation.

---

## 🧠 Topics Covered

### 🔍 Supervised Learning

#### Regression
- Linear Regression (OLS)
- Ridge / Lasso / ElasticNet
- Polynomial Regression
- Poisson & Gamma Generalized Linear Models (GLM)
- Spline Regression

📌 **Projects:**
---
### 🔗 Featured Full-Stack Project

#### 🏠 [House Price Prediction (Full Stack)](https://github.com/Tamaghnatech/HousePricePrediction)

A **production-grade, end-to-end regression pipeline** implementing:

* **Linear Regression (OLS)**
* **Ridge Regression**
* **Lasso Regression**
* **ElasticNet Regression**

🚀 Features:

* 📊 Cross-validated model comparisons
* 🪄 Auto-logging with **Weights & Biases (wandb)**
* 💾 Model saving and artifact tracking
* 🌐 **Interactive Streamlit dashboard** with real-time metrics
* 📁 Clean repo structure, 📉 Matplotlib plots, and 📄 Beautiful README

> *This repo serves as a showcase of how classical linear models can be taken all the way to production-like deployment using modern tools.*

---
- CO2 Emissions vs Engine Size (Polynomial Fit)
- French Motor Insurance Claims (Poisson/Gamma GLM)

#### Classification
- Logistic Regression
- Naive Bayes
- k-Nearest Neighbors (k-NN)
- Linear Discriminant Analysis (LDA)
- SVM (Linear & RBF)

📌 **Projects:**
- SMS Spam Detection
- Credit Card Default Prediction
- Handwritten Digit Recognition (MNIST)
- Wine Type Classification

---

### 🌲 Tree-Based Models
- Decision Tree / Regression Tree (CART)
- Random Forest
- Extra Trees
- Feature Importance & Visualization

📌 **Projects:**
- Titanic Survival Classification
- Heart Disease Prediction
- Diabetes Progression (Regression Tree)

---

### ⚡ Boosting Algorithms
- AdaBoost
- Gradient Boosting
- XGBoost, LightGBM, CatBoost

📌 **Projects:**
- Income Classification (>$50k)
- Loan Approval Prediction
- Gradient Boosting vs AdaBoost Comparison

---

### 📈 Model Evaluation & Selection
- Regression Metrics: MSE, RMSE, MAE, R², MAPE
- Classification Metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, Cohen’s Kappa
- Ranking Metrics: nDCG, MAP
- Cross-validation: k-Fold, Stratified, Time-Series, Nested
- Hyperparameter Tuning: GridSearchCV, RandomizedSearchCV, Optuna

---

### 🧼 Preprocessing & Feature Engineering
- Handling Missing Data
- Scaling: Standard, MinMax, Robust
- Encoding: One-Hot, Target, Frequency
- Log Transform, Box-Cox
- Feature Generation (Polynomial, Interaction)
- Date/Time Feature Extraction
- Text Vectorization: TF-IDF, CountVectorizer

📌 **Datasets:**
- Adult Income Dataset
- Housing Prices (Kaggle)
- UCI Heart, Wine, and Credit Card datasets

---

### 🤖 Unsupervised Learning

#### Clustering
- K-Means
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models (GMM)

📌 **Projects:**
- Customer Segmentation (Mall Data)
- Gene Expression Clustering
- Credit Card Fraud Detection (Anomaly DBSCAN)
- Speaker Identification with GMMs

#### Dimensionality Reduction
- PCA, t-SNE, UMAP
- Autoencoders (shallow)

📌 **Projects:**
- MNIST Visualization (t-SNE, PCA)
- Cancer Gene Mapping (UMAP)
- Image Compression with PCA

#### Association Rule Mining
- Apriori, Eclat, FP-Growth

📌 **Projects:**
- Market Basket Analysis (Groceries)
- Bookstore Purchase Pattern Mining
- Retail Rule Mining on Transactions

---

## 📁 Folder Structure

- `notes/`  
  📚 Concepts, formulas, PDFs, cheat sheets.

- `projects/`  
  💻 Google Colab notebooks implementing supervised and unsupervised models end-to-end.

- `diagrams/`  
  🎨 Manim/draw.io visualizations of decision trees, clusters, PCA, bias-variance, etc.

---

## 🛠️ Suggested Add-ons
- SHAP & LIME for model interpretability
- Pipelines with `sklearn.pipeline`
- Custom Transformers
- Model versioning using DVC

---

> Mastering classical ML gives you both intuition and control.  
> Deep learning begins — but never replaces — this foundation.
