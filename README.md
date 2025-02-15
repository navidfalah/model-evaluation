# 📊 Model Evaluation Techniques

Welcome to the **Model Evaluation** repository! This project explores various techniques for evaluating machine learning models, including cross-validation, grid search, and performance metrics like precision, recall, and ROC curves.

---

## 📂 **Project Overview**

This repository demonstrates how to evaluate machine learning models using **Scikit-learn** and **mglearn**. It covers:

- **Cross-validation** techniques (e.g., K-Fold, Stratified K-Fold, Leave-One-Out)
- **Grid Search** for hyperparameter tuning
- **Performance Metrics** like accuracy, precision, recall, F1-score, and ROC-AUC
- **Confusion Matrix** and **Classification Reports**
- **Precision-Recall Curves** and **ROC Curves**

---

## 🛠️ **Tech Stack**

- **Python**
- **Scikit-learn**
- **mglearn**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**

---

## 📊 **Dataset**

The project uses the following datasets:
- **Iris Dataset** (for multiclass classification)
- **Synthetic Blob Data** (for binary classification)
- **Digits Dataset** (for multiclass classification)

---

## 🧠 **Key Concepts**

### 1. **Cross-Validation**
- **K-Fold Cross-Validation**: Splits data into `k` folds and evaluates the model `k` times.
- **Stratified K-Fold**: Preserves the percentage of samples for each class.
- **Leave-One-Out (LOO)**: Uses one sample as the test set and the rest as the training set.
- **Shuffle-Split**: Randomly splits data into training and test sets multiple times.

### 2. **Grid Search**
- Exhaustive search over specified parameter values for an estimator.
- Uses cross-validation to evaluate each combination of hyperparameters.

### 3. **Performance Metrics**
- **Accuracy**: Percentage of correctly classified samples.
- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.
- **F1-Score**: Harmonic mean of precision and recall.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.

### 4. **Visualizations**
- **Confusion Matrix**: Visualizes true vs. predicted labels.
- **Precision-Recall Curve**: Plots precision vs. recall for different thresholds.
- **ROC Curve**: Plots True Positive Rate (TPR) vs. False Positive Rate (FPR).

---

## 🚀 **Code Highlights**

### Cross-Validation
```python
scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("Cross-validation scores: {}".format(scores))
```

### Grid Search
```python
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: {}".format(grid_search.best_params_))
```

### Precision-Recall Curve
```python
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))
plt.plot(precision, recall, label="Precision-Recall Curve")
```

### ROC Curve
```python
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
plt.plot(fpr, tpr, label="ROC Curve")
```

---

## 🛠️ **Installation**

1. Clone the repository:
   ```bash
   git clone https://github.com/navidfalah/model-evaluation.git
   cd model-evaluation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook model_evaluation.ipynb
   ```

---

## 🤝 **Contributing**

Feel free to contribute to this project! Open an issue or submit a pull request.

---

## 📧 **Contact**

- **Name**: Navid Falah
- **GitHub**: [navidfalah](https://github.com/navidfalah)
- **Email**: navid.falah7@gmail.com
