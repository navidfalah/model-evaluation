# -*- coding: utf-8 -*-
"""model_evaluation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13LTb6Nw6TUHuFxb7JgfUG2ZlumMua0X-
"""

! pip install mglearn

import mglearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_blobs(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression().fit(X_train, y_train)

print("test score : {:.2f}".format(logreg.score(X_test, y_test)))

mglearn.plots.plot_cross_validation()

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
logreg = LogisticRegression()

scores = cross_val_score(logreg, iris.data, iris.target)
print("cross validation scores : {}".format(scores))

scores = cross_val_score(logreg, iris.data, iris.target, cv=5)
print("cross validation scores : {}".format(scores))

print("average cross validation score {:.2f}".format(scores.mean()))

#### stratified k-fold cross validation

from sklearn.datasets import load_iris

iris = load_iris()
print("Iris labels:\n{}".format(iris.target))

mglearn.plots.plot_stratified_cross_validation()

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5)
print("cross validation scores : \n{}".format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

kfold = KFold(n_splits=3)
print("cross validation scores : \n{}".format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

kfold = KFold(n_splits=3, shuffle=True, random_state=0)
print("cross validation scores : \n{}".format(
    cross_val_score(logreg, iris.data, iris.target, cv=kfold)))

### leave one out
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
print("number of cv iterations : ", len(scores))
print("mean accuracy: {:>2f}".format(scores.mean()))

#### shuffle split cross validation
mglearn.plots.plot_shuffle_split()

from sklearn.model_selection import ShuffleSplit

shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("cross validation scores : \n{}".format(scores))

### cross validation with groups

from sklearn.model_selection import GroupKFold
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=12, random_state=0)
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]

scores = cross_val_score(logreg, X, y, groups=groups, cv=GroupKFold(n_splits=3))
print("cross validation scores : \n{}".format(scores))

### grid search

from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
print("size of training set : {} size of test set : {}".format(X_train.shape[0], X_test.shape[0]))

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

print("best score : {:.2f}".format(best_score))
print("best parameters : {}".format(best_parameters))

mglearn.plots.plot_threefold_split()

from sklearn.svm import SVC

X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=1)

print("Size of training set: {} size of validation set: {} size of test set:"
 " {}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_valid, y_valid)

        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)
test_score = svm.score(X_test, y_test)
print("best score on validation set: {:.2f}".format(best_score))
print("best parameters: ", best_parameters)
print("test set score with best parameters: {:.2f}".format(test_score))

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
      svm = SVC(gamma=gamma, C=C)
      scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
      score = np.mean(scores)

      if score>best_score:
        best_score = score
        best_parameters = {'C': C, 'gamma': gamma}

svm = SVC(**best_parameters)
svm.fit(X_trainval, y_trainval)

test_score = svm.score(X_test, y_test)

print("best score on validation set: {:.2f}".format(best_score))
print("best parameters: ", best_parameters)
print("test set score with best parameters: {:.2f}".format(test_score))

mglearn.plots.plot_cross_val_selection()

### using the mean accuracy as we selected 5 times to learn a model

mglearn.plots.plot_grid_search_overview()

param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 1, 10, 100]}
print("parameters to be tested: ")
print(param_grid)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

grid_search = GridSearchCV(SVC(), param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)
grid_search.fit(X_train, y_train)

print("test set score : {:.2f}".format(grid_search.score(X_test, y_test)))

print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

print("best estimator {}".format(grid_search.best_estimator_))

import pandas as pd

results = pd.DataFrame(grid_search.cv_results_)
display(results.head())

scores = np.array(results.mean_test_score).reshape(6, 6)

mglearn.tools.heatmap(scores, xlabel="gamma", xticklabels=param_grid['gamma'], ylabel="C", yticklabels=param_grid['C'], cmap="viridis")

fig, axes = plt.subplots(1, 3, figsize=(13, 5))

param_grid_linear = {'C': np.linspace(1, 2, 6), 'gamma': np.linspace(1, 2, 6)}
param_grid_one_log = {'C': np.linspace(1, 2, 6), 'gamma': np.logspace(-3, 2
                                                                       , 6)}
param_grid_range = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-7, -2, 6)}

for param_grid, ax in zip([param_grid_linear, param_grid_one_log, param_grid_range], axes):
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    scores = grid_search.cv_results_['mean_test_score'].reshape(6, 6)

    scores_image = mglearn.tools.heatmap(scores, xlabel="gamma", ylabel="C", xticklabels=param_grid['gamma'], yticklabels=param_grid['C'], cmap="viridis", ax=ax)

plt.colorbar(scores_image, ax=axes.tolist())

param_grid = [{"kernel":['bf'], 'C':[0.001, 0.01, 0.1, 10, 100], 'gamma':[0.001, 0.01, 0.1, 10, 100]}, {"kernel":['linear'], 'C':[0.001, 0.01, 0.1, 10, 100]}]

print("parameters to be tested: ")
print(param_grid)

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("test set score : {}".format(str(grid_search.best_params_)))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

results = pd.DataFrame(grid_search.cv_results_)
display(results.head())

scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv=5), iris.data, iris.target, cv=5)
print("cross validation scores : {}".format(scores))

def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
    outer_scores = []
    for training_indices, validation_indices in outer_cv.split(X, y):
      best_parms = {}
      best_score = -np.inf
      for parameters in parameter_grid:
        cv_scores = []

        for inner_train, inner_test in inner_cv.split(X[training_indices], y[training_indices]):
          clf = Classifier(**parameters)
          clf.fit(X[inner_train], y[inner_train])
          score = clf.score(X[inner_test], y[inner_test])
          cv_scores.append(score)

        mean_score = np.mean(cv_scores)
        if mean_score > best_score:
          best_score = mean_score
          best_parms = parameters

      clf = Classifier(**best_parms)
      clf.fit(X[training_indices], y[training_indices])
      outer_scores.append(clf.score(X[validation_indices], y[validation_indices]))
    return np.array(outer_scores)

param_grid = [{'kernel': ['rbf', 'linear'], 'C': [0.1, 1, 10]}] # Corrected 'bf' to 'rbf' in the param_grid

from sklearn.model_selection import ParameterGrid, StratifiedKFold

scores = nested_cv(iris.data, iris.target, StratifiedKFold(5),
 StratifiedKFold(5), SVC, ParameterGrid(param_grid))
print("Cross-validation scores: {}".format(scores))

from sklearn.datasets import load_digits

digits = load_digits()
y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)

from sklearn.dummy import DummyClassifier

dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
pred_most_frequent = dummy_majority.predict(X_test)
print("unique predicted labels: {}".format(np.unique(pred_most_frequent)))
print("test score: {:.2f}".format(dummy_majority.score(X_test, y_test)))

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=2).fit(X_train, y_train)
pred_tree = tree.predict(X_test)
print("test score: {:.2f}".format(tree.score(X_test, y_test)))

from sklearn.linear_model import LogisticRegression

dummy = DummyClassifier().fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)
print("dummy score: {:.2f}".format(dummy.score(X_test, y_test)))

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("logreg score: {:.2f}".format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, pred_logreg)
print("confusion matrix: \n{}".format(confusion))

mglearn.plots.plot_confusion_matrix_illustration()

mglearn.plots.plot_binary_confusion_matrix()

print("most frequent class: ")
print(confusion_matrix(y_test, pred_most_frequent))
print("\ntree: ")
print(confusion_matrix(y_test, pred_tree))
print("\ndummy: ")
print(confusion_matrix(y_test, pred_dummy))
print("\nlogreg: ")
print(confusion_matrix(y_test, pred_logreg))

from sklearn.metrics import f1_score

print("f1 score most frequent: {:.2f}".format(f1_score(y_test, pred_most_frequent)))
print("f1 score tree: {:.2f}".format(f1_score(y_test, pred_tree)))
print("f1 score dummy: {:.2f}".format(f1_score(y_test, pred_dummy)))
print("f1 score logreg: {:.2f}".format(f1_score(y_test, pred_logreg)))

from sklearn.metrics import classification_report

print(classification_report(y_test, pred_most_frequent, target_names=['not nine', 'nine']))

print(classification_report(y_test, pred_tree, target_names=['not nine', 'nine']))
print(classification_report(y_test, pred_logreg, target_names=['not nine', 'nine']))

from mglearn.datasets import make_blobs
import numpy as np # Import numpy for array creation
X, y = make_blobs(n_samples=(400, 50), centers=np.array([[1,1],[2,2]]), cluster_std=[7.0, 2], # Modified centers to be array-like. This example uses 2 centers at [1,1] and [2,2].
 random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)

mglearn.plots.plot_decision_threshold()

print(classification_report(y_test, svc.predict(X_test)))

y_pred_lower_threshold = svc.decision_function(X_test) > -.8

print(classification_report(y_test, y_pred_lower_threshold))

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt


# The centers parameter needs to be array-like if n_samples is a sequence.
# Since you want 2 centers, you can specify the center locations like this:
X, y = make_blobs(n_samples=(4000, 500), centers=[[0, 0], [5, 5]], cluster_std=[7.0, 2], random_state=22)
# This creates two centers at [0, 0] and [5, 5]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

svc = SVC(gamma=.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))

close_zero = np.argmin(np.abs(thresholds))
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10, label="threshold 0", fillstyle="none", c='k', mew=2)
plt.plot(precision, recall, label="precision recall curve")
plt.xlabel("precision")
plt.ylabel("recall")
plt.legend(loc="best")
plt.show() # Added to display the plot

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

percision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])

plt.plot(precision, recall, label="svc")
plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10, label="svc threshold 0", fillstyle="none", c='k', mew=2)
plt.plot(percision_rf, recall_rf, label="rf")

close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(percision_rf[close_default_rf], recall_rf[close_default_rf], '^', markersize=10, label="rf threshold 0.5", fillstyle="none", c='k', mew=2)
plt.xlabel("precision")
plt.ylabel("recall")
plt.legend(loc="best")
plt.show()

print("f1 score svc: {:.2f}".format(f1_score(y_test, svc.predict(X_test))))
print("f1 score rf: {:.2f}".format(f1_score(y_test, rf.predict(X_test))))

from sklearn.metrics import average_precision_score

ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print("average precision svc: {:.3f}".format(ap_svc))
print("average precision rf: {:.3f}".format(ap_rf))

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
plt.plot(fpr, tpr, label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
# find threshold closest to zero
close_zero = np.argmin(np.abs(thresholds))
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
 label="threshold zero", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)

from sklearn.metrics import roc_curve

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label="ROC Curve SVC")
plt.plot(fpr_rf, tpr_rf, label="ROC Curve RF")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")

plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, label="threshold zero SVC", fillstyle="none", c='k', mew=2)
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(fpr_rf[close_default_rf], tpr[close_default_rf], '^', markersize=10, label="threshold 0.5 RF", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)

from sklearn.metrics import roc_auc_score

rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
svc_auc = roc_auc_score(y_test, svc.decision_function(X_test))
print("AUC for RF: {:.3f}".format(rf_auc))
print("AUC for SVC: {:.3f}".format(svc_auc))

y = digits.target == 9

X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state=0)
plt.figure()

for gamma in [1, 0.05, 0.01]:
    svm = SVC(gamma=gamma).fit(X_train, y_train)
    accuracy = svm.score(X_test, y_test)
    auc = roc_auc_score(y_test, svm.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test, svm.decision_function(X_test), pos_label=True)
    print("gamma = {:.2f} accuracy = {:.2f} auc = {:.2f}".format(gamma, accuracy, auc))
    plt.plot(fpr, tpr, label="gamma={:.3f}".format(gamma))

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.xlim(-0.01, 1)
plt.ylim(0, 1.02)

### metrics for multiclass classifications

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0)

lr = LogisticRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print("accuracy: {:.2f}".format(accuracy_score(y_test, pred)))
print("confusion matrix: {}".format(confusion_matrix(y_test, pred)))

scores_image = mglearn.tools.heatmap(
 confusion_matrix(y_test, pred), xlabel='Predicted label',
 ylabel='True label', xticklabels=digits.target_names,
 yticklabels=digits.target_names, cmap=plt.cm.gray_r, fmt="%d")
plt.title("Confusion matrix")
plt.gca().invert_yaxis()

print(classification_report(y_test, pred))

print("micro average f1 score: {:.3f}".format(f1_score(y_test, pred, average="micro")))
print("macro average f1 score: {:.3f}".format(f1_score(y_test, pred, average="macro")))

print("Default scoring: {}".format(cross_val_score(SVC(), digits.data, digits.target==9)))

explicit_accuracy = cross_val_score(SVC(), digits.data, digits.target==9, scoring="accuracy")
print("explicit accuracy scoring: {}".format(explicit_accuracy))

roc_auc = cross_val_score(SVC(), digits.data, digits.target==9, scoring="roc_auc")
print("explicit roc_auc scoring: {}".format(roc_auc))

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=0)

param_grid = {'gamma':[0.0001, 0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(SVC(), param_grid=param_grid)
grid.fit(X_train, y_train)

print("Grid-Search with accuracy")
print("Best parameters:", grid.best_params_)
print("Best cross-validation score (accuracy)): {:.3f}".format(grid.best_score_))
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))

grid = GridSearchCV(SVC(), param_grid=param_grid, scoring="roc_auc")
grid.fit(X_train, y_train)
print("\nGrid-Search with AUC")
print("Best parameters:", grid.best_params_)
print("Best cross-validation score (AUC): {:.3f}".format(grid.best_score_))
print("Test set AUC: {:.3f}".format(
 roc_auc_score(y_test, grid.decision_function(X_test))))
print("Test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))

from sklearn.metrics import SCORERS # Changed import statement
print("Available scorers:\n{}".format(sorted(SCORERS.keys())))

