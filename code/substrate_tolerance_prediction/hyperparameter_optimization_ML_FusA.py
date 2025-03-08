import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from datasets import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score, confusion_matrix, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Load data
data = pd.read_csv("../data/FusA_tolerance_dataset.csv")
seqs = data.iloc[:,0].tolist()
Xs = np.load("../data/FusA_LassoESM.npy")
ys = data.iloc[:,1].tolist()

print("Optimization for Models trained on FusA")
# List of models and their corresponding names and hyperparameters for grid search
model_list = [RandomForestClassifier, AdaBoostClassifier, SVC, MLPClassifier]
model_names = ['RF', 'AdaBoost', 'SVC', 'MLP']
parameters_list = [
    {'classifier__n_estimators': [20, 50, 100, 200], 'classifier__max_depth': [10,20,50,100], 'classifier__max_features':('sqrt','log2')},
    {'classifier__n_estimators': [20, 50, 100, 200], 'classifier__learning_rate': [0.1, 1, 5, 10]},
    {'classifier__kernel':('linear', 'rbf', 'sigmoid','poly'), 'classifier__C':[0.1, 1, 10]},
    {'classifier__hidden_layer_sizes': [32,64,128,256,512, (512, 64), (256, 32), (128,32)], 'classifier__batch_size':[16,32], 'classifier__learning_rate_init':[0.01, 0.001], 'classifier__max_iter':[1000], 'classifier__early_stopping':[True]},
]

# Perform grid search for each model
for model, name, parameters in zip(model_list, model_names, parameters_list):
    result_list = []
    steps = [
        #('pca', PCA(n_components=100)),
        ('classifier', model())
    ]
    pipeline = Pipeline(steps)
    grid_search = GridSearchCV(pipeline, parameters,n_jobs = -1, cv=10, scoring='balanced_accuracy')
    grid_search.fit(Xs, ys)
    # Collect and sort the results
    result_list.append(pd.DataFrame.from_dict(grid_search.cv_results_))
    aa = result_list[0].sort_values('rank_test_score')

    # Print the best parameters and the top 5 hyperparameters combination
    print(f'Best paramters for {name} {grid_search.best_params_}')
    print(f'Best model for {name} {grid_search.best_estimator_}')
    print(aa.iloc[:5,4:])
    
