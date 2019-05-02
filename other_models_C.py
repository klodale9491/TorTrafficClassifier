from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from util import load_merged_nibble_datasets
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np


# Inizializzazione del modello
model_type = 'log_reg'
model = None

# Caricamento datasets
datasets = [
    'AUDIO-STREAMING',
    'BROWSING',
    'EMAIL',
    'CHAT',
    'FILE-TRANSFER',
    'P2P',
    'VIDEO-STREAMING',
    'VOIP'
]
# X, Y = load_raw_nibble_dataset(datasets[0])
X, Y = load_merged_nibble_datasets(datasets, samples=50000, classification='label')

# classificatore naive di bayes
if model_type == 'log_reg':
    model = GaussianNB()
# regressione logistica
elif model_type == 'log_reg':
    model = LogisticRegression(solver='liblinear')
# random forest
elif model_type == 'rnd_for':
    model = RandomForestClassifier(n_jobs=2, random_state=0)
# support vector machine
elif model_type == 'svm':
    model = svm.SVC(kernel='rbf', C=1)

# Esecuzione del modello con 10-fold cross-validation
accuracy_scores = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
recall_scores = cross_val_score(model, X, Y, cv=10, scoring='recall_micro')
precision_scores = cross_val_score(model, X, Y, cv=10, scoring='precision_micro')
f1_scores = cross_val_score(model, X, Y, cv=10, scoring='f1_micro')

# Risultati
print('Scores after 10-fold cross-validation : ' + model_type)
print(['Accuracy : ', np.mean(accuracy_scores)])
print(['Recall : ', np.mean(recall_scores)])
print(['Precision : ', np.mean(precision_scores)])
print(['F1 : ', np.mean(f1_scores)])
