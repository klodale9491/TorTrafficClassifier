from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from util import load_merged_nibble_datasets
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
import numpy as np
import sys

'''
Esempio di esecuzione : python3 other_models_C.py -m bayes -n 50000 -c label
'''
args = sys.argv
# nel caso di riga di comando, più facile per listare jobs
try:
    model_type = args[args.index('-m') + 1]
    n_samples = int(args[args.index('-n') + 1])
    classification_type = args[args.index('-c') + 1]
# nel cado harcoding dei parametri
except ValueError:
    model_type = 'bayes'
    n_samples = 50
    classification_type = 'label'

print('Running script with params : ')
print('model_type : ' + str(model_type))
print('number_of_samples  : ' + str(n_samples))
print('classification_type : ' + str(classification_type))

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
X, Y = load_merged_nibble_datasets(datasets, samples=n_samples, classification=classification_type)

# classificatore naive di bayes
if model_type == 'bayes':
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
scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']
scores = cross_validate(model, X, Y, cv=10, scoring=scoring)

# salvataggio dei risultati
filename = model_type + '_' + str(n_samples) + '_' + classification_type + '.txt'
with open('results/other_c/' + filename, mode='w') as file:
    file.write('Accuracy : ' + str(np.mean(scores['test_accuracy'])) + '\n')
    file.write('Precision : ' + str(np.mean(scores['test_precision_micro'])) + '\n')
    file.write('Recall : ' + str(np.mean(scores['test_recall_micro'])) + '\n')
    file.write('F1 : ' + str(np.mean(scores['test_f1_micro'])) + '\n')
file.close()