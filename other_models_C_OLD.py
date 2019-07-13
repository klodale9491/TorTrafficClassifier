'''
Esempio di esecuzione :
python3 other_models_C_OLD.py -m bayes -n 10 -c multi -t original
'''

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from util import load_merged_nibble_datasets
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
import numpy as np
import sys


args = sys.argv
try:
    model_type = args[args.index('-m') + 1]
    n_samples = int(args[args.index('-n') + 1])
    classification_type = args[args.index('-c') + 1]
    ds_type = args[args.index('-t') + 1]
except ValueError:
    model_type = 'bayes'
    n_samples = 10
    classification_type = 'multi'
    ds_type = 'original'

# avvio dello script con i parametri passati da riga di comando
print('Running script with params : ')
print('model_type : ' + model_type)
print('number_of_samples  : ' + str(n_samples))
print('classification_type : ' + classification_type)
print('ds_type : ' + ds_type)

# Caricamento datasets
datasets = [
    'audio',
    'browsing',
    'chat',
    'email',
    'file',
    'p2p',
    'video',
    'voip'
]

# numero di colonne del dataset
if ds_type == 'original':
    n_columns = 108 # features dell'esperimento originale
elif ds_type == 'convolution':
    n_columns = 1728 # features estratte dal modello convolutivo
elif ds_type == 'pca_convolution':
    n_columns = 625 # numero features custom estratte con PCA

# caricamento del dataset
X, Y = load_merged_nibble_datasets(datasets, samples=n_samples, classification=classification_type, n_cols=n_columns, type_col=np.float32, ds_type=ds_type)

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
filename = model_type + '_' + str(n_samples) + '_' + classification_type + '.' + ds_type
with open('results/other_C/' + filename, mode='w+') as file:
    file.write('Accuracy : ' + str(np.mean(scores['test_accuracy'])) + '\n')
    file.write('Precision : ' + str(np.mean(scores['test_precision_micro'])) + '\n')
    file.write('Recall : ' + str(np.mean(scores['test_recall_micro'])) + '\n')
    file.write('F1 : ' + str(np.mean(scores['test_f1_micro'])) + '\n')
file.close()
