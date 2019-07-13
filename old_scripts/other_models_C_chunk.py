from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from util import load_merged_nibble_chunked_datasets
from util import split_and_merge_chunk_data_class
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
import numpy as np
import sys

'''
Esempio di esecuzione : python3 other_models_C_OLD.py -m bayes -n 50000 -c label -t RAW_MIXED_CONV1D
'''
args = sys.argv
try:
    model_type = args[args.index('-m') + 1]
    n_samples = int(args[args.index('-n') + 1])
    classification_type = args[args.index('-c') + 1]
    ds_type = args[args.index('-t') + 1]
# nel cado harcoding dei parametri
except ValueError:
    model_type = 'rnd_for'
    n_samples = 1000
    classification_type = 'label'
    ds_type = 'RAW_MIXED_CONV1D'


# avvio dello script con i parametri passati da riga di comando
print('Running script with params : ')
print('model_type : ' + model_type)
print('number_of_samples  : ' + str(n_samples))
print('classification_type : ' + classification_type)
print('ds_type : ' + ds_type)


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
# ds_type = 'RAW_MIXED'
# ds_type = 'RAW_MIXED_CONV1D'
# ds_type = 'RAW_MIXED_CONV1D_PCA'


# numero di colonne del dataset
if ds_type == 'RAW_MIXED':
    n_columns = 108 # features dell'esperimento originale
elif ds_type == 'RAW_MIXED_CONV1D':
    n_columns = 1728 # features estratte dal modello convolutivo
elif ds_type == 'RAW_MIXED_CONV1D_PCA':
    n_columns = 625 # numero features custom estratte con PCA


'''
Il caricamento questa volta è fatto utilizzando i puntatori 
al chunk corrente
'''
chunks_ptr = {}
chunks_ptr = load_merged_nibble_chunked_datasets(datasets, samples=n_samples, classification=classification_type, n_cols=n_columns,
                                                 type_col=np.float32, ds_type=ds_type, chunks=10)

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


'''
Esecuzione del modello con 10-fold cross-validation per ognuno 
dei chunks
'''
test_accuracy = []
test_precision_micro = []
test_recall_micro = []
test_f1_micro = []
scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']


# processo tutti i chunk del dataset
X, Y = split_and_merge_chunk_data_class(chunks_ptr, protocols=datasets, n_cols=1728)
while X is not None and Y is not None:
    print('classifing ' + str(len(X)) + ' samples...')
    scores = cross_validate(model, X, Y, cv=10, scoring=scoring)
    test_accuracy.append(np.mean(scores['test_accuracy']))
    test_precision_micro.append(np.mean(scores['test_precision_micro']))
    test_recall_micro.append(np.mean(scores['test_recall_micro']))
    test_f1_micro.append(np.mean(scores['test_f1_micro']))
    X, Y = split_and_merge_chunk_data_class(chunks_ptr, protocols=datasets, n_cols=1728)


# calcolo degli indici di prestazione con la media delle medie
test_accuracy_glob = np.mean(test_accuracy)
test_precision_micro_glob = np.mean(test_precision_micro)
test_recall_micro_glob = np.mean(test_recall_micro)
test_f1_micro_glob = np.mean(test_f1_micro)


# salvataggio dei risultati
filename = model_type + '_' + str(n_samples) + '_' + classification_type + '.' + ds_type
with open('results/other_C/' + filename, mode='w+') as file:
    file.write('Accuracy : ' + str(test_accuracy_glob) + '\n')
    file.write('Precision : ' + str(test_precision_micro_glob) + '\n')
    file.write('Recall : ' + str(test_recall_micro_glob) + '\n')
    file.write('F1 : ' + str(test_f1_micro_glob) + '\n')
file.close()
