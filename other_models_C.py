'''
Esempio di esecuzione :
python3 other_models_C_OLD.py -m bayes -n 10 -c multi -t original
'''

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.callbacks import EarlyStopping
from util import load_merged_nibble_datasets
from sklearn.model_selection import StratifiedShuffleSplit
from util import oversample_dataset
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
x, y = load_merged_nibble_datasets(datasets, samples=n_samples, classification=classification_type, n_cols=n_columns, type_col=np.float32, ds_type=ds_type)

# kfold cross validation
n_splits = 10
kfold = StratifiedShuffleSplit(n_splits=n_splits)

# numero di classi e tipologia di queste
num_classes = 16
class_categorical = True

# inizializzazione delle metriche
total_accuracy = 0.0
total_precision = 0.0
total_recall = 0.0
total_fscore = 0.0

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

# apertura dei file per il salvataggio dei risutltati
with open('results/other_C2/' + str(n_samples) + '_samples.txt', 'w+') as file:
    early_stopping_monitor = EarlyStopping(patience=3)
    for train_index, test_index in kfold.split(x, y):
        x_train, y_train = oversample_dataset(x[train_index], y[train_index])
        x_train, y_train = x[train_index], y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        n_cols = x_train.shape[1]
        # allenamento e predizione del modello
        model.fit(x_train, y_train)
        # predizione dei dati di test
        y_pred = model.predict(x_test)
        if class_categorical:
            y_pred_vec = y_pred.argmax(1)
            y_test = y_test.argmax(1)
        else:
            y_pred_vec = np.asarray([1 if y[0] > 0.5 else 0 for y in y_pred])
        conf_mat = confusion_matrix(y_test, y_pred_vec)
        total_accuracy += accuracy_score(y_test, y_pred_vec)
        total_precision += precision_score(y_test, y_pred_vec, average='micro')
        total_recall += recall_score(y_test, y_pred_vec, average='micro')
        total_fscore += f1_score(y_test, y_pred_vec, average='micro')
        file.write(conf_mat + '\n')
# scrittura dei risultati medi di prestazione
file.write('Accuracy : ' + str(total_accuracy / n_splits) + '\n')
file.write('Precision : ' + str(total_precision / n_splits) + '\n')
file.write('Recall : ' + str(total_recall / n_splits) + '\n')
file.write('F1-Score : ' + str(total_fscore / n_splits) + '\n')
file.close()
