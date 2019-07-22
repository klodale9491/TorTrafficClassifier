'''
Esempio di esecuzione :
python3 other_models_C.py -m bayes -n 10 -c multi -t original -pc 20
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
import numpy as np
import sys
import os

args = sys.argv
try:
    model_type = args[args.index('-m') + 1]
except:
    model_type = 'bayes'
try:
    n_samples = int(args[args.index('-n') + 1])
except:
    n_samples = 10
try:
    classification_type = args[args.index('-c') + 1]
except:
    classification_type = 'multi'
try:
    ds_type = args[args.index('-t') + 1]
except:
    ds_type = 'original'
try:
    pca_comp = args[args.index('-p') + 1]
except:
    pca_comp = None

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

# caricamento del dataset
x, y = load_merged_nibble_datasets(datasets, samples=n_samples, classification=classification_type, n_cols=n_columns, type_col=np.float32, ds_type=ds_type)

# Faccio PCA, facendo prima rescaling dei dati se richiesto
if pca_comp is not None:
    print('Running PCA ...')
    scaler = MinMaxScaler(feature_range=[0, 1])
    x = scaler.fit_transform(x[0:, 0:n_columns])
    pca = PCA(n_components=int(pca_comp))
    x = pca.fit_transform(x)

# kfold cross validation
n_splits = 10
kfold = StratifiedShuffleSplit(n_splits=n_splits)

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
    model = RandomForestClassifier(n_jobs=16, random_state=0)
# support vector machine
elif model_type == 'svm':
    model = svm.SVC(kernel='rbf', C=1)

# cartella dei risultati per l'esperimento, metriche e matrici di confusione
result_path_type = 'results/other_C/' + ds_type + '/'
if not os.path.exists(result_path_type):
    os.mkdir(result_path_type)
result_path_samples = result_path_type + str(n_samples) + '/'
if not os.path.exists(result_path_samples):
    os.mkdir(result_path_samples)
result_path_model = result_path_samples + model_type + '/'
if pca_comp is not None:
    result_path_model += 'pca_' + str(pca_comp) + '/'
if not os.path.exists(result_path_model):
    os.mkdir(result_path_model)
metrics_file = open(result_path_model + 'metrics.txt', 'w+')

index = 1
conf_mat_tot = None
with open(result_path_model + 'metrics.txt', 'w+') as metrics_file:
    early_stopping_monitor = EarlyStopping(patience=3)
    for train_index, test_index in kfold.split(x, y):
        # x_train, y_train = oversample_dataset(x[train_index], y[train_index])
        x_train, y_train = x[train_index], y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        n_cols = x_train.shape[1]
        # allenamento e predizione del modello
        model.fit(x_train, y_train)
        # predizione dei dati di test e calcolo delle matrici di confusione
        y_pred = model.predict(x_test)
        conf_mat = confusion_matrix(y_test, y_pred)
        if index == 1:
            conf_mat_tot = conf_mat
        else:
            conf_mat_tot += conf_mat
        # salavataggio della matrice di confusione corrente
        np.save(result_path_model + 'conf_mat_' + str(index) , conf_mat)
        index += 1
        total_accuracy += accuracy_score(y_test, y_pred)
        total_precision += precision_score(y_test, y_pred, average='micro')
        total_recall += recall_score(y_test, y_pred, average='micro')
        total_fscore += f1_score(y_test, y_pred, average='micro')
    # scrittura dei risultati medi di prestazione
    metrics_file.write('Accuracy : ' + str(total_accuracy / n_splits) + '\n')
    metrics_file.write('Precision : ' + str(total_precision / n_splits) + '\n')
    metrics_file.write('Recall : ' + str(total_recall / n_splits) + '\n')
    metrics_file.write('F1-Score : ' + str(total_fscore / n_splits) + '\n')
    metrics_file.close()
    # salvataggio della matrice di confusione media
    conf_mat_avg = conf_mat_tot / n_splits
    np.save(result_path_model + 'conf_mat_avg', conf_mat_avg)
