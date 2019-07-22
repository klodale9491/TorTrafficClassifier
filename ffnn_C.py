'''
Esempio di esecuzione : python3 ffnn_C.py -n 1000 -c multi
'''
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from util import load_merged_nibble_datasets
from deep_models import create_deep_model_5
import numpy as np
import sys
import os


args = sys.argv
# nel caso di riga di comando, più facile per listare jobs
try:
    n_samples = int(args[args.index('-n') + 1])
    classification_type = args[args.index('-c') + 1]
# nel cado harcoding dei parametri
except ValueError:
    n_samples = 1000
    classification_type = 'multi'
print('Running script with params : ')
print('number_of_samples  : ' + str(n_samples))
print('classification_type  : ' + str(classification_type))


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
x, y = load_merged_nibble_datasets(datasets, samples=n_samples, classification='multi')

# kfold cross validation
n_splits = 10
kfold = StratifiedShuffleSplit(n_splits=n_splits)

# numero di neuroni e layer da utilizzare per testare i modelli
num_layers = [5, 10, 20, 25]
num_neurons = [125, 62, 31, 25]

# numero di classi e tipologia di queste
num_classes = 16
class_categorical = True

for i in range(len(num_layers)):
    # parametri di valutazione delle prestazioni dell'algoritmo
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_fscore = 0.0

    # cartella dei risultati per l'esperimento, metriche e matrici di confusione
    result_path = 'results/ffnn_C/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_path_samples = result_path + str(n_samples) + '/'
    if not os.path.exists(result_path_samples):
        os.mkdir(result_path_samples)
    result_path_model = result_path_samples + str(num_layers[i]) + '/'
    if not os.path.exists(result_path_model):
        os.mkdir(result_path_model)
    metrics_file = open(result_path_model + 'metrics.txt', 'w+')

    index = 1
    conf_mat_tot = None
    for train_index, test_index in kfold.split(x, y):
        x_train, y_train = x[train_index], y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        n_cols = x_train.shape[1]

        # reshaping per la rete convolutiva
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        # creazione ed allenamento del modello
        model = create_deep_model_5(n_cols=x_train.shape[1], classes=16, layers=num_layers[i])
        early_stopping_monitor = EarlyStopping(patience=3)
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, callbacks=[early_stopping_monitor], verbose=1)
        accuracy = model.evaluate(x_test, y_test, verbose=0)

        # predizione dei dati di test
        y_pred = model.predict(x_test)
        if class_categorical:
            y_pred_vec = y_pred.argmax(1)
            y_test = y_test.argmax(1)
        else:
            y_pred_vec = np.asarray([1 if y[0] > 0.5 else 0 for y in y_pred])
        conf_mat = confusion_matrix(y_test, y_pred_vec)
        if index == 1:
            conf_mat_tot = conf_mat
        else:
            conf_mat_tot += conf_mat
        np.save(result_path_model + 'conf_mat_' + str(index), conf_mat)
        index += 1
        total_accuracy += accuracy_score(y_test, y_pred_vec)
        total_precision += precision_score(y_test, y_pred_vec, average='micro')
        total_recall += recall_score(y_test, y_pred_vec, average='micro')
        total_fscore += f1_score(y_test, y_pred_vec, average='micro')
    # scrittura dei risultati medi di prestazione
    metrics_file.write('Accuracy : ' + str(total_accuracy / n_splits) + '\n')
    metrics_file.write('Precision : ' + str(total_precision / n_splits) + '\n')
    metrics_file.write('Recall : ' + str(total_recall / n_splits) + '\n')
    metrics_file.write('F1-Score : ' + str(total_fscore / n_splits) + '\n')
    metrics_file.close()
    # salvataggio della matrice di confusione media
    conf_mat_avg = conf_mat_tot / n_splits
    np.save(result_path_model + 'conf_mat_avg', conf_mat_avg)
