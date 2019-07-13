from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from util import load_dataset_time_based_a
from util import oversample_dataset
from deep_models import *
import numpy as np
import csv
import sys


# caricamento dei file di input e output
file_data = open(sys.argv[1], 'r')
file_results = open(sys.argv[1] + '.res', 'w+')
csv_writer = csv.writer(file_results, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
csv_writer.writerow(['LAYERS', 'NEURONS', 'ACCURACY', 'PRECISION', 'RECALL', 'F1-SCORE'])

# caricamento del dataset
X, Y = load_dataset_time_based_a(file_data)

# kfold cross validation
n_splits = 10
kfold = StratifiedShuffleSplit(n_splits=n_splits)

# numero di neuroni e layer da utilizzare per testare i modelli
num_layers = [2, 3, 4, 5, 6, 7, 8, 9, 10]
num_neurons = [250, 167, 125, 100, 83, 72, 63, 56, 50]


for i in range(len(num_layers)):

    # parametri di valutazione delle prestazioni dell'algoritmo
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_fscore = 0.0

    # numero di classi e tipologia di queste
    num_classes = 2
    class_categorical = True

    # creazione del modello
    model = create_deep_model_3(n_cols, num_classes, num_layers[i], num_neurons[i])
    early_stopping_monitor = EarlyStopping(patience=3)

    for train_index, test_index in kfold.split(X, Y):
        X_train, Y_train = oversample_dataset(X[train_index], Y[train_index])
        X_train, Y_train = X[train_index], Y[train_index]
        X_test = X[test_index]
        Y_test = Y[test_index]
        n_cols = X_train.shape[1]

        # trasformazione delle classi in forma di categorie
        if class_categorical:
            Y_train = to_categorical(Y_train)
            Y_test = to_categorical(Y_test)

        # allenamento del modello
        model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, callbacks=[early_stopping_monitor], verbose=1)

        # predizione dei dati di test
        Y_pred = model.predict(X_test)
        if class_categorical:
            Y_pred_vec = Y_pred.argmax(1)
            Y_test = Y_test.argmax(1)
        else:
            Y_pred_vec = np.asarray([1 if y[0] > 0.5 else 0 for y in Y_pred])

        # calcolo degli indici di prestazione dell'algoritmo
        conf_mat = confusion_matrix(Y_test, Y_pred_vec)
        total_accuracy += accuracy_score(Y_test, Y_pred_vec)
        total_precision += precision_score(Y_test, Y_pred_vec)
        total_recall += recall_score(Y_test, Y_pred_vec)
        total_fscore += f1_score(Y_test, Y_pred_vec)

    # risultati
    csv_writer.writerow([num_layers[i], num_neurons[i], total_accuracy / n_splits, total_precision / n_splits, total_recall / n_splits, total_fscore / n_splits])

# fine del programma e scrittura risultati
file_results.close()