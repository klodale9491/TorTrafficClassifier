from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from util import load_dataset_time_based_b
from util import undersample_dataset
from util import oversample_dataset
from deep_models import *
import numpy as np



# caricamento dataset
X, Y = load_dataset_time_based_b('datasets/time_based/scenario-b/TimeBasedFeatures-30s-Layer2.arff')

# kfold cross validation
n_splits = 10
kfold = StratifiedShuffleSplit(n_splits=n_splits)

# parametri di valutazione delle prestazioni dell'algoritmo
total_accuracy = 0.0
total_precision = 0.0
total_recall = 0.0
total_fscore = 0.0

# numero di classi e tipologia di queste
class_categorical = True
num_classes = 8

for train_index, test_index in kfold.split(X, Y):
    # splitting training/test set + oversampling delle classi in quantità minore
    X_train, Y_train = X[train_index], Y[train_index]
    # X_train, Y_train = undersample_dataset(X[train_index], Y[train_index])
    X_train, Y_train = oversample_dataset(X[train_index], Y[train_index])
    X_test = X[test_index]
    Y_test = Y[test_index]
    n_cols = X_train.shape[1]

    # trasformazione delle classi in forma di categorie
    if class_categorical:
        Y_train = to_categorical(Y_train)
        Y_test = to_categorical(Y_test)

    # creazione ed allenamento del modello
    model = create_deep_model_3(n_cols, num_classes, 20, 25)
    early_stopping_monitor = EarlyStopping(patience=3)
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30,
              callbacks=[early_stopping_monitor], verbose=1)


    # predizione dei dati di test
    Y_pred = model.predict(X_test)
    if class_categorical:
        Y_pred_vec = Y_pred.argmax(1)
        Y_test = Y_test.argmax(1)
    else:
        Y_pred_vec = np.asarray([1 if y[0] > 0.5 else 0 for y in Y_pred])

    # calcolo degli indici di prestazione dell'algoritmo
    conf_mat = confusion_matrix(Y_test, Y_pred_vec)
    print(conf_mat)
    total_accuracy += accuracy_score(Y_test, Y_pred_vec)
    print(accuracy_score(Y_test, Y_pred_vec))
    total_precision += precision_score(Y_test, Y_pred_vec, average='micro')
    total_recall += recall_score(Y_test, Y_pred_vec, average='micro')
    total_fscore += f1_score(Y_test, Y_pred_vec, average='micro')


# risultati
print(['Accuracy 10-fold : ', total_accuracy / n_splits])
print(['Precision 10-fold : ', total_precision / n_splits])
print(['Recall 10-fold : ', total_recall / n_splits])
print(['F-score 10-fold : ', total_fscore / n_splits])
