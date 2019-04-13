from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from util import load_dataset_time_based_a
from util import load_dataset_csv_a
from util import oversample_dataset_a
import numpy as np

'''
# caricamento datasets 1
X_train, Y_train = load_dataset_time_based_a('../datasets/scenario-a/TimeBasedFeatures-30s-TOR-NonTOR-85.arff')
X_test, Y_test = load_dataset_time_based_a('../datasets/scenario-a/TimeBasedFeatures-30s-TOR-NonTOR-15.arff')
X_train, Y_train = oversample_dataset_a(X_train, Y_train)
'''

# caricamento dataset 2
X, Y = load_dataset_time_based_a('../datasets/scenario-a/TimeBasedFeatures-30s-TOR-NonTOR.arff')


# kfold cross validation
n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
total_accuracy = 0.0
total_precision = 0.0
total_recall = 0.0
total_fscore = 0.0


for train_index, test_index in kfold.split(X, Y):
    # oversampling del train
    X_train, Y_train = oversample_dataset_a(X[train_index], Y[train_index])
    X_test = X[test_index]
    Y_test = Y[test_index]

    # crea il modello
    model = Sequential()

    # numero di attributi del training-set
    n_cols = X_train.shape[1]

    # hidden layers
    hidden_layers = 5

    # numero di neuroni del modello
    num_neurons = 100

    # aggiungo i vari layer della rete depp
    model.add(Dense(n_cols, input_shape=(n_cols,), activation='sigmoid'))
    for l in range(0, hidden_layers-1):
        model.add(Dense(num_neurons, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # compila il modello senttando # ottimizzatore, e funzione di errore
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # imposta early stopping in modo tale da fermarsi dopo che il modello non migliora più
    early_stopping_monitor = EarlyStopping(patience=3)

    # alleno il modello
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30,
              callbacks=[early_stopping_monitor], verbose=1)

    # predizione dei dati di test
    Y_pred = model.predict(X_test)
    Y_pred_vec = np.asarray([1 if y[0] > 0.5 else 0 for y in Y_pred])

    # calcolo degli errori
    conf_mat = confusion_matrix(Y_test, Y_pred_vec)
    total_accuracy += accuracy_score(Y_test, Y_pred_vec)
    total_precision += precision_score(Y_test, Y_pred_vec)
    total_recall += recall_score(Y_test, Y_pred_vec)
    total_fscore += f1_score(Y_test, Y_pred_vec)



# risultati
print(['Accuracy 10-fold : ', total_accuracy / n_splits])
print(['Precision 10-fold : ', total_precision / n_splits])
print(['Recall 10-fold : ', total_recall / n_splits])
print(['F-score 10-fold : ', total_fscore / n_splits])
