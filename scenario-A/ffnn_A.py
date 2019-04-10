from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from util import load_dataset_time_based_a
from util import load_dataset_csv_a
from util import oversample_dataset_a
import numpy as np


# caricamento datasets
# X, Y = load_dataset_csv_a('../datasets/scenario-a/merged_5s_clean.csv')
X, Y = load_dataset_time_based_a('../datasets/scenario-a/TimeBasedFeatures-30s-TOR-NonTOR.arff')

# oversampling: bilancio il dataset in favore della classe con meno samples ..
X, Y = oversample_dataset_a(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

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
early_stopping_monitor = EarlyStopping(patience=5)

# alleno il modello
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, callbacks=[early_stopping_monitor])

# predizione dei dati di test
Y_pred = model.predict(X_test)
Y_pred_vec = np.asarray([1 if y[0] > 0.5 else 0 for y in Y_pred])

# calcolo degli errori
conf_mat = confusion_matrix(Y_test, Y_pred_vec)
accuracy = accuracy_score(Y_test, Y_pred_vec)
class_rep = classification_report(Y_test, Y_pred_vec)

# risultati
print('confusion_matrix : ')
print(conf_mat)
print('accuracy : ')
print(accuracy)
print(class_rep)
