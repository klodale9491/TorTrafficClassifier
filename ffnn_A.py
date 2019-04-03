from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from util import load_dataset_time_based_a


# X, Y = loadDataset('datasets/scenario-a/SelectedFeatures-15s-TOR-NonTOR.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-15s-TOR-NonTOR.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-15s-TOR-NonTOR-15.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-15s-TOR-NonTOR-85.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-30s-TORNonTOR.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-30s-TORNonTOR-15.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-30s-TORNonTOR-85.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-60s-TOR-NonTOR.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-60s-TOR-NonTOR-15.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-60s-TOR-NonTOR-85.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-120s-TOR-NonTOR.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-120s-TOR-NonTOR-15.arff')
X, Y = load_dataset_time_based_a('datasets/scenario-a/TimeBasedFeatures-120s-TOR-NonTOR-85.arff')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
Y_train_cat = to_categorical(Y_train)
Y_test_cat = to_categorical(Y_test)

# crea il modello
model = Sequential()

# numero di attributi del training-set
n_cols = X_train.shape[1]

# hidden layers
hidden_layers = 2

# aggiungo i vari layer della rete depp
model.add(Dense(250, activation='sigmoid', input_shape=(n_cols,)))
for l in range(hidden_layers):
    model.add(Dense(250, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))

# compila il modello senttando :
# ottimizzatore, e funzione di errore
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# imposta early stopping in modo tale da fermarsi dopo che il modello non migliora più
early_stopping_monitor = EarlyStopping(patience=4)

# alleno il modello
model.fit(X_train, Y_train_cat, validation_split=0.2, epochs=100, callbacks=[early_stopping_monitor])

# predizione dei dati di test
Y_pred_cat = model.predict(X_test)
Y_pred = Y_pred_cat.argmax(1)

# calcolo degli errori
conf_mat = confusion_matrix(Y_test, Y_pred)
class_rep = classification_report(Y_test, Y_pred)

print(conf_mat)
print(class_rep)
