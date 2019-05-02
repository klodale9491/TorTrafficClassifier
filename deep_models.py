from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten


# creazione del modello 1 : ffnn con sigmoidi
def create_deep_model_1(n_cols, layers=5, neurons=100):
    hidden_layers = layers
    num_neurons = neurons
    model = Sequential()
    # layer fully connected verso l'input
    model.add(Dense(n_cols, input_shape=(n_cols,), activation='sigmoid'))
    for l in range(0, hidden_layers-1):
        model.add(Dense(num_neurons, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# creazione del modello 2 : ffnn con LeakyReLU e sigmoidi
def create_deep_model_2(n_cols, layers=5, neurons=100):
    hidden_layers = layers
    num_neurons = neurons
    model = Sequential()
    # layer fully connected verso l'input
    model.add(Dense(n_cols, input_shape=(n_cols,), activation='sigmoid'))
    for l in range(0, hidden_layers-1):
        model.add(Dense(num_neurons, activation='linear'))
        model.add(LeakyReLU())  # aumento delle prestazioni di precisione,recall e f1-score
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# creazione modello 3 : ffnn con fully connected + leaky relu + batch optimization
def create_deep_model_3(n_cols, classes, layers=5, neurons=100):
    num_neurons = neurons
    model = Sequential()
    # layer fully connected verso l'input
    model.add(Dense(n_cols, input_shape=(n_cols,), activation='linear'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    # inserimento degli strati nascosti
    for i in range(0, layers - 1):
        model.add(Dense(num_neurons, activation='linear'))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
    # layer di softmax alla fine della rete
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# creazione del modello 4 : rete convolutiva 1D
def create_deep_model_4(n_cols, classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(n_cols, 1)))
    model.add(MaxPooling1D(pool_size=3, padding='same', strides=2))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, padding='same', strides=2))
    model.add(Dense(64, activation='linear'))
    model.add(Flatten())  #linearizza lo strato di uscita
    model.add(Dense(625, activation='linear'))
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# creazione del modello 5 : rete convolutiva 1D + Leaky Relu + Batch Optimization
def create_deep_model_5(n_cols, classes, layers):
    model = Sequential()
    # estrazione delle features tramite layers convolutivi
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(n_cols, 1)))
    model.add(MaxPooling1D(pool_size=3, padding='same', strides=2))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=3, padding='same', strides=2))
    model.add(Dense(64, activation='linear'))
    model.add(Flatten())  # linearizza lo strato di uscita
    # classificazione mediante leaky_relu + batch optimization
    for i in range(0, layers - 1):
        model.add(Dense(125, activation='linear'))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model