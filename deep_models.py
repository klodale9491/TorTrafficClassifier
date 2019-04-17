from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization


# creazione del modello 1 : ffnn con sigmoidi
def create_deep_model_1(n_cols):
    hidden_layers = 5
    num_neurons = 100
    model = Sequential()
    # layer fully connected verso l'input
    model.add(Dense(n_cols, input_shape=(n_cols,), activation='sigmoid'))
    for l in range(0, hidden_layers-1):
        model.add(Dense(num_neurons, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# creazione del modello 2 : ffnn con LeakyReLU e sigmoidi
def create_deep_model_2(n_cols):
    hidden_layers = 5
    num_neurons = 100
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
def create_deep_model_3(n_cols, classes):
    hidden_layers = 10
    num_neurons = 50
    model = Sequential()
    # layer fully connected verso l'input
    model.add(Dense(n_cols, input_shape=(n_cols,), activation='linear'))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    # inserimento degli strati nascosti
    for i in range(0, hidden_layers - 1):
        model.add(Dense(num_neurons, activation='linear'))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
    # layer di softmax alla fine della rete
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
