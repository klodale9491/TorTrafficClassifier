'''
   Genera dei dataset ottenuti tramite estrazione delle caratteristiche
   da parte di un modello convolutivo.
   Esempio di esecuzione :

   python3 save_model.py -t 100000
'''
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from util import load_merged_nibble_datasets
import sys

# parametri per lo script
n_features = 108

# nel caso di riga di comando, più facile per listare jobs
args = sys.argv
try:
    n_samples_trn = int(args[args.index('-t') + 1])
# nel cado harcoding dei parametri
except ValueError:
    n_samples_trn = 1000


print('loading datasets ...')
# Caricamento datasets
datasets = [
    'AUDIO-STREAMING',
    'BROWSING',
    'EMAIL',
    'CHAT',
    'FILE-TRANSFER',
    'P2P',
    'VIDEO-STREAMING',
    'VOIP'
]
x, y = load_merged_nibble_datasets(datasets, samples=n_samples_trn, classification='label')
y = to_categorical(y)
x = x.reshape(x.shape[0], x.shape[1], 1)


# creazione del modello convolutivo per l'estrazione delle features
print('creation of model ...')
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(n_features, 1)))
model.add(MaxPooling1D(pool_size=3, padding='same', strides=2))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=3, padding='same', strides=2))
model.add(Dense(64, activation='linear'))
model.add(Flatten())
model.add(Dense(625, activation='linear'))
model.add(Dense(16, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# output del layer di mezzo
layer_name = 'dense_2' # layer di estrazione della caratteristiche della rete convolutiva
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


# training del modello convolutivo
print('training model with' + str(n_samples_trn) + '....')
early_stopping_monitor = EarlyStopping(patience=3)
model.fit(x, y, epochs=30, callbacks=[early_stopping_monitor], verbose=1)
# salvataggio del modello
print('saving model to backup...')
model_json = model.to_json() # salva il modello allenato in json
with open('model_bkp/model_' + str(n_samples_trn) + '.json', 'w') as model_file:
    model_file.write(model_json)
model_file.close()
