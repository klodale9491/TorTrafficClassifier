'''
   Genera dei dataset ottenuti tramite estrazione delle caratteristiche
   da parte di un modello convolutivo.
   Esempio di esecuzione :

   python3 gen_ds_conv1d.py -n 2000 -t 100000
   python3 gen_ds_conv1d.py -n 2000 -t 100000
'''
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from util import load_merged_nibble_datasets
import pandas as pd
import numpy as np
import sys
import os

# parametri per lo script
n_features = 108
n_components = 27


# nel caso di riga di comando, più facile per listare jobs
args = sys.argv
try:
    n_samples_gen = int(args[args.index('-n') + 1])
    n_samples_trn = int(args[args.index('-t') + 1])
# nel cado harcoding dei parametri
except ValueError:
    n_samples_gen = 2000
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
with open('model_' + str(n_samples_trn) + '.json', 'w') as model_file:
    model_file.write(model_json)
model_file.close()


# creazione della cartelle
if os.path.isdir('datasets/conv1d_pca/'+str(n_samples_gen)+'_samples/') is False:
    os.mkdir('datasets/conv1d_pca/'+str(n_samples_gen)+'_samples/')
if os.path.isdir('datasets/conv1d_pca/'+str(n_samples_gen)+'_samples/binary') is False:
    os.mkdir('datasets/conv1d_pca/'+str(n_samples_gen)+'_samples/binary')
if os.path.isdir('datasets/conv1d_pca/'+str(n_samples_gen)+'_samples/label') is False:
    os.mkdir('datasets/conv1d_pca/'+str(n_samples_gen)+'_samples/label')


base_read_path = 'datasets/raw/mixed/'+str(n_samples_trn)+'_samples/'
base_write_path = 'datasets/conv1d_pca/'+str(n_samples_gen)+'_samples/'
# caricamento dei datasets e tipizzazione della struttura dati più opportuna
col_dtypes = {}
for i in range(0, 108):
    col_dtypes[i] = np.uint8
col_dtypes[108] = str
for ds in datasets:
    print('extracting features from ' + ds + '....')
    df_bin = pd.read_csv(base_read_path + 'binary/' + ds + '.csv', sep=',', header=None, dtype=col_dtypes)
    df_label = pd.read_csv(base_read_path + 'label/' + ds + '.csv', sep=',', header=None, dtype=col_dtypes)
    data_bin = np.asarray(df_bin[df_bin.columns[0:-1]])
    data_bin = data_bin.reshape(data_bin.shape[0], data_bin.shape[1], 1)
    data_label = np.asarray(df_label[df_label.columns[0:-1]])
    data_label = data_label.reshape(data_label.shape[0], data_label.shape[1], 1)
    classes_bin = np.asarray(df_bin[108])
    classes_label = np.asarray(df_label[108])
    features_bin = intermediate_layer_model.predict(data_bin)
    features_label = intermediate_layer_model.predict(data_label)
    # estrazione delle caratteristiche principali
    print('extracting principal features via pca ...')
    pca_bin = PCA(n_components=n_components)
    pca_label = PCA(n_components=n_components)
    pca_bin.fit(features_bin)
    pca_label.fit(features_label)
    features_bin_trx = pca_bin.transform(features_bin)
    features_label_trx = pca_label.transform(features_label)
    # salvataggio delle features estratte
    print('writing principal features of ' + ds + '....')
    pd.DataFrame(np.column_stack((features_bin_trx, classes_bin))).to_csv(base_write_path + 'binary/' + ds + '.csv', header=None, index=False)
    pd.DataFrame(np.column_stack((features_label_trx, classes_label))).to_csv(base_write_path + 'label/' + ds + '.csv', header=None, index=False)