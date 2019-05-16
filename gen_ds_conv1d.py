'''
   Genera dei dataset ottenuti tramite estrazione delle caratteristiche
   da parte di un modello convolutivo, da un dataset campione già creato
   Esempio di esecuzione :

   python3 gen_ds_conv1d.py -n 10000
'''
from keras.models import Model
from keras.models import model_from_json
import pandas as pd
import numpy as np
import sys
import os



# nel caso di riga di comando, più facile per listare jobs
args = sys.argv
try:
    n_samples_gen = int(args[args.index('-n') + 1])
# nel cado harcoding dei parametri
except ValueError:
    n_samples_gen = 1000



print('Running script with these params : ')
print('n_samples_gen : ' + str(n_samples_gen))


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


print('loading model from bkp ...')
model = model_from_json(open('model_bkp/model_1000000.json').read())
model.load_weights('model_bkp/model_1000000.h5')
# output del layer di mezzo
layer_name = 'flatten_1' # layer di estrazione della caratteristiche della rete convolutiva
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


# cartella 1 - cartella col numero di esempi
if os.path.isdir('datasets/raw/mixed_conv1d/'+str(n_samples_gen)+'_samples/') is False:
    os.mkdir('datasets/raw/mixed_conv1d/'+str(n_samples_gen)+'_samples/')

# cartella 2 - tipologia di calssificazione binaria
if os.path.isdir('datasets/raw/mixed_conv1d/'+str(n_samples_gen)+'_samples/binary') is False:
    os.mkdir('datasets/raw/mixed_conv1d/'+str(n_samples_gen)+'_samples/binary')

# cartella 3 - tipologia di classificazione multipla
if os.path.isdir('datasets/raw/mixed_conv1d/'+str(n_samples_gen)+'_samples/label') is False:
    os.mkdir('datasets/raw/mixed_conv1d/'+str(n_samples_gen)+'_samples/label')


base_read_path  = 'datasets/raw/mixed/' + str(n_samples_gen) + '_samples/'
base_write_path = 'datasets/raw/mixed_conv1d/' + str(n_samples_gen) + '_samples/'

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

    # estrazione delle caratteristiche
    features_bin_trx = intermediate_layer_model.predict(data_bin)
    features_label_trx = intermediate_layer_model.predict(data_label)

    # salvataggio delle features estratte
    print('writing principal features of ' + ds + '....')
    print(base_write_path + 'binary/' + ds + '.csv')
    pd.DataFrame(np.column_stack((features_bin_trx, classes_bin))).to_csv(base_write_path + 'binary/' + ds + '.csv', header=None, index=False)
    print(base_write_path + 'label/' + ds + '.csv')
    pd.DataFrame(np.column_stack((features_label_trx, classes_label))).to_csv(base_write_path + 'label/' + ds + '.csv', header=None, index=False)