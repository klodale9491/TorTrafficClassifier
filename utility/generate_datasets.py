import os
import pandas as pd
import numpy as np

# genera dei dataset di un numero arbitrario di campioni per ugnuno dei protocolli
# in seguito genera 2 cartelle con i dataset etichetti in modo binario e per ogni
# protocollo
def generate_samples_datasets(datasets, replace_labels, n_samples=100000):
    #genero le cartelle necessario
    base_dir = '../datasets/raw/mixed/'
    sample_fold_name = str(n_samples)+'_samples'
    os.mkdir(base_dir + sample_fold_name)
    os.mkdir(base_dir + sample_fold_name + '/binary')
    os.mkdir(base_dir + sample_fold_name + '/label')
    # impostazione della memoria da allocare per i campi
    col_dtypes = {}
    for i in range(0, 108):
        col_dtypes[i] = np.uint8
    col_dtypes[108] = str
    # per ognuno dei datasets faccio shuffling e salvo il campione
    for i in range(len(datasets)):
        ds = datasets[i]
        csv_file = ds + '.csv'
        print('shuffling ' + ds + '...')
        os.system("shuf -n " + str(n_samples) + ' ' + base_dir + csv_file + ' > ' + base_dir + sample_fold_name + '/binary/' + csv_file)
        # rimpiazzo ora l'etichetta tor/non_tor con una numerica
        print('replacing labels ' + ds + '...')
        df = pd.read_csv(base_dir + sample_fold_name + '/binary/' + csv_file, sep=",", header=None, dtype=col_dtypes)
        df = df.replace('nonTor', replace_labels[i][0])
        df = df.replace('tor', replace_labels[i][1])
        df.to_csv(base_dir + sample_fold_name + '/label/' + csv_file, header=None, index=False)
    print('all done!')



datasets = [
    'AUDIO-STREAMING',
    'BROWSING',
    'CHAT',
    'EMAIL',
    'FILE-TRANSFER',
    'P2P',
    'VIDEO-STREAMING',
    'VOIP'
]
replace_labels = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [12, 13],
    [14, 15]
]
generate_samples_datasets(datasets, replace_labels, 100000)
