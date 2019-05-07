import os
import sys
import pandas as pd
import numpy as np


'''
Genera dei dataset di un numero arbitrario di campioni per ugnuno dei protocolli
in seguito genera 2 cartelle con i dataset etichetti in modo binario e per ogni
protocollo.
'''
def generate_samples_datasets(datasets, replace_labels, n_samples=100000):
    #genero le cartelle necessario
    base_dir = 'datasets/raw/mixed/'
    sample_fold_name = str(n_samples)+'_samples'
    os.mkdir(base_dir + sample_fold_name)
    os.mkdir(base_dir + sample_fold_name + '/tmp')
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
        os.system("shuf -n " + str(n_samples) + ' ' + base_dir + csv_file + ' > ' + base_dir + sample_fold_name + '/tmp/' + csv_file)
        # rimpiazzo ora l'etichetta tor/non_tor con una numerica
        print('replacing labels ' + ds + '...')
        df_bin = pd.read_csv(base_dir + sample_fold_name + '/tmp/' + csv_file, sep=",", header=None, dtype=col_dtypes)
        df_multi = pd.read_csv(base_dir + sample_fold_name + '/tmp/' + csv_file, sep=",", header=None, dtype=col_dtypes)
        df_bin = df_bin.replace('nonTor', 0)
        df_bin = df_bin.replace('tor', 1)
        df_multi = df_multi.replace('nonTor', replace_labels[i][0])
        df_multi = df_multi.replace('tor', replace_labels[i][1])
        df_bin.to_csv(base_dir + sample_fold_name + '/binary/' + csv_file, header=None, index=False)
        df_multi.to_csv(base_dir + sample_fold_name + '/label/' + csv_file, header=None, index=False)
    os.system("rm -R " + base_dir + sample_fold_name + '/tmp/')
    print('all done!')



'''
Pari    : 0,2,4,6,8,.... -> NON TOR
Dispari : 1,3,5,7,9,.... -> TOR
'''
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


'''
Esempio di esecuzione : python3 generate_dataset.py -n 1000
'''
args = sys.argv
# nel caso di riga di comando, più facile per listare jobs
try:
    n_samples = int(args[args.index('-n') + 1])
# nel cado harcoding dei parametri
except ValueError:
    n_samples = 2000
print('Running script with params : ')
print('number_of_samples  : ' + str(n_samples))


if n_samples > 0:
    generate_samples_datasets(datasets, replace_labels, n_samples)
else:
    print('Errore nella generazione del dataset ....')
