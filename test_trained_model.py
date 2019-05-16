'''
Test che verifica la validità del modello salvato
allenato.
'''

from keras.models import model_from_json
from keras.utils import to_categorical
from util import load_merged_nibble_datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sys


# impostazione dei parametri dello script
args = sys.argv
try:
    n_samples_tst = int(args[args.index('-n') + 1])
except ValueError:
    n_samples_tst = 1000000


# Caricamento datasets
print('loading datasets ...')
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

x, y = load_merged_nibble_datasets(datasets, samples=n_samples_tst, classification='label')
y = to_categorical(y)
x = x.reshape(x.shape[0], x.shape[1], 1)

# caricamento del modello allenato
print('loading model from bkp ...')
model = model_from_json(open('model_bkp/model_1000000.json').read())
model.load_weights('model_bkp/model_1000000.h5')

# predizione dei dati di test
print('prediction of samples ...')
y_pred = model.predict(x)
y_pred_vec = y_pred.argmax(1)
y_test = y.argmax(1)

# calcolo degli indici di prestazione dell'algoritmo
conf_mat = confusion_matrix(y_test, y_pred_vec)
accuracy = accuracy_score(y_test, y_pred_vec)
precision = precision_score(y_test, y_pred_vec, average='micro')
recall = recall_score(y_test, y_pred_vec, average='micro')
f1score = f1_score(y_test, y_pred_vec, average='micro')


# stampa dei risultati
print(conf_mat)
print('Accuracy : ' + str(accuracy))
print('Precision : ' + str(precision))
print('Recall : ' + str(recall))
print('F1-Score : ' + str(f1score))