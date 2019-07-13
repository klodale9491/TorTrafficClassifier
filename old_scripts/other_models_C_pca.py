from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import IncrementalPCA
import numpy as np
import pandas as pd
import sys



'''
Esempio di esecuzione : python3 other_models_C_OLD.py -m bayes -n 50000 -c label -d 20

args = sys.argv
try:
    model_type = args[args.index('-m') + 1]
    n_samples = int(args[args.index('-n') + 1])
    classification_type = args[args.index('-c') + 1]
    splits = args[args.index('-s') + 1]
    dims = args[args.index('-d') + 1]
# nel cado harcoding dei parametri
except ValueError:
    model_type = 'bayes'
    n_samples = 1000
    classification_type = 'label'
    splits = 10
    dims = 20
'''
model_type = 'svm'
n_samples = 1000
classification_type = 'label'
dims = 20

print('Executing PCA into dataset for ' + str(dims) + ' dimensions ...')
# esecuzione di pca sul dataset
base_path = 'datasets/raw/mixed_conv1d/' + str(n_samples) + '_samples/' + classification_type + '/'
df = pd.read_csv(base_path + 'ALL.csv', header=None, chunksize=10000)
ipca = IncrementalPCA(n_components=dims)
n_cols = 1728
for chunk in df:
    chunk_data = np.asarray(chunk[chunk.columns[0:-1]])
    chunk_classes = np.asarray(chunk[n_cols])
    ipca.partial_fit(chunk_data, chunk_classes)
df = pd.read_csv(base_path + 'ALL.csv', header=None, chunksize=10000)
pca_data = np.asarray([])
for chunk in df:
    chunk_data = np.asarray(chunk[chunk.columns[0:-1]])
    chunk_classes = np.asarray(chunk[n_cols])
    chunk_classes = chunk_classes.reshape(chunk_classes.shape[0], 1)
    pca_chunk_data = ipca.transform(chunk_data)
    pca_chunk_data = np.append(pca_chunk_data, chunk_classes, axis=1)
    if pca_data.size == 0:
        pca_data = pca_chunk_data
    else:
        pca_data = np.concatenate((pca_data, pca_chunk_data))
print('Done!')


print('Executing classification on PCA reducted data ....')
# esecuzione del modello di classificazione su dataset ridotto
if model_type == 'bayes':
    model = GaussianNB()
elif model_type == 'log_reg':
    model = LogisticRegression(solver='liblinear')
elif model_type == 'rnd_for':
    model = RandomForestClassifier(n_jobs=2, random_state=0)
elif model_type == 'svm':
    model = svm.SVC(kernel='rbf', C=1)
scoring = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro']
scores = cross_validate(model, pca_data[:, range(dims)], pca_data[:, dims], cv=10, scoring=scoring)
print('Done!')


print('Saving results data ...')
# salvataggio dei risultati
filename = model_type + '_' + str(n_samples) + '_' + classification_type + '.conv1d_pca_' + str(dims)
with open('results/other_C/' + filename, mode='w+') as file:
    file.write('Accuracy : ' + str(np.mean(scores['test_accuracy'])) + '\n')
    file.write('Precision : ' + str(np.mean(scores['test_precision_micro'])) + '\n')
    file.write('Recall : ' + str(np.mean(scores['test_recall_micro'])) + '\n')
    file.write('F1 : ' + str(np.mean(scores['test_f1_micro'])) + '\n')
file.close()
print('All done!')