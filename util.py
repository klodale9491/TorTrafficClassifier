from scipy.io import arff
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np


# funzioni di caricamento dei vari dataset, in base allo scenario
def load_dataset_time_based_a(filename):
    arffData = arff.loadarff(filename)
    cleanData = arffData[0].__array__()
    cleanDataList = [list(row) for row in cleanData]
    for row in cleanDataList:
        row[-1] = row[-1].decode('utf-8')
    data = np.asarray([cleanDataList[i][0:22] for i in range(len(cleanDataList))])
    classes = np.asarray([cleanDataList[i][23] for i in range(len(cleanDataList))])
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(['TOR', 'NONTOR'])
    encoded_classes = label_encoder.transform(classes)
    return data, encoded_classes


def load_dataset_time_based_b(filename):
    arffData = arff.loadarff(filename)
    cleanData = arffData[0].__array__()
    cleanDataList = [list(row) for row in cleanData]
    for row in cleanDataList:
        row[-1] = row[-1].decode('utf-8')
    classes = np.asarray([cleanDataList[i][23] for i in range(len(cleanDataList))])
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(['AUDIO-STREAMING', 'BROWSING', 'CHAT', 'FILE-TRANSFER', 'EMAIL', 'P2P', 'VOIP', 'VIDEO-STREAMING'])
    encoded_classes = label_encoder.transform(classes)
    dataset = np.asarray([cleanDataList[i][0:22] for i in range(len(cleanDataList))])
    return dataset, encoded_classes


def load_dataset_csv_a(filename):
    dataframe = pd.read_csv(filename, ',')
    columns = dataframe.columns[5:-1]
    data = np.asarray(dataframe[columns])
    classes = np.asarray(dataframe['label'])
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(['TOR', 'nonTOR'])
    encoded_classes = label_encoder.transform(classes)
    return data, encoded_classes


def load_dataset_csv_b(filename):
    dataframe = pd.read_csv(filename, ',')
    columns = dataframe.columns[5:-1]
    data = np.asarray(dataframe[columns])
    classes = np.asarray(dataframe['label'])
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(['AUDIO', 'BROWSING', 'CHAT', 'FILE-TRANSFER', 'MAIL', 'P2P', 'VOIP', 'VIDEO'])
    encoded_classes = label_encoder.transform(classes)
    return data, encoded_classes


# funzione che effettua UNDERSAMPLIG su entrambi i dataset
def undersample_dataset(x, y):
    rus = RandomUnderSampler(random_state=0, replacement=True)
    return rus.fit_resample(x, y)


# funzione che effettua OVERSAMPLIG su entrambi i dataset
def oversample_dataset(x, y):
    ros = RandomOverSampler(random_state=0)
    return ros.fit_resample(x, y)

