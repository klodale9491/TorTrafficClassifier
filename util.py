from scipy.io import arff
from sklearn import preprocessing
import pandas as pd
import numpy as np


def load_dataset_time_based_a(filename):
    # caricamento dei dati da dataset
    arffData = arff.loadarff(filename)
    cleanData = arffData[0].__array__()
    # manipolo il dataset convertendolo in liste
    cleanDataList = [list(row) for row in cleanData]
    # la classi le voglio in utf-8 e non binarie
    for row in cleanDataList:
        row[-1] = row[-1].decode('utf-8')
    dataset = np.asarray([cleanDataList[i][0:22] for i in range(len(cleanDataList))])
    classes = np.asarray([1 if cleanDataList[i][23] == 'TOR' else 0 for i in range(len(cleanDataList))])
    return dataset, classes


def load_dataset_time_based_b(filename):
    # caricamento dei dati da dataset
    arffData = arff.loadarff(filename)
    cleanData = arffData[0].__array__()
    # manipolo il dataset convertendolo in liste
    cleanDataList = [list(row) for row in cleanData]
    # la classi le voglio in utf-8 e non binarie
    for row in cleanDataList:
        row[-1] = row[-1].decode('utf-8')
    classes = []
    for row in cleanDataList:
        if (row[23] == 'AUDIO-STREAMING'):
            classes.append(0)
        elif (row[23] == 'BROWSING'):
            classes.append(1)
        elif (row[23] == 'CHAT'):
            classes.append(2)
        elif (row[23] == 'FILE-TRANSFER'):
            classes.append(3)
        elif (row[23] == 'EMAIL'):
            classes.append(4)
        elif (row[23] == 'P2P'):
            classes.append(5)
        elif (row[23] == 'VIDEO'):
            classes.append(6)
        elif (row[23] == 'VOIP'):
            classes.append(7)
        elif (row[23] == 'VIDEO-STREAMING'):
            classes.append(8)
    dataset = np.asarray([cleanDataList[i][0:22] for i in range(len(cleanDataList))])
    classes = np.asarray(classes)
    return dataset, classes



def load_dataset_csv_a(filename):
    dataframe = pd.read_csv(filename, ',')
    columns = dataframe.columns[5:-1]
    data = np.asarray(dataframe[columns])
    classes = np.asarray(dataframe['label'])
    # codifica delle label da stringhe a interi
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(['TOR', 'nonTOR'])
    encoded_classes = label_encoder.transform(classes)
    return data, encoded_classes


'''
Permette il caricamento del dataset csv filtrato 
in base al protocollo che ci interessa : Email, FTP, P2P etc ....
'''
def load_dataset_csv_b(filename, protocol):
    dataframe = pd.read_csv(filename, ',')
    columns = dataframe.columns[5:-1]
    data = np.asarray(dataframe[columns])
    if protocol != -1:
        protocol_data = data[data[:, 0] == protocol]
    else:
        protocol_data = data
    protocol_classes = np.asarray(dataframe['label'])
    protocol_classes_num = []
    for prot in protocol_classes:
        if(prot == 'AUDIO'):
            protocol_classes_num.append(0)
        elif(prot == 'BROWSING'):
            protocol_classes_num.append(1)
        elif (prot == 'CHAT'):
            protocol_classes_num.append(2)
        elif (prot == 'FILE-TRANSFER'):
            protocol_classes_num.append(3)
        elif (prot == 'MAIL'):
            protocol_classes_num.append(4)
        elif (prot == 'P2P'):
            protocol_classes_num.append(5)
        elif (prot == 'VIDEO'):
            protocol_classes_num.append(6)
        elif (prot == 'VOIP'):
            protocol_classes_num.append(7)
    return protocol_data, protocol_classes_num