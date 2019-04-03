from scipy.io import arff
import numpy as np
import pandas as pd


def loadDataset(filename):
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


'''
Permette il caricamento del dataset csv filtrato 
in base al protocollo che ci interessa : Email, FTP, P2P etc ....
'''
def loadDataset2(filename, protocol):
    dataframe = pd.read_csv(filename, ',')
    columns = dataframe.columns[5:]
    data = np.asarray(dataframe[columns])
    protocol_data = data[data[:, 0] == protocol]
    protocol_classes = data[:][-1]
    return protocol_data, protocol_classes