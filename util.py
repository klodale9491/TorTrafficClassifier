from scipy.io import arff
from sklearn import preprocessing
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from scapy.all import *
from lib.Byte import Byte
import pandas as pd
import numpy as np
import csv as csv


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


# caricamento dei dataset di header di pacchetti ethernet raw
def load_raw_ethernet_header_data(file_pcap):
    print('Loading pcap file ...')
    packets = PcapReader(file_pcap)
    print('Loading done ...')
    nibble_decimal_dataset = []
    print('Analyzing packets')
    for pkt in packets:
        header_dump_str = chexdump(pkt, dump=True).split(', ')
        # padding nel caso la dimensione del header sia inferiore a 54 bytes
        if len(header_dump_str) < 54:
            for i in range(len(header_dump_str), 54 + 1):
                header_dump_str.append('0x00')
        header_dump_hex = [int(x, 16) for x in header_dump_str[0:54]]
        header_dump_nib = []
        for y in header_dump_hex:
            byte = Byte(y)
            header_dump_nib.append(int(byte.low_nibble, 2))
            header_dump_nib.append(int(byte.high_nibble, 2))
        nibble_decimal_dataset.append(header_dump_nib)
    return nibble_decimal_dataset


# salva i dati inerenti gli header dei pacchetti ethernet
def save_raw_ethernet_features(nibble_raw_dataset, file_pcap):
    with open(file_pcap + '.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in nibble_raw_dataset:
            csv_writer.writerow(row)
    csv_file.close()


# creazione di dei dataset csv di dati TOR/NON-TOR in base al tipo di protocollo
def create_raw_nibble_protocol_dataset(protocol='AUDIO-STREAMING', splits=10):
    split_count = 1
    file_tor = open('datasets/raw/tor/' + protocol + '/' + protocol + '_COMPLETE.csv', mode='r')
    file_not_tor = open('datasets/raw/nonTor/' + protocol + '/' + protocol + '_COMPLETE.csv', mode='r')
    lines_tor = len(file_tor.readlines())
    lines_not_tor = len(file_not_tor.readlines())
    # partizionamento del dataset csv
    tor_df = pd.read_csv('datasets/raw/tor/' + protocol + '/' + protocol + '_COMPLETE.csv', chunksize=int(lines_tor / splits), iterator=True)
    not_tor_df = pd.read_csv('datasets/raw/nonTor/' + protocol + '/' + protocol + '_COMPLETE.csv', chunksize=int(lines_not_tor / splits), iterator=True)
    tor_iterator = tor_df.__iter__()
    not_tor_iterator = not_tor_df.__iter__()
    curr_chunk_tor = tor_iterator.__next__()
    chunk_filenames = []
    while curr_chunk_tor is not None:
        curr_chunk_tor['class'] = 'tor'
        curr_chunk_not_tor = not_tor_iterator.__next__()
        curr_chunk_not_tor['class'] = 'nonTor'
        cat_tor_not_tor = shuffle(pd.concat([curr_chunk_tor, curr_chunk_not_tor], sort=True))
        chunk_filename = 'datasets/raw/mixed/'+protocol+'.part'+str(split_count)+'.csv'
        print('Writing ' + protocol+'.part'+str(split_count)+'.csv' + ' on HDD')
        cat_tor_not_tor.to_csv(chunk_filename)
        chunk_filenames.append(chunk_filename)
        try:
            curr_chunk_tor = tor_iterator.__next__()
            split_count += 1
        except Exception:
            break


# carica un dataset di nibble in formato CSV
def load_raw_nibble_dataset(protocol='AUDIO-STREAMING'):
    # preparazione della struttura dati minimale in termini di memoria per il dataset
    col_dtypes = {}
    for i in range(0, 108):
        col_dtypes[i] = np.uint8
    col_dtypes[108] = str
    df = pd.read_csv('datasets/raw/mixed/' + protocol + '.csv', sep=",", header=None, dtype=col_dtypes)
    # shuffling del dataset
    df = df.reindex(np.random.permutation(df.index))
    # codifica delle classi in valori numerici
    data = np.asarray(df[df.columns[0:-1]])
    label_encoder = preprocessing.LabelEncoder().fit(['tor', 'nonTor'])
    encoded_classes = label_encoder.transform(np.asarray(df[108]))
    return data, encoded_classes


# carica il merge di una lista di datasets
def load_merged_nibble_datasets(protocols, samples=10, classification='binary', n_cols=108, type_col=np.uint8, ds_type='original'):
    # preparazione della struttura dati minimale in termini di memoria per il dataset
    col_dtypes = {}
    for i in range(0, n_cols):
        col_dtypes[i] = type_col
    col_dtypes[n_cols] = str
    all_data = np.asarray([])
    all_classes = np.asarray([])
    # caricamento dei dati
    for protocol in protocols:
        print('Loading ' + protocol + ' data ....')
        # 1 - caricamento del dataset raw
        if ds_type == 'original':
            df = pd.read_csv('datasets/csv/mix/samples/original/' + str(samples) + '/' + classification + '/' + protocol + '.csv', sep=",", header=None, dtype=col_dtypes)
        # 2 - caricamento del dataset raw con features estratte tramite rete convolutiva
        if ds_type == 'convolution':
            df = pd.read_csv('datasets/csv/mix/samples/original/' + str(samples) + '/' + classification + '/' + protocol + '.csv', sep=",", header=None, dtype=col_dtypes)
        # 3 - caricamento dateset feature estratte con rete convolutiva e successiva PCA
        if ds_type == 'pca_convolution':
            print('Not implemented yet')
            return
        # shuffling del dataset e codifica delle classi in valori numerici
        print('Shuffling ' + protocol + ' data ....')
        df = df.reindex(np.random.permutation(df.index))
        data = np.asarray(df[df.columns[0:-1]])
        classes = np.asarray(df[n_cols])
        if all_data.size == 0 and all_classes.size == 0:
            all_data = data
            all_classes = classes
        else:
            all_data = np.concatenate((all_data, data))
            all_classes = np.concatenate((all_classes, classes))
    if classification == 'binary':
        label_encoder = preprocessing.LabelEncoder().fit(['tor', 'nonTor'])
        all_classes = label_encoder.transform(all_classes)
    return all_data, all_classes


'''
Funzione per effettaure il caricamento di dataset di grosse dimensioni
effettudo il mixing di questo al fine di ottenere un unico dataset
come compososizione randomica di quelli elementari
'''
def load_merged_nibble_chunked_datasets(protocols, samples=100000, classification='label', n_cols=108, type_col=np.uint8, ds_type='RAW_MIXED', chunks=10):
    # preparazione della struttura dati minimale in termini di memoria per il dataset
    col_dtypes = {}
    for i in range(0, n_cols):
        col_dtypes[i] = type_col
    col_dtypes[n_cols] = str
    datasets = {}
    # effettuiamo il caricamento di tutti i dataset, caricando solo i chunk in memoria
    for protocol in protocols:
        print('Loading ' + protocol + ' data ....')
        # 1 - caricamento del dataset raw
        if ds_type == 'RAW_MIXED':
            path = 'datasets/raw/mixed/' + str(samples) + '_samples/' + classification + '/' + protocol + '.csv'
        # 2 - caricamento del dataset raw con features estratte tramite rete convolutiva
        if ds_type == 'RAW_MIXED_CONV1D':
            path = 'datasets/raw/mixed_conv1d/' + str(samples) + '_samples/' + classification + '/' + protocol + '.csv'
        datasets[protocol] = pd.read_csv(path, sep=",", header=None, dtype=col_dtypes, chunksize=samples / chunks)
    return datasets


'''
Estrazione e merge dei chunk correnti di ogni protocollo
shuffling di questi e return di dati e classi
'''
def split_and_merge_chunk_data_class(chunk_ptrs, protocols, n_cols=1728):
    all_chunk = np.asarray([])
    for p in protocols:
        try:
            df = chunk_ptrs[p].get_chunk()
        except Exception:
            return None, None # ho finito non ci sono più chunks
        if all_chunk.size == 0:
            all_chunk = df
        else:
            all_chunk = np.concatenate((all_chunk, df))
    np.random.shuffle(all_chunk)
    all_chunk_data = np.asarray(all_chunk[:, range(0, n_cols)])
    all_chunk_classes = np.asarray(all_chunk[:, n_cols])
    return all_chunk_data, all_chunk_classes



# funzione che effettua UNDERSAMPLIG su entrambi i dataset
def undersample_dataset(x, y):
    rus = RandomUnderSampler(random_state=0, replacement=True)
    return rus.fit_resample(x, y)


# funzione che effettua OVERSAMPLIG su entrambi i dataset
def oversample_dataset(x, y):
    ros = RandomOverSampler(random_state=0)
    return ros.fit_resample(x, y)
