import numpy as np
import os
import csv

# calcolo indice della colonna per un protocollo dentro
# la matrice di confusione.
def get_class_col(protocol, tor=True):
    '''
    Pari    : 0,2,4,6,8,.... -> NON TOR
    Dispari : 1,3,5,7,9,.... -> TOR
    '''
    datasets = [
        'audio',
        'browsing',
        'chat',
        'email',
        'file',
        'p2p',
        'video',
        'voip'
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
    pos = datasets.index(protocol)
    return replace_labels[pos][0] if tor is False else replace_labels[pos][1]


def get_class_precision(classid, confmat):
    tp = confmat[classid, classid]
    return round(tp / sum(confmat[:, classid]), 4)

def get_class_recall(classid, confmat):
    tp = confmat[classid, classid]
    return round(tp / sum(confmat[classid, :]), 4)

def get_class_f1score(classid, confmat):
    precision = get_class_precision(classid, confmat)
    recall = get_class_recall(classid, confmat)
    return 2 * (precision * recall) / (precision + recall)

def load_confmat_deep(samples, layers=10):
    rootpath = os.path.join('results\\csv\\ffnn_C\\', str(samples), 'avg')
    filename = 'conf_mat_avg-' + str(layers) + '.csv'
    return np.genfromtxt(os.path.join(rootpath, filename), delimiter=',')

def load_confmat_baseline(samples, model, scenario):
    rootpath = os.path.join('results\\csv\\other_C\\', scenario, str(samples), 'avg')
    filename = 'conf_mat_avg-' + model + '.csv'
    return np.genfromtxt(os.path.join(rootpath, filename), delimiter=',')


# routine principale
protocols = ['audio', 'browsing', 'chat', 'email', 'file', 'p2p', 'video', 'voip']
istor = [True, False]
samples = [1000, 5000, 10000, 25000]
models = ['bayes', 'log_reg', 'rnd_for']
layers = [5, 10, 20, 25]
scenarios = ['original', 'convolution']
all_results_ffnn = []
all_results_other = []
csv_headers = ['protocol', 'tor_precision', 'tor_recall', 'not_tor_precision', 'not_tor_recall']
csv.register_dialect('myDialect', delimiter=';', quoting=csv.QUOTE_NONE)

# ffnn
for smp in samples:
    store_base_root = 'results\\csv\\ffnn_C\\' + str(smp) + '\\tables\\'
    if not os.path.exists(store_base_root):
        os.makedirs(store_base_root)
    for layer in layers:
        result_filename = 'table-layer-' + str(layer) + '.csv'
        confmat = load_confmat_deep(smp, layer)
        with open(os.path.join(store_base_root, result_filename), 'wb') as result_file:
            csv_writer = csv.DictWriter(result_file, fieldnames=csv_headers, dialect='myDialect')
            csv_writer.writeheader()
            rows_data = []
            for prot in protocols:
                row = {'protocol': prot}
                for bool in istor:
                    classid = get_class_col(prot, bool)
                    precision = get_class_precision(classid, confmat)
                    recall = get_class_recall(classid, confmat)
                    if bool is True:
                        row['tor_precision'] = precision
                        row['tor_recall'] = recall
                    else:
                        row['not_tor_precision'] = precision
                        row['not_tor_recall'] = recall
                        rows_data.append(row)
            csv_writer.writerows(rows_data)
            print(os.path.join(store_base_root, result_filename))
            result_file.close()


# other
for scn in scenarios:
    for smp in samples:
        store_base_root = 'results\\csv\\other_C\\' + scn + '\\' + str(smp) + '\\tables\\'
        if not os.path.exists(store_base_root):
            os.makedirs(store_base_root)
        for mod in models:
            result_filename = 'table-' + mod + '.csv'
            confmat = load_confmat_baseline(smp, mod, scn)
            with open(os.path.join(store_base_root, result_filename), 'wb') as result_file2:
                csv_writer = csv.DictWriter(result_file2, fieldnames=csv_headers, dialect='myDialect')
                csv_writer.writeheader()
                rows_data = []
                for prot in protocols:
                    row = {'protocol': prot}
                    for bool in istor:
                        classid = get_class_col(prot, bool)
                        confmat = load_confmat_baseline(smp, mod, 'original')
                        precision = get_class_precision(classid, confmat)
                        recall = get_class_recall(classid, confmat)
                        if bool is True:
                            row['tor_precision'] = precision
                            row['tor_recall'] = recall
                        else:
                            row['not_tor_precision'] = precision
                            row['not_tor_recall'] = recall
                            rows_data.append(row)
                csv_writer.writerows(rows_data)
                print(os.path.join(store_base_root, result_filename))
                result_file.close()


