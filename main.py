from scipy.io import arff
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def loadDataset(filename):
    # caricamento dei dati da dataset
    arffData = arff.loadarff(filename)
    cleanData = arffData[0].__array__()
    # manipolo il dataset convertendolo in liste
    cleanDataList = [list(row) for row in cleanData]

    # la classi le voglio in utf-8 e non binarie
    for row in cleanDataList:
        row[-1] = row[-1].decode('utf-8')

    dataset = [cleanDataList[i][0:22] for i in range(len(cleanDataList))]
    classes = [cleanDataList[i][23] for i in range(len(cleanDataList))]
    return dataset, classes


# X, Y = loadDataset('datasets/scenario-a/SelectedFeatures-15s-TOR-NonTOR.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-15s-TOR-NonTOR.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-15s-TOR-NonTOR-15.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-15s-TOR-NonTOR-85.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-30s-TORNonTOR.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-30s-TORNonTOR-15.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-30s-TORNonTOR-85.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-60s-TOR-NonTOR.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-60s-TOR-NonTOR-15.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-60s-TOR-NonTOR-85.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-120s-TOR-NonTOR.arff')
# X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-120s-TOR-NonTOR-15.arff')
X, Y = loadDataset('datasets/scenario-a/TimeBasedFeatures-120s-TOR-NonTOR-85.arff')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.90, random_state=42)

# allenamento della SVM
clf = svm.SVC(gamma='scale')
clf.fit(X_train, Y_train)

# predizione
Y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(Y_test, Y_pred)
print(conf_mat)
