from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from util import load_dataset_time_based_b


# caricamento dei dati
X, Y = load_dataset_time_based_b('datasets/scenario-b/TimeBasedFeatures-120s-layer2-85.arff')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

# allenamentX, Y = loadDataset2('datasets/scenario-b/merged_5s.csv', -1)o della SVM
clf = svm.SVC(gamma='scale')
clf.fit(X_train, Y_train)

# predizione
Y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(Y_test, Y_pred)
class_rep = classification_report(Y_test, Y_pred)

# risultati
print(conf_mat)
print(class_rep)
