from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from util import load_dataset_time_based_a
from util import load_dataset_csv_a
from util import oversample_dataset_a
import numpy as np

# caricamento datasets
# X, Y = load_dataset_csv_a('../datasets/scenario-a/merged_5s_clean.csv')
X, Y = load_dataset_time_based_a('../datasets/scenario-a/TimeBasedFeatures-30s-TOR-NonTOR.arff')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=0)
X_train, Y_train = oversample_dataset_a(X_train, Y_train)

# classificatore bayes naive con distribuzione gaussiana delle features, mi da la probabilità di ognuna delle due classi
gnb = GaussianNB()

# alleno il modello
gnb.fit(X_train, Y_train)

# predizione
Y_pred = gnb.predict(X_test)

# risultati
conf_mat = confusion_matrix(Y_test, Y_pred)
accuracy = accuracy_score(Y_test, Y_pred)
class_rep = classification_report(Y_test, Y_pred)
print('confusion_matrix : ')
print(conf_mat)
print('accuracy : ')
print(accuracy)
print(class_rep)

# 10 fold cross-validation
accuracy_scores = cross_val_score(gnb, X, Y, cv=10, scoring='accuracy')
recall_scores = cross_val_score(gnb, X, Y, cv=10, scoring='recall')
precision_scores = cross_val_score(gnb, X, Y, cv=10, scoring='precision')
f1_scores = cross_val_score(gnb, X, Y, cv=10, scoring='f1')

print('Scores after 10-folf cross-validation : ')
print(['Accuracy : ', np.mean(accuracy_scores)])
print(['Recall : ', np.mean(recall_scores)])
print(['Precision : ', np.mean(precision_scores)])
print(['F1 : ', np.mean(f1_scores)])
