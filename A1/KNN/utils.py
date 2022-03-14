import numpy as np
#the output of plotting commands is displayed inline within frontends
#%matplotlib inline                                  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix   
import sys      


def test(best_k, x_train, y_train, x_test, y_test):
    # Apply the best_k to the test dataset
    print('===> Test')
    best_model = KNeighborsClassifier(n_neighbors = best_k)
    best_model.fit(x_train, y_train)
    test_pred = best_model.predict(x_test)

    acc = np.mean(test_pred == y_test)
    np.set_printoptions(linewidth=400)
    print(confusion_matrix(y_test, test_pred))
    print('The accuracy is ', f'{acc:.3f}')

def cross_validation(k_list, x_train, y_train):
    print('===> Cross validation')
    ks = k_list
    best_k = ks[0]
    best_score = 0
    l = len(x_train)//5

    train_1 = []
    train_2 = []
    train_3 = []
    train_4 = []
    train_5 = []
    valid_1 = []
    valid_2 = []
    valid_3 = []
    valid_4 = []
    valid_5 = []

    for i in range(l, len(x_train)):
        train_1.append(i)
    for i in range(l):
        valid_1.append(i)
    for i in range(l):
        train_2.append(i)
    for i in range(2*l, len(x_train)):
        train_2.append(i)
    for i in range(l, 2*l):
        valid_2.append(i)
    for i in range(2*l):
        train_3.append(i)
    for i in range(3*l, len(x_train)):
        train_3.append(i)
    for i in range(2*l, 3*l):
        valid_3.append(i)
    for i in range(3*l):
        train_4.append(i)
    for i in range(4*l, len(x_train)):
        train_4.append(i)
    for i in range(3*l, 4*l):
        valid_4.append(i)
    for i in range(4*l):
        train_5.append(i)
    for i in range(4*l, len(x_train)):
        valid_5.append(i)

    train_set = [train_1, train_2, train_3, train_4, train_5]
    valid_set = [valid_1, valid_2, valid_3, valid_4, valid_5]

    train_acc = []
    valid_acc = []

    for k in ks:
        train_score = 0
        valid_score = 0
        for i in range(5):
            knn = KNeighborsClassifier(n_neighbors = k)
            knn.fit(x_train.iloc[train_set[i]], y_train.iloc[train_set[i]])
            valid_score = valid_score + knn.score(x_train.iloc[valid_set[i]], y_train.iloc[valid_set[i]])
            train_score = train_score + knn.score(x_train.iloc[train_set[i]], y_train.iloc[train_set[i]])
        avg_valid = valid_score / 5
        avg_train = train_score / 5
        train_acc.append(avg_train)
        valid_acc.append(avg_valid)
        if avg_valid > best_score:
            best_k = k
            best_score = avg_valid
        print ("current best score is: %.3f"%best_score, "best k: %d"%best_k)
    print ("after cross validation, the final best k is: %d"%best_k)
    print(train_acc)
    print(valid_acc)

    return best_k, train_acc, valid_acc