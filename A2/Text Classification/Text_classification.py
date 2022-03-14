import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from dataset import dataLoading
from visualization import visualize_val
import sys

print("========> Loading data")
X_train, y_train = dataLoading("fake_news/fake_news_train.csv")
X_test, y_test = dataLoading("fake_news/fake_news_test.csv")
X_val, y_val = dataLoading("fake_news/fake_news_val.csv")


# Find the best model
print("========> Validation")
best_score = 0.0
val_accuracy_l1 = []
val_accuracy_l2 = []
val_accuracy = []
C_list = range(1, 21)
for penalty in ['l1','l2']:
    for C in C_list:
        pipe = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('model', LogisticRegression(penalty = penalty, C = C, solver = "liblinear"))])
        val_model = pipe.fit(X_train, y_train.ravel())          
        score = val_model.score(X_val, y_val.ravel()) 
        val_accuracy.append(score)  
        if score > best_score:           
            best_score = score
            best_parameters = {'penalty': penalty, 'C':C}
        print("current score: %.2f"%(score))
        print("best score: %.2f"%(best_score))
        print("best parameter:{}".format(best_parameters))
        print("========================================")

val_accuracy_l1 = val_accuracy[:20]
val_accuracy_l2 = val_accuracy[20:40]

# Cross validation visualization
visualize_val(val_accuracy_l1, val_accuracy_l2, C_list)

print("========> Testing")
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression(**best_parameters, solver="liblinear"))])

# Fitting the model
model = pipe.fit(X_train, y_train.ravel())

# Accuracy
prediction = model.predict(X_test)
print(prediction)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))
