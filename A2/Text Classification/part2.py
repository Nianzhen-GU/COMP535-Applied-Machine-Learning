import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import nltk
from nltk import tokenize
import seaborn as sns 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from dataset import dataLoading
from visualization import visualize_val

# Cleaning the data

def dataLoading(file):
    labels = ["text", "label"]
    data = pd.read_csv(file, names = labels)

    # remove the first line
    data = data.drop(0)
    data = data.reset_index(drop = True)

    # change all words to lowercase
    data['text'] = data['text'].apply(lambda x: x.lower())
   
    # remove punctuation
    data['text'] = data['text'].apply(punctuation_removal)

    # print(data[0:5])

    # data visualization
    visualize_data(data)


    x = data["text"]
    y = data["label"]

    print("X.shape: ", x.shape)
    print("y.shape: ", y.shape)

    return x, y

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


# Show the information of data

def visualize_data(data):
    # How many fake and real articles?
    print(data.groupby(['label'])['text'].count())
    data.groupby(['label'])['text'].count().plot(kind="bar")
    plt.show()

    # Most frequent words counter
    for x in [data[data["label"]=="0"], data[data["label"]=="1"]]:
        token_space = tokenize.WhitespaceTokenizer()
        all_words = ' '.join([text for text in x["text"]])
        token_phrase = token_space.tokenize(all_words)
        frequency = nltk.FreqDist(token_phrase)
        df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                    "Frequency": list(frequency.values())})
        df_frequency = df_frequency.nlargest(columns = "Frequency", n = 20)
        plt.figure(figsize=(12,8))
        ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
        ax.set(ylabel = "Count")
        plt.xticks(rotation='vertical')
        plt.show()

# Show the validation accuracy among different penalty and C

def visualize_val(val1, val2, C):
    plt.title('Valid accuracy between different penalty and C')
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy')
    plt.plot(C, val1, marker = 'o', markersize = 3)
    plt.plot(C, val2, marker = 'o', markersize = 3)
    plt.legend(['l1', 'l2'])
    plt.show()


# main part of Test Classification

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
