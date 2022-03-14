import numpy as np
import pandas as pd
import string
from visualization import visualize_data
import sys

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



if __name__ == "__main__":
    dataLoading("fake_news/fake_news_train.csv")
    # dataLoading("fake_news/fake_news_val.csv")
    # dataLoading("fake_news/fake_news_test.csv")