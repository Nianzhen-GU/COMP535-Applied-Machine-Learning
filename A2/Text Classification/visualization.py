import matplotlib.pyplot as plt
import nltk
from nltk import tokenize
import seaborn as sns 
import pandas as pd
import sys

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


def visualize_val(val1, val2, C):
    plt.title('Valid accuracy between different penalty and C')
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy')
    plt.plot(C, val1, marker = 'o', markersize = 3)
    plt.plot(C, val2, marker = 'o', markersize = 3)
    plt.legend(['l1', 'l2'])
    plt.show()

