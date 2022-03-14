import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from visualization import DataVisualization_adult, DataVisualization_letter
import sys

def adult_data(file):
    labels = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", 
        "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

    data = pd.read_csv(file, names = labels)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # check whether there are any data missing
    #print(data.isnull().any())
    #print(data.shape)

    # check whether there are any incorrect data 
    data_clean = data.replace(regex=[r'\?|\$'], value=np.nan)

    # drop the data that are incorrect
    adult_data = data_clean.dropna(how='any')

    # drop the useless dimention
    adult_data = adult_data.drop(['fnlwgt'], axis=1)
    adult_data = adult_data.drop(['education'], axis=1)
    adult_data = adult_data.drop(['capital-gain'], axis=1)
    adult_data = adult_data.drop(['capital-loss'], axis=1)

    # Undo command to see the visulation 
    DataVisualization_adult(adult_data)

    adult_data = pd.get_dummies(adult_data)
    adult_data.head()

    # labels = ["age", "workclass", "education-num", "marital-status", "occupation", 
    #     "relationship", "race", "sex", "hours-per-week", "native-country", "income"]
    
    # workclass_dic = {' State-gov': 0, ' Self-emp-not-inc': 1, ' Private': 2, ' Federal-gov': 3, ' Local-gov': 4, 
    #                     ' Self-emp-inc': 5, ' Without-pay': 6}
    # marital_status_dic = {' Never-married': 0, ' Married-civ-spouse': 1, ' Divorced': 2,
    #                         ' Married-spouse-absent': 3, ' Separated': 4, ' Married-AF-spouse': 5, ' Widowed': 6} 
    # occupation_dic = {' Adm-clerical': 0, ' Exec-managerial': 1, ' Handlers-cleaners': 2, ' Prof-specialty': 3,
    #                     ' Other-service': 4, ' Sales': 5, ' Transport-moving': 6, ' Farming-fishing': 7,
    #                     ' Machine-op-inspct': 8, ' Tech-support': 9, ' Craft-repair': 10, ' Protective-serv': 11, 
    #                     ' Armed-Forces': 12, ' Priv-house-serv': 13}
    # relationship_dic = {' Not-in-family': 0, ' Husband': 1, ' Wife': 2, ' Own-child': 3, ' Unmarried': 4,
    #                     ' Other-relative': 5}
    # race_dic = {' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4}
    # sex_dic = {' Male': 0, ' Female': 1}
    # native_country_dic = {' United-States': 0, ' Cuba': 1, ' Jamaica': 2, ' India': 3, ' Mexico': 4, ' Puerto-Rico': 5,
    #                         ' Honduras': 6, ' England': 7, ' Canada': 8, ' Germany': 9, ' Iran': 10, ' Philippines': 11,
    #                         ' Poland': 12, ' Columbia': 13, ' Cambodia': 14, ' Thailand': 15, ' Ecuador': 16, ' Laos': 17,
    #                         ' Taiwan': 18, ' Haiti': 19, ' Portugal': 20, ' Dominican-Republic': 21, ' El-Salvador': 22,
    #                         ' France': 23, ' Guatemala': 24, ' Italy': 25, ' China': 26, ' South': 27, ' Japan': 28, ' Yugoslavia': 29,
    #                         ' Peru': 30, ' Outlying-US(Guam-USVI-etc)': 31, ' Scotland': 32, ' Trinadad&Tobago': 33, 
    #                         ' Greece': 34, ' Nicaragua': 35, ' Vietnam': 36, ' Hong': 37, ' Ireland': 38, ' Hungary': 39,
    #                         ' Holand-Netherlands': 40}
    # income_dic = {' <=50K': 0, ' >50K': 1}

    # adult_data['workclass'] = adult_data['workclass'].map(workclass_dic)
    # adult_data['marital-status'] = adult_data['marital-status'].map(marital_status_dic)
    # adult_data['occupation'] = adult_data['occupation'].map(occupation_dic)
    # adult_data['relationship'] = adult_data['relationship'].map(relationship_dic)
    # adult_data['race'] = adult_data['race'].map(race_dic)
    # adult_data['sex'] = adult_data['sex'].map(sex_dic)
    # adult_data['native-country'] = adult_data['native-country'].map(native_country_dic)
    # adult_data['income'] = adult_data['income'].map(income_dic)

    # adult_data = adult_data.dropna(how='any')
    print(adult_data[0:10])

    #split the train and test sets
    attribute = adult_data.loc[:, 'age':'native-country_ Yugoslavia']

    result = adult_data['income_ >50K']

    return attribute, result

def letter_data(file):
    labels = ["lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", 
    "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]

    letter_data = pd.read_csv(file, names = labels)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Undo command to see the visulation 
    DataVisualization_letter(letter_data)

    letter_dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
                    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17,
                    'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25}

    letter_data['lettr'] = letter_data['lettr'].map(letter_dic)

    #split the train and test sets
    train = letter_data[0:16000]
    test = letter_data[16000:len(letter_data)]

    train_attribute = train[labels[1:17]]
    train_result = train[labels[0]]
    test_attribute = test[labels[1:17]]
    test_result = test[labels[0]]

    return train_attribute, train_result, test_attribute, test_result

if __name__ == '__main__' :
    #adult_data('adult.test.csv')
    adult_data('adult.data.csv')
    #letter_data('letter-recognition.data.csv')