from numpy import mean
from numpy import std
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from keras.models import Model, load_model
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization, Dropout,GaussianNoise
from keras.layers import Flatten
import tensorflow as tf
from keras.initializers import glorot_normal, RandomNormal, Zeros
import time
from keras import regularizers, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback, EarlyStopping, LambdaCallback

loaded_model = tf.keras.models.load_model('model112420.h5')

with open('images_ul.pkl', 'rb') as f:
    data_test = pickle.load(f)

data_test= data_test.reshape((data_test.shape[0], 56, 56, 1))
data_test= data_test.astype('float32')
data_test = data_test / 255.0

prediction = loaded_model.predict(data_test)
def toNormalLabel(prediction):
    string_list = []
    for i in range(len(prediction)):
        text = "000000000000000000000000000000000000"
        s=list(text)
        a = np.argmax(prediction[i])
        alpha = int(a % 26)
        digit = int((a-alpha) / 26)
        s[digit]= "1"
        s[10+alpha] = "1"
        st = "".join(s)
        string_list.append(st)
    return string_list
    
output = toNormalLabel(prediction)  
import pandas
df = pandas.DataFrame(data={"Category": output})
df["Category"] = df["Category"].apply(str) + '\t'
df
df.to_csv("./submission3.csv",index=True)