from keras.backend import dropout
from keras.layers.core import Dropout
from keras.layers.noise import GaussianNoise
from keras.layers.normalization.batch_normalization import BatchNormalization
from numpy import mean
from numpy import std
import numpy as np
import cv2
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

def image_preprocessing(img, height=128, width=128):
    # enlarge image
    dimension = (height, width)
    img = cv2.resize(img, dimension, interpolation=cv2.INTER_LINEAR)
    # denoise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img[..., np.newaxis]
    return img

def load_data():
    with open('comp551/images_l.pkl', 'rb') as f:
        trainX = pickle.load(f)

    with open('comp551/labels_l.pkl', 'rb') as f:
        label = pickle.load(f)

	trainX= np.array([image_preprocessing(x) for x in trainX])

    trainX = trainX.reshape((trainX.shape[0], 56, 56, 1))

    trainX= trainX.astype('float32')
    trainX = trainX / 255.0

    return trainX, label



# define cnn model
def define_model():
	model = Sequential()
	model.add(BatchNormalization(), input_shape=(128, 128, 1))
	model.add(GaussianNoise(0.1))
	model.add(Conv2D(32, (3, 3), activation='relu'), padding='same')
	model.add(Conv2D(32, (3, 3), activation='relu'), padding='same')
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))

	model.add(BatchNormalization())
	model.add(GaussianNoise(0.1))
	model.add(Conv2D(64, (3, 3), activation='relu'), padding='same')
	model.add(Conv2D(64, (3, 3), activation='relu'), padding='same')
	model.add(Conv2D(64, (3, 3), activation='relu'), padding='same')
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.2))

	model.add(BatchNormalization())
	model.add(GaussianNoise(0.1))
	model.add(Conv2D(128, (3, 3), activation='relu'), padding='same')
	model.add(Conv2D(128, (3, 3), activation='relu'), padding='same')
	model.add(Conv2D(128, (3, 3), activation='relu'), padding='same')
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.25))

	model.add(BatchNormalization())
	model.add(GaussianNoise(0.1))
	model.add(Conv2D(256, (3, 3), activation='relu'), padding='same')
	model.add(Conv2D(256, (3, 3), activation='relu'), padding='same')
	model.add(Conv2D(256, (3, 3), activation='relu'), padding='same')
	model.add(MaxPooling2D((2, 2)))
	model.add(Dropout(0.25))
	model.add(GaussianNoise(0.05))

	model.add(BatchNormalization())
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(GaussianNoise(0.05))
	model.add(Dense(260, activation='softmax'))
	# compile model
	#opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		print(model)
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		print("here")
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories


    # plot diagnostic learning curves
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	plt.show()
 
# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()
 
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, label = load_data()
	# evaluate model
	scores, histories = evaluate_model(trainX, label)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)
 

if __name__ == '__main__': 
    # entry point, run the test harness 
    run_test_harness()






