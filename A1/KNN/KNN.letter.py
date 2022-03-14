import numpy as np                                
from dataset import letter_data
from utils import cross_validation, test
from visualization import result

#it is important to set the seed for reproducibility as it initializes the random number generator
np.random.seed(1234)
print('===> Load data')
train_file = 'letter-recognition.data.csv'
x_train, y_train, x_test, y_test = letter_data(train_file)

# Use cross validation to determine the best k
t_acc = []
v_acc = []
k_list = range(1, 11)

for i in range(1, 4):
    x_Train = x_train[0:(len(x_train)//3)*i]
    y_Train = y_train[0:(len(x_train)//3)*i]
    print(i,'/3 of whole data: ' )
    best_k, train_acc, valid_acc = cross_validation(k_list, x_Train, y_Train)
    t_acc.append(train_acc)
    v_acc.append(valid_acc)

# Use the best K to test
test(best_k, x_train, y_train, x_test, y_test)

# Visulation of the result
result(k_list, t_acc, v_acc)
