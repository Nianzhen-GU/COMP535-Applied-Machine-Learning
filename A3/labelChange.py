import numpy as np


if __name__ == '__main__':
    label = [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    digit = np.argmax(label[:10])
    alpha = np.argmax(label[-26:])
    new_index = digit*26+alpha
    new_label = np.zeros(260)
    new_label[new_index] = 1

    print(int(0 % 26))

