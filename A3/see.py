import pickle
import cv2

if __name__ == '__main__':

    with open('comp551/images_test.pkl', 'rb') as f:
            img = pickle.load(f)

    for i in range(100):
        print(i)
        cv2.imshow('image.png', img[i])
        cv2.waitKey(0)
        
