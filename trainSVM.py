from os.path import join

import cv2
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

import importData

model_dir = 'model'

def main():
    # Get train labeled data and shuffle
    train_data = importData.main()
    np.random.shuffle(train_data)
    print("Input data size:", np.shape(train_data))
    
    # To data
    n_samples = np.shape(train_data)[0]
    n_tests = int(n_samples*0.25)
    trX = train_data[:n_tests,:-1]
    trY = train_data[:n_tests,-1].astype(np.int32)
    teX = train_data[n_tests:,:-1]
    teY = train_data[n_tests:,-1].astype(np.int32)
    print("Train data size:", np.shape(trX))
    print("Label data size:", np.shape(trY))
    
    # Use scikit-learn library
#     C = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
    C = [100]
    kernel = 'rbf'
    tol = 1e-4
    max_iter = -1
    decision_function_shape = None
    for C in C:
#         gamma = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        gamma = [0.003]
        for gamma in gamma:
            clf = svm.SVC(C=C, kernel=kernel, gamma=gamma, 
                          tol=tol, max_iter=max_iter, 
                          decision_function_shape=decision_function_shape)
            clf.fit(trX, trY)
    
            # Prediction
            prediction = clf.predict(teX)
#             print(prediction)
            print('C = %f, gamma = %f, accuracy =' % (C, gamma), np.mean(prediction == teY))
    
    # Save trained model 
    joblib.dump(clf, join(model_dir, 'svm.pkl')) 

if __name__=='__main__':
    main()