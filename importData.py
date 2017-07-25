# Import and return all the data as labeled train data as an array: [X, y]
# Class types are: nil: 0, cross: 1, circle: 2, triangle: 3, marker: 4
import os
from os.path import join
import imghdr

import cv2
import numpy as np

data_dir = 'data'
# class_dir = ['nil', 'cross']
class_dir = ['nil', 'cross', 'circle', 'triangle', 'marker']
model_dir = 'model'
debug = True

model_dir = 'model'
HOG_file = 'hog.xml'

def defaultHOG(): 
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    useSignedGradients = True
     
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, 
                            nbins, derivAperture, winSigma, histogramNormType, 
                            L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)
    return hog
    
def saveHOG(hog):
    hog.save(os.path.join(model_dir, HOG_file))
    
def loadHOG(hog_file):
    return cv2.HOGDescriptor(hog_file)

def read_absolute_image_path(path):
    absolute_path = [os.path.join(path, rel_path) for rel_path in os.listdir(path)]
    image_list = [file for file in absolute_path if os.path.isfile(file) and imghdr.what(file)]
    return image_list

def getImgVariance(img):
    rows, cols, _ = img.shape
    varImg = []
    for angle in range(0, 180, 5):
        M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        varImg.append(dst)
    return varImg

def getHOGDescriptor(data):
    # Get HOG descriptor, either create anew or load from saved
    hog = defaultHOG()
#     hog = loadHog(join(model_dir, HOG_file)
    saveHOG(hog) 

    # Find HOG for each sample
    hist = [hog.compute(img) for img in data]
    hist = np.array(hist)[:,:,0]
#     print("Histogram size of %s:" % item, np.shape(hist))
    return hist

def getGeometricDescriptor(data):
    print(np.shape(data))
    cv2.waitKey(-1)

def getLabel(item):
    return class_dir.index(item)

def getData(item):
    # Get images
    image_list = read_absolute_image_path(join(data_dir, item))
    input_file = [cv2.resize(cv2.imread(file), (64, 64)) for file in image_list]
    
    # Get more files using image rotation
    raw_data = [getImgVariance(img) for img in input_file]
    arranged_data = [item for sublist in raw_data for item in sublist]
    
    # Get feature data
    feature = getHOGDescriptor(arranged_data)
#     feature = getGeometricDescriptor(arranged_data)
#     print(np.shape(arranged_data))
    
    # Get label
    item_label = getLabel(item)
    
    # Add label as the final column in data
    n_samples = np.shape(arranged_data)[0]
    output = np.c_[feature, np.ones((n_samples, 1)) * item_label]
    
    # Show images in debug mode
    if debug:    
        train_data = []   
        for file in raw_data:
            train_data.append(np.hstack(file for file in file))
            merged_image = np.vstack(file for file in train_data)
        cv2.imshow('image', merged_image)
        cv2.waitKey(-1)
    
#     print("Labeled data size:", np.shape(output))
    return output
    
def main():
    return np.vstack(getData(item) for item in class_dir)

if __name__=='__main__':
    main()