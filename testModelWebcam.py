from os.path import join
import sys

import cv2
import numpy as np
from sklearn.externals import joblib

import importData

model_dir = 'model'

class objectDetection(object):
    # Image-wise parameters
    kernel_blur = np.ones((5,5),np.float32)/25
    kernel_morph = np.ones((2,2), np.uint8)
    min_area_ratio = 0.01
    red_range = [((0, 100, 70), (25, 255, 255)), ((163, 100, 70), (179, 255, 255))]
    green_range = ((40, 100, 70), (90, 255, 255))
    blue_range = ((100, 100, 100), (130, 230, 230))
    
    # Load trained model
    hog = importData.defaultHOG()
    clf = joblib.load(join(model_dir, 'svm.pkl'))
    
    def __init__(self, frame):
        self.src = frame.copy()
        self.image_height, self.image_width, self.channels = np.shape(frame) # Get frame dimension    
        self.min_area = self.image_height*self.image_width*self.min_area_ratio
        self.hsv = cv2.cvtColor(self.src, cv2.COLOR_BGR2HSV)
        self.mask = self.imgProcess('green')
        cv2.imshow('src', self.src)
    
    def imgProcess(self, color):
        binary_image = None
        if color == 'red':
            lower_red = cv2.inRange(self.hsv, self.red_range[0][0], self.red_range[0][1])
            upper_red = cv2.inRange(self.hsv, self.red_range[1][0], self.red_range[1][1])
            binary_image = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)
        elif color == 'green':
            binary_image = cv2.inRange(self.hsv, self.green_range[0], self.green_range[1])
        elif color == 'blue':
            binary_image = cv2.inRange(self.hsv, self.blue_range[0], self.blue_range[1])
            
        # Smooth the image
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, self.kernel_morph)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, self.kernel_morph)
        
        _, contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process each blob found
        mask = []
        for idx, cnt in enumerate(contours):
            cnt_area = cv2.contourArea(cnt)
               
            # SKip small blobs
            if cnt_area < self.min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(self.src, (x, y), (x + w, y + h), (0, 255, 255), 2)
            img_roi = cv2.resize(self.src[y:y+h, x:x+w], (64, 64))
            
            # Predict
            hist = self.hog.compute(img_roi)
            hist = np.array(hist.T)
            verdict = self.clf.predict(hist)
            print(np.shape(hist), verdict)
            if verdict != 0:
                cv2.rectangle(self.src, (x, y), (x + w, y + h), (0, 0, 255), 3)
            
        # Show output on screen
        cv2.imshow('bin', binary_image)
        
        return mask
        
def main(argv):    
    # Start capturing
    print("Press q for quit. Press p for pause.")
    cap = cv2.VideoCapture(argv)
    while cap.isOpened():
        # User control
        userInput = cv2.waitKey(20) & 0xFF
        if userInput == ord('q'):
            break
        elif userInput == ord('p'):
            cv2.waitKey(0)
            
        # Get frame from video
        retval, frame = cap.read()
        if retval:
            # Do testing
            objectDetection(frame)
        else:
            print("Unable to obtain image from camera.")

if __name__=='__main__':
    if len(sys.argv) < 2:
        print('No input file indicated. Using default camera.')
        main(0)
    else:
        main(sys.argv[1])