import cv2

def main(onscreen):
    print("Press q for quit. Press p for pause.")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # User control
        userInput = cv2.waitKey(30) & 0xFF
        if userInput == ord('q'):
            break
        elif userInput == ord('p'):
            cv2.waitKey(0)
            
        # Get frame from default device (webcam)
        retval, frame = cap.read()
        if retval and onscreen:
            cv2.imshow("frame", frame)
        else:
            print("Unable to obtain image from camera.")

if __name__=='__main__':
    main(True)