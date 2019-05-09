## GROUP 

import numpy as np
import cv2


#### GROUP5 code ####

cap = cv2.VideoCapture(0)
framesize = 300 # squared
grayscale = False

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read() 

    # Our operations on the frame come here
    if grayscale == True:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hight, width, depth = frame.shape
    crop_pixel = int((width - hight)/2) # crop square
    cropped_frame = frame[:, crop_pixel:width-crop_pixel]
    resized_frame = cv2.resize(cropped_frame, (framesize, framesize)) 

    #print(frame.shape, " ", cropped_frame.shape, " ", resized_frame.shape)

    # Display the resulting frame
    cv2.imshow('frame',resized_frame)
    if cv2.waitKey(1) == 27: 

        break  # esc to quit

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#### END of code ####