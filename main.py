import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    #Reading the frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    # Converting the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Defining the range of colors to detect (red)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Mask the frame to only select red colors. Mask will select colors only in this range
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Apply morphological transformations to remove small blobs,remove noise and improve the shape of the arrow.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    corners = cv2.goodFeaturesToTrack(mask,20,0.1,10)

    # Find contours in the mask
    # Contours -  Curve joining all the continuous points (along the boundary), having same color or intensity.
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # arguments - first one is source image, second is contour retrieval mode, third is contour approximation method
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Loop over the contours
    for i in contours:

        # Check the contour size
        if cv2.contourArea(i) <500:  #Ends the iteration for small areas to avoid false positives
            continue
        if len(corners) >=7 :
        # Draw a bounding box around the contour using boundingRect
        # create an approximate rectangle along with the image
            x, y, w, h = cv2.boundingRect(i)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  #arg- image, starting point,ending point, color,thickness

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('0'):  #the key for closing webcam
        break

cap.release()
cv2.destroyAllWindows()