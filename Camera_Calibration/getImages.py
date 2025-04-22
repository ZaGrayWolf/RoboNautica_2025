import cv2
import os
cap = cv2.VideoCapture(0)
num = 0
if not os.path.exists("Camera_Calibration/images"):
    os.makedirs("Camera_Calibration/images")

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        try:
            cv2.imwrite('Camera_Calibration/images/img' + str(num) + '.png', img)
            print("image saved!")
            num += 1
            print(num)
        except Exception as e:
            print("Error saving image:", e)
    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()