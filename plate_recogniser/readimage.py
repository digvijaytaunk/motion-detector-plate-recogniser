import os
import cv2
import numpy as np

xml_path = os.path.join(os.getcwd(), 'plate.xml')
image_path = os.path.join(os.getcwd(), 'plate1.jpg')

numberPlateCascade = cv2.CascadeClassifier(xml_path)
plat_detector = cv2.CascadeClassifier(xml_path)
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plates = plat_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))

for (x, y, w, h) in plates:
    cv2.putText(img, text='My detected License Plate', org=(x - 3, y - 3), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 0, 255),
                thickness=1, fontScale=0.6)
    img[y:y + h, x:x + w] = cv2.blur(img[y:y + h, x:x + w], ksize=(10, 10))
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('plates', img)
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
