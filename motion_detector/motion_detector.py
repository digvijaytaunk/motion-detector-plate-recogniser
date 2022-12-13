import cv2


import winsound

cam = cv2.VideoCapture(0)


while cam.isOpened():
    ret, frame1 = cam.read() # static frame
    ret, frame2 = cam.read()  # moving frame to compare with frame1

    diff = cv2.absdiff(frame1, frame2)  # diff between frame1 & frame2
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)  # convert diff into gray shade

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=5)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    for c in contours:
        print(cv2.contourArea(c))
        if cv2.contourArea(c) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        winsound.PlaySound('alert.wav', winsound.SND_ASYNC)

    if cv2.waitKey(10) == ord('q'):
        break

    cv2.imshow('Motion detector', frame1)



