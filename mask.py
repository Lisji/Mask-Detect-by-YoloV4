import numpy as np
import cv2
import os

#影像辨識功能
def detect(image):

    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layer)

    boxes = []
    confidences = []
    classIDs = []


    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.5:

                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
      
        for i in idxs.flatten():

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1, lineType=cv2.LINE_AA)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            if (text[0:7]== 'without'):
                for i in range(3):
                   duration = 0.1  # seconds
                   freq = 440  # Hz
                   os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, lineType=cv2.LINE_AA)
    cv2.imshow("mask detecting...", image)
    # cv2.waitKey(0)
    


LABELS = open("mask.names").read().strip().split("\n")
np.random.seed(666)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
net = cv2.dnn.readNet('yolov4-mask.cfg', 'yolov4-mask.weights')
layer = net.getUnconnectedOutLayersNames()


# #圖片讀取區塊
# img = cv2.imread('1.jpg')
# detect(img)


# 攝影機讀取區塊
cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()
    detect(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

