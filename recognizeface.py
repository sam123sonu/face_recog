# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:11:04 2020

@author: sambit mohapatra
"""

import cv2
import urllib
import numpy as np
from tensorflow.keras.models import load_model

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = load_model("FACE-DETECT.h5")

url = "http://100.91.81.62:8080/shot.jpg"

def get_pred_label(pred):
    labels = ['Arnab','Ashutosh','Durga','Malay','sambit']
    return labels[pred]

def preprocess(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(100,100))
    img = img.reshape(1,100,100,1)
    img = img/255
    return img


ret = True
while ret:
    img_url = urllib.request.urlopen(url)
    image = np.array(bytearray(img_url.read()),np.uint8)
    frame = cv2.imdecode(image,-1)
    
    faces = classifier.detectMultiScale(frame,1.5,5)
    
    for x,y,w,h in faces:
        face = frame[y:y+h,x:x+w]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        cv2.putText(frame,
                    get_pred_label(model.predict_classes(preprocess(face))[0]),
                    (200,500),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
        print(get_pred_label(model.predict_classes(preprocess(face))[0]))
    cv2.imshow("cap",frame)
    
    if cv2.waitKey(30) == ord("q"):
        break
    
cv2.destroyAllWindows()

