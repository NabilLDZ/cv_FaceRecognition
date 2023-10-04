#Import Library
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
from PIL import Image
from keras.models import load_model
import numpy as np
from keras_facenet import FaceNet
from numpy import asarray
from numpy import expand_dims
import pickle


#Load FaceNet
HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = FaceNet()

#load model
myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
model = YOLO("best.pt")


#Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)


# Bounding Box
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
    )

# Face Detec and Face Recog
while True:
    # Frame Video
    _, framevideo = cap.read()
    result = model(framevideo, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)

    framevideo = box_annotator.annotate(
            scene=framevideo, 
            detections=detections
        ) 


    wajah = HaarCascade.detectMultiScale(framevideo,1.1,4)

    if len(wajah)>0:
        x1, y1, width, height = wajah[0]        
    else:
        x1, y1, width, height = 1, 1, 10, 10
    
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    
    gbr = cv2.cvtColor(framevideo, cv2.COLOR_BGR2RGB)
    gbr = Image.fromarray(gbr)                  
    gbr_array = asarray(gbr)
    
    face = gbr_array[y1:y2, x1:x2]                        
    
    face = Image.fromarray(face)                       
    face = face.resize((160,160))
    face = asarray(face)
    
    
    face = expand_dims(face, axis=0)
    signature = MyFaceNet.embeddings(face)
    
    min_dist=100
    identity=' '
    for key, value in database.items() :
        dist = np.linalg.norm(value-signature)
        if dist < min_dist:
            min_dist = dist
            identity = key


    # If If Face is detected confidence 50%, Then the face will be recognized
    confident = detections.confidence
    if confident.size > 0:
        confident = confident[:1]
        confident_float = float(confident)
        if confident_float > 0.50:
            cv2.putText(framevideo,identity, (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face Recognatition', framevideo)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
        
cv2.destroyAllWindows()
cap.release()