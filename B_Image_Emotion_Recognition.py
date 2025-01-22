
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
 

detection_model_path = 'haarcascade_frontalface_default.xml'   
emotion_model_path = '_mini_XCEPTION.96-0.64.hdf5'      

img_path = '5.jpg'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


orig_frame = cv2.imread(img_path)
frame = cv2.imread(img_path, 0)
faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)

if len(faces):
    faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)



print('-----------------------------Probabilistic results-------------------------------')
print('angry:    ',preds[0])
print('disgust:  ',preds[1])
print('scared:   ',preds[2])
print('happy:    ',preds[3])
print('sad:      ',preds[4])
print('surprised:',preds[5])
print('neutral:  ',preds[6])
print('-----------------------------Final Decision-------------------------------')
print('Most likely this person is:',label)

plt.imshow(orig_frame)
plt.show()


