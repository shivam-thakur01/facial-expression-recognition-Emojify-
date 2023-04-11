from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

output= {
    'Happy':r"C:\Users\satya\OneDrive\Documents\codechef_contest\ML Project-shareble\ML Project\OutPut emoji\happy.jpg",
    'Angry':r"C:\Users\satya\OneDrive\Documents\codechef_contest\ML Project-shareble\ML Project\OutPut emoji\angry.jpg",
    'Sad':r"C:\Users\satya\OneDrive\Documents\codechef_contest\ML Project-shareble\ML Project\OutPut emoji\sad.jpg",
    'Neutral':r"C:\Users\satya\OneDrive\Documents\codechef_contest\ML Project-shareble\ML Project\OutPut emoji\neutral.jpg",
    'Surprise':r"C:\Users\satya\OneDrive\Documents\codechef_contest\ML Project-shareble\ML Project\OutPut emoji\surprise.jpg",
    'Fear': r"C:\Users\satya\OneDrive\Documents\codechef_contest\ML Project-shareble\ML Project\OutPut emoji\fear.jpg",
    'Disgust':r"C:\Users\satya\OneDrive\Documents\codechef_contest\ML Project-shareble\ML Project\OutPut emoji\disgust.jpg"
}


face_classifier = cv2.CascadeClassifier(r"C:\Users\satya\OneDrive\Documents\codechef_contest\ML Project-shareble\ML Project\haarcascade_frontalface_default.xml")
classifier =load_model(r"C:\Users\satya\OneDrive\Documents\codechef_contest\ML Project-shareble\ML Project\classifier_model.h5")

class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# For reading image from web camera
# cap = cv2.VideoCapture(0)

# For reading image from video
cap = cv2.VideoCapture(r"C:\Users\satya\Downloads\The 7 basic emotions - Do you recognise all facial expressions_.mp4")   



img2=cv2.resize(cv2.imread(output['Sad']),(640,480))
while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_AREA)
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            img2=cv2.resize(cv2.imread(output[label]),(640,480))
        else:
            cv2.putText(frame,'No Face Found',(640,480),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    img1=frame
    
    Hori = np.concatenate((img1, img2), axis=1)
    # Verti = np.concatenate((img1, img2), axis=0)
 
    cv2.imshow('Emoji as per your expression', Hori)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
