import streamlit as st
import cv2
import numpy as np
import pandas as pd

# Load the pre-trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# names related to ids: example ==> Marcelo: id=1,  etc
names = pd.read_csv('data.csv')['Name'].tolist()

# Create a Streamlit app
st.title("Face Identifier")

# Upload an image
image = st.file_uploader("Upload an image")

# If an image is uploaded, predict the person's name
if image is not None:
    # Convert the image to a NumPy array
    # img = cv2.imread(image.read())
    # img = cv2.imdecode(image.getvalue(), cv2.IMREAD_COLOR)
    # img = cv2.imdecode(image.read(), cv2.IMREAD_COLOR)
    img = cv2.imdecode(np.frombuffer(image.getvalue(), np.uint8), cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(0.1 * img.shape[1]), int(0.1 * img.shape[0])),
       )

    # For each face in the image, predict the person's name
    for(x,y,w,h) in faces:

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
       
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        # Draw a rectangle around the face
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # Write the person's name on the image
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1) 

        # img[y:y+h,x:x+w] = cv2.cvtColor(img[y:y+h,x:x+w], cv2.COLOR_BGR2RGB) 
    
    # Show the image with the person's name
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)
